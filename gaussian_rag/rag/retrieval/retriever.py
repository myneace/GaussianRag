from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from gaussian_rag.core.types import QueryRepresentation, RetrievedChunk
from gaussian_rag.core.store import KnowledgeStore
from gaussian_rag.rag.retrieval.ann_index import MuAnnIndex
from gaussian_rag.rag.retrieval.diversity import diversity_filter
from gaussian_rag.rag.retrieval.manifold_ranker import (
    rerank_candidates,
    rerank_multi_sense,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class GaussianRetriever:
    store: KnowledgeStore
    ann_index: MuAnnIndex = field(default_factory=MuAnnIndex)

    def refresh(self) -> None:
        self.ann_index.build(self.store.values())

    def retrieve(
        self,
        query: QueryRepresentation,
        top_k: int = 5,
        metric: str = "elk",
        adaptive: bool = True,
        buffer_size: int = 2,
    ) -> list[RetrievedChunk]:
        fetch_k = None if adaptive else max(top_k * 5, 10)
        candidate_ids = self.ann_index.search(query.mu, top_k=fetch_k)
        candidates = self.store.get(candidate_ids)
        ranked = rerank_candidates(query, candidates, metric=metric)

        if adaptive and len(ranked) >= 2:
            from gaussian_rag.rag.retrieval.manifold_ranker import _ranking_score

            scores = [_ranking_score(c, metric) for c in ranked]
            gaps = [scores[i + 1] - scores[i] for i in range(len(scores) - 1)]
            if gaps:
                max_gap_idx = gaps.index(max(gaps))
                cutoff = max_gap_idx + 1 + buffer_size
                ranked = ranked[:cutoff]
        elif not adaptive:
            ranked = ranked[:top_k]

        return ranked

    def retrieve_with_diversity(
        self,
        query: QueryRepresentation,
        top_k: int = 5,
        diversity_threshold: float = 0.3,
        adaptive: bool = True,
        buffer_size: int = 2,
    ) -> list[RetrievedChunk]:
        ranked = self.retrieve(
            query, top_k=top_k, adaptive=adaptive, buffer_size=buffer_size
        )
        filter_k = None if adaptive else top_k
        diverse = diversity_filter(
            ranked, threshold=diversity_threshold, top_k=filter_k
        )
        for index, item in enumerate(diverse, start=1):
            item.rank = index
        return diverse

    # ------------------------------------------------------------------ #
    #  Legacy serial multi-sense (kept for backward compat)              #
    # ------------------------------------------------------------------ #
    def retrieve_multi_sense(
        self,
        senses: list[tuple[str, float, QueryRepresentation]],
        top_k: int = 5,
        metric: str = "elk",
        fusion: str = "interleave",
        adaptive: bool = True,
        buffer_size: int = 2,
    ) -> tuple[dict[str, list[RetrievedChunk]], list[RetrievedChunk]]:
        per_sense: list[tuple[float, list[RetrievedChunk]]] = []
        per_sense_ranked: dict[str, list[RetrievedChunk]] = {}
        weight_map = {sense_id: weight for sense_id, weight, _query in senses}
        order_map = {
            sense_id: index for index, (sense_id, _weight, _query) in enumerate(senses)
        }

        for sense_id, weight, query in senses:
            ranked = self.retrieve(
                query,
                top_k=top_k,
                metric=metric,
                adaptive=adaptive,
                buffer_size=buffer_size,
            )
            per_sense_ranked[sense_id] = ranked
            per_sense.append((weight, ranked))

        return self._fuse_sense_results(
            senses,
            per_sense_ranked,
            per_sense,
            top_k=top_k,
            adaptive=adaptive,
            fusion=fusion,
            weight_map=weight_map,
            order_map=order_map,
        )

    # ------------------------------------------------------------------ #
    #  MMASA: Mother-Agent Orchestrator (async parallel)                 #
    # ------------------------------------------------------------------ #

    async def _retrieve_node(
        self,
        sense_id: str,
        weight: float,
        query: QueryRepresentation,
        *,
        top_k: int,
        metric: str,
        adaptive: bool,
        buffer_size: int,
        executor: ThreadPoolExecutor,
    ) -> tuple[str, float, list[RetrievedChunk]]:
        """Worker Node: executes a single sense retrieval in a thread pool.

        Runs the CPU-bound retrieval off the event loop so asyncio can
        drive all senses concurrently.
        """
        loop = asyncio.get_running_loop()
        logger.info(
            f"[MMASA] Worker node dispatched: sense='{sense_id}' weight={weight:.3f}"
        )
        t0 = time.perf_counter()
        ranked = await loop.run_in_executor(
            executor,
            lambda: self.retrieve(
                query,
                top_k=top_k,
                metric=metric,
                adaptive=adaptive,
                buffer_size=buffer_size,
            ),
        )
        elapsed = time.perf_counter() - t0
        logger.info(
            f"[MMASA] Worker node done:       sense='{sense_id}' nodes={len(ranked)} elapsed={elapsed:.3f}s"
        )
        return sense_id, weight, ranked

    async def retrieve_multi_sense_async(
        self,
        senses: list[tuple[str, float, QueryRepresentation]],
        top_k: int = 5,
        metric: str = "elk",
        fusion: str = "interleave",
        adaptive: bool = True,
        buffer_size: int = 2,
    ) -> tuple[dict[str, list[RetrievedChunk]], list[RetrievedChunk]]:
        """Mother Agent: decomposes senses, dispatches worker nodes in parallel,
        then aggregates and synthesizes the fused result.

        Implements the MMASA workflow:
          Step 3 – Parallel Task Allocation  (asyncio.gather)
          Step 4 – Parallel Node Execution   (_retrieve_node per sense)
          Step 5 – Synchronization           (await all futures)
          Step 6 – Aggregation & Synthesis   (_fuse_sense_results)
        """
        logger.info(
            f"[MMASA] Mother Agent: decomposing {len(senses)} senses → parallel dispatch"
        )
        t_start = time.perf_counter()

        weight_map = {sense_id: weight for sense_id, weight, _query in senses}
        order_map = {sense_id: idx for idx, (sense_id, _w, _q) in enumerate(senses)}

        # ── Step 3 & 4: fire all worker nodes concurrently ───────────────
        with ThreadPoolExecutor(max_workers=len(senses)) as executor:
            node_tasks = [
                self._retrieve_node(
                    sense_id,
                    weight,
                    query,
                    top_k=top_k,
                    metric=metric,
                    adaptive=adaptive,
                    buffer_size=buffer_size,
                    executor=executor,
                )
                for sense_id, weight, query in senses
            ]
            results = await asyncio.gather(*node_tasks)

        # ── Step 5: collect async outputs ────────────────────────────────
        per_sense_ranked: dict[str, list[RetrievedChunk]] = {}
        per_sense: list[tuple[float, list[RetrievedChunk]]] = []
        for sense_id, weight, ranked in results:
            per_sense_ranked[sense_id] = ranked
            per_sense.append((weight, ranked))

        t_parallel = time.perf_counter() - t_start
        logger.info(
            f"[MMASA] All nodes complete. Parallel wall-time={t_parallel:.3f}s → synthesizing..."
        )

        # ── Step 6: Mother Agent synthesizes ─────────────────────────────
        return self._fuse_sense_results(
            senses,
            per_sense_ranked,
            per_sense,
            top_k=top_k,
            adaptive=adaptive,
            fusion=fusion,
            weight_map=weight_map,
            order_map=order_map,
        )

    def retrieve_multi_sense_parallel(
        self,
        senses: list[tuple[str, float, QueryRepresentation]],
        top_k: int = 5,
        metric: str = "elk",
        fusion: str = "interleave",
        adaptive: bool = True,
        buffer_size: int = 2,
    ) -> tuple[dict[str, list[RetrievedChunk]], list[RetrievedChunk]]:
        """Synchronous entry-point for the MMASA parallel retriever.

        Wraps `retrieve_multi_sense_async` in a blocking call so existing
        callers don't need to be rewritten as async.
        """
        return asyncio.run(
            self.retrieve_multi_sense_async(
                senses,
                top_k=top_k,
                metric=metric,
                fusion=fusion,
                adaptive=adaptive,
                buffer_size=buffer_size,
            )
        )

    # ------------------------------------------------------------------ #
    #  Shared synthesis helper (used by both serial and parallel paths)  #
    # ------------------------------------------------------------------ #

    def _fuse_sense_results(
        self,
        senses: list[tuple[str, float, QueryRepresentation]],
        per_sense_ranked: dict[str, list[RetrievedChunk]],
        per_sense: list[tuple[float, list[RetrievedChunk]]],
        *,
        top_k: int,
        adaptive: bool,
        fusion: str,
        weight_map: dict[str, float],
        order_map: dict[str, int],
    ) -> tuple[dict[str, list[RetrievedChunk]], list[RetrievedChunk]]:
        fused = rerank_multi_sense(per_sense, fusion=fusion)
        if not adaptive:
            fused = fused[:top_k]

        origins: dict[str, list[tuple[str, int]]] = {}
        for sense_id, ranked in per_sense_ranked.items():
            for rank, chunk in enumerate(ranked, start=1):
                origins.setdefault(chunk.knowledge.id, []).append((sense_id, rank))

        provenance_preserving: dict[str, list[RetrievedChunk]] = {
            sense_id: [] for sense_id, _weight, _query in senses
        }
        for chunk in fused:
            chunk_origins = origins.get(chunk.knowledge.id, [])
            if not chunk_origins:
                continue
            ordered_origins = sorted(
                chunk_origins,
                key=lambda item: (
                    item[1],
                    -weight_map[item[0]],
                    order_map[item[0]],
                ),
            )
            for sense_id, _rank in ordered_origins:
                provenance_preserving[sense_id].append(chunk)

        return provenance_preserving, fused
