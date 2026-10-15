"""Disambiguation Router — MMASA-aware.

Applies the entropy gate from polysemy_rag_spec.md §6:

  entropy ≤ ENTROPY_SINGLE (1.5 bits) → single-sense routing
  entropy ≤ ENTROPY_DUAL   (2.5 bits) → dual-sense routing  (MMASA parallel)
  entropy >  ENTROPY_DUAL              → high-entropy fallback (MMASA parallel)

All multi-sense paths now use the MMASA parallel retriever so that each sense
is fetched concurrently (one Worker Node per sense) and results are synthesized
by the Mother Agent, matching the workflow defined in mma.txt §4 Steps 3–6.
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

from gaussian_rag.core.gaussian_embedding import GaussianEmbedder
from gaussian_rag.core.types import (
    DisambiguationPacket,
    QueryRepresentation,
    RetrievedChunk,
)
from gaussian_rag.rag.retrieval.retriever import GaussianRetriever

from .field_mapper import ENTROPY_DUAL, ENTROPY_SINGLE

logger = logging.getLogger(__name__)

# Number of chunks to fetch per sense in fallback mode (wider net)
FALLBACK_TOP_K = 10


def _metric_requires_query_covariance(metric: str) -> bool:
    return metric in {"elk", "fisher_rao"}


def _build_sense_query(
    sense,
    embedder: GaussianEmbedder,
    metric: str,
) -> QueryRepresentation:
    encoded = embedder.encode_query(
        sense.retrieval_query,
        with_covariance=_metric_requires_query_covariance(metric),
    )
    return QueryRepresentation(
        text=sense.retrieval_query,
        mu=sense.mu_anchor if sense.mu_anchor is not None else encoded.mu,
        sigma_diag=sense.sigma_anchor
        if sense.sigma_anchor is not None
        else encoded.sigma_diag,
        sigma_L=sense.sigma_L_anchor
        if sense.sigma_L_anchor is not None
        else encoded.sigma_L,
        metadata=encoded.metadata,
    )


def route(
    packet: DisambiguationPacket,
    embedder: GaussianEmbedder,
    retriever: GaussianRetriever,
    top_k: int = 5,
    metric: str = "elk",
    fusion: str = "interleave",
    max_senses_dual: int = 2,
    max_senses_fallback: int = 3,
) -> tuple[dict[str, list[RetrievedChunk]], bool]:
    """Route the packet to the appropriate retrieval path.

    Args:
        packet:               Disambiguation result containing ranked senses.
        embedder:             Active GaussianEmbedder instance.
        retriever:            Active GaussianRetriever instance.
        top_k:                Chunks per sense for normal retrieval.
        metric:               Retrieval metric (default: elk).
        fusion:               Multi-sense fusion strategy (default: interleave).
        max_senses_dual:      Max senses for dual-sense MMASA path (default 2).
        max_senses_fallback:  Max senses for high-entropy fallback (default 3).

    Returns:
        sense_results:      sense_id → list[RetrievedChunk]
        fallback_triggered: True when high-entropy fallback was used
    """
    senses = packet.senses
    if not senses:
        return {}, False

    # Sort by descending field weight for consistent ordering
    ordered = sorted(
        senses,
        key=lambda s: packet.field_weights.get(s.sense_id, 0.0),
        reverse=True,
    )

    entropy = packet.entropy

    # ------------------------------------------------------------------
    # SINGLE-SENSE path (entropy ≤ 1.5 bits) — no parallelism needed
    # ------------------------------------------------------------------
    if entropy <= ENTROPY_SINGLE and packet.selected_sense_id:
        sense = next(
            (s for s in ordered if s.sense_id == packet.selected_sense_id), ordered[0]
        )
        query = _build_sense_query(sense, embedder, metric)
        logger.info(f"[Router] Single-sense path → sense='{sense.sense_id}'")
        results = retriever.retrieve(query, top_k=top_k, metric=metric)
        return {sense.sense_id: results}, False

    # ------------------------------------------------------------------
    # DUAL-SENSE path (entropy ≤ 2.5 bits) — MMASA parallel retrieval
    # ------------------------------------------------------------------
    if entropy <= ENTROPY_DUAL:
        top_n = ordered[:max_senses_dual]
        weighted_queries: list[tuple[str, float, QueryRepresentation]] = []
        for sense in top_n:
            weight = packet.field_weights.get(sense.sense_id, 0.5)
            qr = _build_sense_query(sense, embedder, metric)
            weighted_queries.append((sense.sense_id, weight, qr))

        logger.info(
            f"[Router] MMASA dual-sense path → {len(weighted_queries)} worker nodes dispatched"
        )
        # ── Mother Agent: parallel worker nodes ──────────────────────────
        sense_results, _fused = retriever.retrieve_multi_sense_parallel(
            weighted_queries,
            top_k=top_k,
            metric=metric,
            fusion=fusion,
        )
        ordered_results = {
            sense.sense_id: sense_results.get(sense.sense_id, []) for sense in top_n
        }
        return ordered_results, False

    # ------------------------------------------------------------------
    # HIGH-ENTROPY FALLBACK (entropy > 2.5 bits) — MMASA parallel
    # Each sense gets its own Worker Node for concurrent retrieval.
    # ------------------------------------------------------------------
    logger.info(
        f"[Router] MMASA high-entropy fallback → {min(len(ordered), max_senses_fallback)} "
        f"worker nodes dispatched (entropy={entropy:.2f})"
    )
    top_n = ordered[:max_senses_fallback]

    async def _fallback_node(sense) -> tuple[str, list[RetrievedChunk]]:
        """Worker Node for high-entropy fallback — runs retrieve() off the event loop."""
        qr = _build_sense_query(sense, embedder, metric)
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=1) as ex:
            results = await loop.run_in_executor(
                ex,
                lambda: retriever.retrieve(qr, top_k=FALLBACK_TOP_K, metric=metric),
            )
        logger.info(
            f"[Router] Fallback Worker Node done: sense='{sense.sense_id}' nodes={len(results)}"
        )
        return sense.sense_id, results

    async def _run_fallback_parallel():
        tasks = [_fallback_node(s) for s in top_n]
        return await asyncio.gather(*tasks)

    fallback_results = asyncio.run(_run_fallback_parallel())
    sense_results = {sid: chunks for sid, chunks in fallback_results}

    return sense_results, True  # fallback_triggered=True
