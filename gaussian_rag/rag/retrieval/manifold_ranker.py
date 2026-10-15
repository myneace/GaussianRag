from __future__ import annotations

import logging

from gaussian_rag.core.covariance import elk_score_low_rank
from gaussian_rag.core.information_geometry import fisher_rao, wasserstein2
from gaussian_rag.core.types import (
    GaussianKnowledge,
    QueryRepresentation,
    RetrievedChunk,
)

logger = logging.getLogger(__name__)


def _directional_uncertainty_weight(candidate: GaussianKnowledge) -> float:
    """Directional uncertainty weighting: w(Σ) = 1 / (1 + tr(Σ)).

    High-variance candidates are down-weighted, giving preference to confident
    (low-uncertainty) chunks during manifold reranking.
    """
    return 1.0 / (1.0 + candidate.trace_sigma)


def _ranking_score(chunk: RetrievedChunk, metric: str) -> float:
    if metric == "elk":
        score = chunk.elk_score if chunk.elk_score is not None else float("-inf")
        # ELK is a log-similarity (higher = better); negate for ascending sort
        return -score
    distance = chunk.w2_distance if metric == "wasserstein2" else chunk.fr_distance
    if distance is None:
        distance = chunk.w2_distance
    if distance is None:
        return float("inf")
    return distance / max(chunk.confidence, 1e-9)


def rerank_candidates(
    query: QueryRepresentation,
    candidates: list[GaussianKnowledge],
    *,
    metric: str = "elk",
) -> list[RetrievedChunk]:
    """Rerank *candidates* by the requested geometric metric.

    When ``metric == "elk"``, only the low-rank ELK score is computed — the
    O(d³) Wasserstein-2 distance is **skipped entirely**.  This keeps reranking
    well below the 15 ms SLA for 30 candidates.

    Directional uncertainty weighting ``w(Σ) = 1 / (1 + tr(Σ))`` is applied to
    ELK scores so that high-variance candidates are down-ranked relative to
    tight, confident chunks.
    """
    query_gaussian = query.as_gaussian()
    ranked: list[RetrievedChunk] = []

    logger.info(
        "Reranking %d candidates with metric='%s' (OpenBLAS natively parallelized).",
        len(candidates),
        metric,
    )

    for i, candidate in enumerate(candidates):
        # ---------- Wasserstein-2 ----------------------------------------
        # Only computed when explicitly requested; skipped for elk and fisher_rao
        # to avoid the O(d³) eigh bottleneck.
        if metric == "wasserstein2":
            w2_distance: float | None = wasserstein2(query_gaussian, candidate)
        else:
            w2_distance = None

        # ---------- Fisher-Rao -------------------------------------------
        fr_distance: float | None = (
            fisher_rao(query_gaussian, candidate) if metric == "fisher_rao" else None
        )

        # ---------- ELK (low-rank, O(d·r²)) --------------------------------
        overlap_score: float | None = None
        if metric == "elk":
            raw_elk = elk_score_low_rank(
                query_gaussian.mu,
                query_gaussian.sigma_diag,
                query_gaussian.sigma_L if query_gaussian.sigma_L.size > 0 else None,
                candidate.mu,
                candidate.sigma_diag,
                candidate.sigma_L if candidate.sigma_L.size > 0 else None,
            )
            # Apply directional uncertainty weighting: confident chunks score higher
            w = _directional_uncertainty_weight(candidate)
            overlap_score = raw_elk * w

        ranked.append(
            RetrievedChunk(
                knowledge=candidate,
                w2_distance=w2_distance,
                fr_distance=fr_distance,
                elk_score=overlap_score,
                confidence=candidate.confidence,
                uncertainty_label=candidate.uncertainty_label,
                rank=0,
            )
        )

        if (i + 1) % 100 == 0:
            logger.info("Evaluated %d/%d candidates.", i + 1, len(candidates))

    ranked.sort(key=lambda chunk: _ranking_score(chunk, metric))
    for index, item in enumerate(ranked, start=1):
        item.rank = index
    return ranked


def _dedup_by_id(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    seen: set[str] = set()
    result: list[RetrievedChunk] = []
    for chunk in chunks:
        chunk_id = chunk.knowledge.id
        if chunk_id not in seen:
            seen.add(chunk_id)
            result.append(chunk)
    return result


def rerank_multi_sense(
    sense_results: list[tuple[float, list[RetrievedChunk]]],
    *,
    fusion: str = "interleave",
) -> list[RetrievedChunk]:
    if not sense_results:
        return []

    if fusion == "weighted_sum":
        score_map: dict[str, tuple[float, RetrievedChunk]] = {}
        for weight, chunks in sense_results:
            for chunk in chunks:
                cid = chunk.knowledge.id
                base_score = (
                    chunk.elk_score
                    if chunk.elk_score is not None
                    else 1.0 / max(chunk.w2_distance or 1.0, 1e-9)
                )
                contrib = weight * base_score
                if cid not in score_map or contrib > score_map[cid][0]:
                    score_map[cid] = (contrib, chunk)
        weighted_merged = sorted(score_map.values(), key=lambda t: -t[0])
        ranked = [chunk for _, chunk in weighted_merged]
    else:
        iterators = [iter(chunks) for _, chunks in sense_results]
        merged: list[RetrievedChunk] = []
        while iterators:
            next_iterators = []
            for iterator in iterators:
                try:
                    merged.append(next(iterator))
                    next_iterators.append(iterator)
                except StopIteration:
                    pass
            iterators = next_iterators
        ranked = _dedup_by_id(merged)

    for index, item in enumerate(ranked, start=1):
        item.rank = index
    return ranked
