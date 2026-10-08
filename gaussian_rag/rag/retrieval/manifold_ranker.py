from __future__ import annotations

import logging

from gaussian_rag.core.covariance import elk_score
from gaussian_rag.core.information_geometry import fisher_rao, wasserstein2
from gaussian_rag.core.types import (
    GaussianKnowledge,
    QueryRepresentation,
    RetrievedChunk,
)

logger = logging.getLogger(__name__)


def _ranking_score(chunk: RetrievedChunk, metric: str) -> float:
    if metric == "elk":
        return -(chunk.elk_score if chunk.elk_score is not None else float("-inf"))
    distance = chunk.w2_distance if metric == "wasserstein2" else chunk.fr_distance
    if distance is None:
        distance = chunk.w2_distance
    return distance / max(chunk.confidence, 1e-9)


def rerank_candidates(
    query: QueryRepresentation,
    candidates: list[GaussianKnowledge],
    *,
    metric: str = "elk",
) -> list[RetrievedChunk]:
    query_gaussian = query.as_gaussian()
    ranked: list[RetrievedChunk] = []

    logger.info(
        "Starting sequential evaluation of %d candidates (OpenBLAS natively parallelized)...",
        len(candidates),
    )

    for i, candidate in enumerate(candidates):
        w2_distance = wasserstein2(query_gaussian, candidate)
        fr_distance = (
            fisher_rao(query_gaussian, candidate) if metric == "fisher_rao" else None
        )
        overlap_score = None
        if metric == "elk":
            overlap_score = elk_score(
                query_gaussian.mu,
                query_gaussian.sigma,
                candidate.mu,
                candidate.sigma,
            )
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
                    else 1.0 / max(chunk.w2_distance, 1e-9)
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
