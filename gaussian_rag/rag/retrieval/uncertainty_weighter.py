from __future__ import annotations

from gaussian_rag.core.types import RetrievedChunk


def weighted_distance(chunk: RetrievedChunk) -> float:
    return chunk.w2_distance / max(chunk.confidence, 1e-9)


def sort_by_weighted_distance(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    return sorted(chunks, key=weighted_distance)
