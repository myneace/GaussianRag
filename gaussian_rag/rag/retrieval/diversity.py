from __future__ import annotations

from gaussian_rag.core.information_geometry import wasserstein2
from gaussian_rag.core.types import RetrievedChunk


def diversity_filter(
    chunks: list[RetrievedChunk], threshold: float = 0.3, top_k: int | None = 5
) -> list[RetrievedChunk]:
    selected: list[RetrievedChunk] = []
    for chunk in chunks:
        if all(
            wasserstein2(chunk.knowledge, item.knowledge) > threshold
            for item in selected
        ):
            selected.append(chunk)
        if top_k is not None and len(selected) >= top_k:
            break
    return selected
