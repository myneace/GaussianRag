from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from gaussian_rag.core.types import GaussianKnowledge


@dataclass(slots=True)
class MuAnnIndex:
    ids: list[str] = field(default_factory=list)
    matrix: np.ndarray | None = None

    def build(self, items: list[GaussianKnowledge]) -> None:
        self.ids = [item.id for item in items]
        self.matrix = (
            np.vstack([item.mu for item in items])
            if items
            else np.zeros((0, 0), dtype=float)
        )

    def search(self, mu: np.ndarray, top_k: int | None) -> list[str]:
        if self.matrix is None or self.matrix.size == 0:
            return []
        distances = np.linalg.norm(self.matrix - mu[None, :], axis=1)
        top_indices = np.argsort(distances)
        if top_k is not None:
            top_indices = top_indices[:top_k]
        return [self.ids[index] for index in top_indices]

    def search_excluding(
        self, mu: np.ndarray, top_k: int | None, exclude_ids: set[str]
    ) -> list[str]:
        """Like search() but filters out specific chunk IDs (soft NOT-filter)."""
        if self.matrix is None or self.matrix.size == 0:
            return []
        distances = np.linalg.norm(self.matrix - mu[None, :], axis=1)
        sorted_indices = np.argsort(distances)
        results: list[str] = []
        for idx in sorted_indices:
            chunk_id = self.ids[idx]
            if chunk_id not in exclude_ids:
                results.append(chunk_id)
            if top_k is not None and len(results) >= top_k:
                break
        return results
