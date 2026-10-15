from __future__ import annotations

import numpy as np

from gaussian_rag.core.information_geometry import wasserstein2
from gaussian_rag.core.types import GaussianKnowledge


def coverage_score(chunks: list[GaussianKnowledge]) -> float:
    if len(chunks) < 2:
        return 0.0
    distances: list[float] = []
    for index, chunk in enumerate(chunks):
        peers = chunks[index + 1 :]
        if not peers:
            continue
        distances.extend(wasserstein2(chunk, peer) for peer in peers)
    return float(np.mean(distances)) if distances else 0.0
