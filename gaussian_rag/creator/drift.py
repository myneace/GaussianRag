from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gaussian_rag.core.information_geometry import wasserstein2
from gaussian_rag.core.types import GaussianKnowledge


@dataclass(slots=True)
class DriftMeasurement:
    item_id: str
    drift_score: float


def measure_drift(
    previous: list[GaussianKnowledge], current: list[GaussianKnowledge]
) -> list[DriftMeasurement]:
    current_by_id = {item.id: item for item in current}
    measurements: list[DriftMeasurement] = []
    for item in previous:
        updated = current_by_id.get(item.id)
        if updated is None:
            continue
        measurements.append(
            DriftMeasurement(item_id=item.id, drift_score=wasserstein2(item, updated))
        )
    return measurements


def mean_drift(
    previous: list[GaussianKnowledge], current: list[GaussianKnowledge]
) -> float:
    scores = [item.drift_score for item in measure_drift(previous, current)]
    return float(np.mean(scores)) if scores else 0.0
