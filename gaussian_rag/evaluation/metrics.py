from __future__ import annotations

import math

import numpy as np


def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    retrieved_set = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    return len(retrieved_set & relevant_set) / len(relevant_set)


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    gains = [1.0 if item in set(relevant_ids) else 0.0 for item in retrieved_ids[:k]]
    dcg = sum(gain / math.log2(index + 2) for index, gain in enumerate(gains))
    ideal = sum(
        1.0 / math.log2(index + 2) for index in range(min(k, len(relevant_ids)))
    )
    if ideal == 0.0:
        return 0.0
    return dcg / ideal


def expected_calibration_error(
    confidences: list[float], outcomes: list[int], bins: int = 10
) -> float:
    if not confidences:
        return 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    conf_array = np.asarray(confidences, dtype=float)
    out_array = np.asarray(outcomes, dtype=float)
    for index in range(bins):
        lower = edges[index]
        upper = edges[index + 1]
        mask = (conf_array >= lower) & (
            conf_array < upper if index < bins - 1 else conf_array <= upper
        )
        if not np.any(mask):
            continue
        avg_conf = float(np.mean(conf_array[mask]))
        avg_acc = float(np.mean(out_array[mask]))
        error += (float(np.sum(mask)) / len(confidences)) * abs(avg_conf - avg_acc)
    return error


def brier_score(confidences: list[float], outcomes: list[int]) -> float:
    if not confidences:
        return 0.0
    conf = np.asarray(confidences, dtype=float)
    out = np.asarray(outcomes, dtype=float)
    return float(np.mean((conf - out) ** 2))


def abstention_curve(
    confidences: list[float], outcomes: list[int]
) -> list[tuple[float, float]]:
    pairs = sorted(
        zip(confidences, outcomes, strict=False), key=lambda item: item[0], reverse=True
    )
    curve: list[tuple[float, float]] = []
    kept: list[int] = []
    for index, (confidence, outcome) in enumerate(pairs, start=1):
        kept.append(outcome)
        selective_accuracy = sum(kept) / len(kept)
        retention = index / len(pairs)
        curve.append((retention, selective_accuracy))
    return curve
