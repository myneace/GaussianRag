from __future__ import annotations

from ..evaluation.ablation import AblationResult
from .phase1_heuristic_sigma import run as run_phase1
from .phase3_geometry import run as run_phase3


def run() -> list[AblationResult]:
    phase1_agg, _ = run_phase1()
    return [
        AblationResult(
            name="heuristic_sigma",
            score=phase1_agg["ndcg@3"],
            notes="W2 retrieval baseline",
        ),
        AblationResult(
            name="coverage_proxy",
            score=run_phase3(),
            notes="Mean pairwise manifold spread",
        ),
    ]
