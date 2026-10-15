from __future__ import annotations

from gaussian_rag.main import build_demo_pipeline

from ..manifold.coverage import coverage_score


def run() -> float:
    retriever = build_demo_pipeline()
    return coverage_score(retriever.store.values())
