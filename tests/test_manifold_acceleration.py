"""Tests for Phase 05-02: Manifold Ranker Metric Pruning & Uncertainty Weighting.

Verifies:
  1. elk metric skips wasserstein2 (w2_distance is None)
  2. Directional uncertainty weighting down-ranks high-variance candidates
  3. Reranking 30 384-dimensional candidates executes in < 15ms
"""

from __future__ import annotations

import time
import unittest
from unittest.mock import patch

import numpy as np

from gaussian_rag.core.types import GaussianKnowledge, QueryRepresentation
from gaussian_rag.rag.retrieval.manifold_ranker import rerank_candidates


_DIM = 384
_RNG = np.random.default_rng(0)


def _make_candidate(
    dim: int = _DIM,
    *,
    trace_sigma: float | None = None,
    mu: np.ndarray | None = None,
    id_suffix: str = "",
) -> GaussianKnowledge:
    if mu is None:
        mu = _RNG.standard_normal(dim)
        mu /= np.linalg.norm(mu) + 1e-9
    if trace_sigma is None:
        sigma_diag = np.clip(_RNG.uniform(1e-5, 1e-3, dim), 1e-6, None)
    else:
        sigma_diag = np.full(dim, trace_sigma / dim)
    sigma_L = np.zeros((dim, 0))
    return GaussianKnowledge(
        id=f"doc_{id_suffix or hash(mu.tobytes()) & 0xFFFF}",
        text="test",
        mu=mu,
        sigma_diag=sigma_diag,
        sigma_L=sigma_L,
    )


def _make_query(dim: int = _DIM) -> QueryRepresentation:
    mu = _RNG.standard_normal(dim)
    mu /= np.linalg.norm(mu) + 1e-9
    sigma_diag = np.full(dim, 1e-4)
    sigma_L = np.zeros((dim, 0))
    return QueryRepresentation(
        text="test query",
        mu=mu,
        sigma_diag=sigma_diag,
        sigma_L=sigma_L,
    )


class TestElkSkipsWasserstein2(unittest.TestCase):
    """w2_distance must be None when metric == 'elk'."""

    def test_elk_metric_yields_w2_none(self) -> None:
        query = _make_query()
        candidates = [_make_candidate(id_suffix=str(i)) for i in range(5)]

        # Patch wasserstein2 to confirm it is NOT called
        with patch(
            "gaussian_rag.rag.retrieval.manifold_ranker.wasserstein2",
            side_effect=AssertionError("wasserstein2 must not be called for metric='elk'"),
        ):
            results = rerank_candidates(query, candidates, metric="elk")

        for chunk in results:
            self.assertIsNone(
                chunk.w2_distance,
                f"w2_distance should be None for elk metric, got {chunk.w2_distance}",
            )

    def test_elk_scores_are_populated(self) -> None:
        query = _make_query()
        candidates = [_make_candidate(id_suffix=str(i)) for i in range(5)]
        results = rerank_candidates(query, candidates, metric="elk")
        for chunk in results:
            self.assertIsNotNone(chunk.elk_score, "elk_score must be set for elk metric")

    def test_wasserstein2_metric_populates_w2_distance(self) -> None:
        """Ensure w2_distance is populated when explicitly requesting wasserstein2."""
        query = _make_query()
        candidates = [_make_candidate(id_suffix=str(i)) for i in range(3)]
        results = rerank_candidates(query, candidates, metric="wasserstein2")
        for chunk in results:
            self.assertIsNotNone(
                chunk.w2_distance,
                "w2_distance must be set when metric='wasserstein2'",
            )
            self.assertIsNone(
                chunk.elk_score,
                "elk_score must be None when metric='wasserstein2'",
            )


class TestDirectionalUncertaintyWeighting(unittest.TestCase):
    """High-variance candidates should rank lower than low-variance ones."""

    def test_uncertainty_weighting_down_ranks_high_variance(self) -> None:
        query = _make_query()

        # Create a near-query mu so ELK raw score is similar; differ only in variance
        query_gaussian = query.as_gaussian()
        near_mu = query_gaussian.mu + _RNG.standard_normal(_DIM) * 0.001

        # Low-variance candidate: trace_sigma ≈ 0.01 → w ≈ 0.99
        low_var = GaussianKnowledge(
            id="low_var",
            text="low variance doc",
            mu=near_mu.copy(),
            sigma_diag=np.full(_DIM, 0.01 / _DIM),
            sigma_L=np.zeros((_DIM, 0)),
        )
        # High-variance candidate: trace_sigma = 50.0 → w ≈ 0.02
        high_var = GaussianKnowledge(
            id="high_var",
            text="high variance doc",
            mu=near_mu.copy(),
            sigma_diag=np.full(_DIM, 50.0 / _DIM),
            sigma_L=np.zeros((_DIM, 0)),
        )

        results = rerank_candidates(query, [high_var, low_var], metric="elk")
        self.assertEqual(len(results), 2)

        # low_var should rank first (rank=1) since it has higher confidence weighting
        id_by_rank = {chunk.rank: chunk.knowledge.id for chunk in results}
        self.assertEqual(
            id_by_rank[1],
            "low_var",
            f"Expected low_var at rank 1 but got {id_by_rank[1]}. "
            f"ELK scores: {[f'{c.knowledge.id}={c.elk_score:.6f}' for c in results]}",
        )

    def test_uncertainty_weighting_scaling(self) -> None:
        """Verify w(Σ) = 1/(1+tr(Σ)) scales ELK linearly with confidence."""
        from gaussian_rag.rag.retrieval.manifold_ranker import (
            _directional_uncertainty_weight,
        )

        low_var = _make_candidate(trace_sigma=0.1, id_suffix="low")
        high_var = _make_candidate(trace_sigma=100.0, id_suffix="high")

        w_low = _directional_uncertainty_weight(low_var)
        w_high = _directional_uncertainty_weight(high_var)

        self.assertAlmostEqual(w_low, 1.0 / (1.0 + 0.1), places=5)
        self.assertAlmostEqual(w_high, 1.0 / (1.0 + 100.0), places=5)
        self.assertGreater(w_low, w_high, "Low-variance candidate should have higher weight")


class TestRerankerLatency(unittest.TestCase):
    """Reranking 30 384-dim candidates must complete in < 15ms."""

    def test_elk_reranking_latency_30_candidates(self) -> None:
        query = _make_query()
        candidates = [_make_candidate(id_suffix=str(i)) for i in range(30)]

        # Warm up (first call may initialise JIT caches)
        rerank_candidates(query, candidates, metric="elk")

        iterations = 10
        start = time.perf_counter()
        for _ in range(iterations):
            rerank_candidates(query, candidates, metric="elk")
        elapsed = (time.perf_counter() - start) / iterations * 1000  # ms

        self.assertLess(
            elapsed,
            15.0,
            f"Reranking 30 candidates took {elapsed:.2f}ms (must be < 15ms)",
        )

    def test_elk_reranking_ordering_is_stable(self) -> None:
        """Same query and candidates always produce the same ranking."""
        query = _make_query()
        candidates = [_make_candidate(id_suffix=str(i)) for i in range(10)]

        r1 = rerank_candidates(query, candidates, metric="elk")
        r2 = rerank_candidates(query, candidates, metric="elk")

        ids1 = [c.knowledge.id for c in r1]
        ids2 = [c.knowledge.id for c in r2]
        self.assertEqual(ids1, ids2, "Reranking is not deterministic")


if __name__ == "__main__":
    unittest.main()
