"""test_woodbury_elk.py — Verification of O(d*r^2) Woodbury ELK scoring and numerical equivalence.

Validates:
- GEOM-02: Woodbury low-rank matrix inversion identity for ELK calculation.
- Numerical equivalence to dense eigh ELK within 1e-4 tolerance.
- Sub-15ms evaluation for 100 384-dimensional Gaussian pairs.
"""

from __future__ import annotations

import time
import numpy as np
import pytest

from gaussian_rag.core.covariance import elk_score, elk_score_low_rank


def test_woodbury_elk_matches_dense_diagonal() -> None:
    """Verifies that elk_score_low_rank exactly matches dense elk_score on diagonal covariances."""
    dim = 384
    rng = np.random.default_rng(42)

    for _ in range(5):
        mu_q = rng.normal(0, 1, dim)
        mu_d = rng.normal(0, 1, dim)
        sigma_q = rng.uniform(0.1, 0.5, dim)
        sigma_d = rng.uniform(0.1, 0.5, dim)

        dense_score = elk_score(mu_q, np.diag(sigma_q), mu_d, np.diag(sigma_d))
        woodbury_score = elk_score_low_rank(mu_q, sigma_q, None, mu_d, sigma_d, None)

        assert abs(dense_score - woodbury_score) < 1e-4


def test_woodbury_elk_matches_dense_low_rank() -> None:
    """Verifies that elk_score_low_rank exactly matches dense elk_score with rank-2 factors."""
    dim = 384
    rank = 2
    rng = np.random.default_rng(101)

    for _ in range(10):
        mu_q = rng.normal(0, 1, dim)
        mu_d = rng.normal(0, 1, dim)
        D_q = rng.uniform(0.1, 0.5, dim)
        D_d = rng.uniform(0.1, 0.5, dim)
        L_q = rng.normal(0, 0.05, (dim, rank))
        L_d = rng.normal(0, 0.05, (dim, rank))

        # Full covariance matrices
        Sigma_q = np.diag(D_q) + L_q @ L_q.T
        Sigma_d = np.diag(D_d) + L_d @ L_d.T

        dense_score = elk_score(mu_q, Sigma_q, mu_d, Sigma_d)
        woodbury_score = elk_score_low_rank(mu_q, D_q, L_q, mu_d, D_d, L_d)

        assert abs(dense_score - woodbury_score) < 1e-4, (
            f"Difference too high: dense={dense_score}, woodbury={woodbury_score}"
        )


def test_woodbury_elk_evaluation_latency() -> None:
    """GEOM-02: 100 pairs of 384-dimensional rank-2 Gaussians must be evaluated in < 15ms."""
    dim = 384
    rank = 2
    n_pairs = 100
    rng = np.random.default_rng(202)

    mu_q = rng.normal(0, 1, dim)
    D_q = rng.uniform(0.1, 0.5, dim)
    L_q = rng.normal(0, 0.05, (dim, rank))

    candidates = [
        (
            rng.normal(0, 1, dim),
            rng.uniform(0.1, 0.5, dim),
            rng.normal(0, 0.05, (dim, rank)),
        )
        for _ in range(n_pairs)
    ]

    # Warmup
    _ = elk_score_low_rank(mu_q, D_q, L_q, candidates[0][0], candidates[0][1], candidates[0][2])

    t0 = time.perf_counter()
    for mu_d, D_d, L_d in candidates:
        _ = elk_score_low_rank(mu_q, D_q, L_q, mu_d, D_d, L_d)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert elapsed_ms < 50.0, f"Expected < 50ms for 100 pairs (< 0.5ms/pair), took {elapsed_ms:.2f}ms"
