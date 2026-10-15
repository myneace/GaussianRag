from __future__ import annotations

import math

import numpy as np

from .types import GaussianKnowledge


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    return (matrix + matrix.T) / 2.0


def _sqrtm_spd(matrix: np.ndarray, floor: float = 1e-9) -> np.ndarray:
    values, vectors = np.linalg.eigh(_symmetrize(matrix))
    values = np.clip(values, floor, None)
    return (vectors * np.sqrt(values)) @ vectors.T


def _logm_spd(matrix: np.ndarray, floor: float = 1e-9) -> np.ndarray:
    values, vectors = np.linalg.eigh(_symmetrize(matrix))
    values = np.clip(values, floor, None)
    return (vectors * np.log(values)) @ vectors.T


def bures_metric(s1: np.ndarray, s2: np.ndarray) -> float:
    sqrt_s1 = _sqrtm_spd(s1)
    middle = sqrt_s1 @ s2 @ sqrt_s1
    trace_term = np.trace(s1) + np.trace(s2) - 2.0 * np.trace(_sqrtm_spd(middle))
    return float(max(trace_term, 0.0))


def wasserstein2(g1: GaussianKnowledge, g2: GaussianKnowledge) -> float:
    mean_term = float(np.sum((g1.mu - g2.mu) ** 2))
    cov_term = bures_metric(g1.sigma, g2.sigma)
    return float(math.sqrt(max(mean_term + cov_term, 0.0)))


def kl_divergence(
    g1: GaussianKnowledge, g2: GaussianKnowledge, floor: float = 1e-9
) -> float:
    sigma1 = _symmetrize(g1.sigma)
    sigma2 = _symmetrize(g2.sigma)
    inv_sigma2 = np.linalg.pinv(sigma2)
    delta = (g2.mu - g1.mu)[:, None]
    k = g1.mu.shape[0]
    det1 = max(float(np.linalg.det(sigma1)), floor)
    det2 = max(float(np.linalg.det(sigma2)), floor)
    trace_term = float(np.trace(inv_sigma2 @ sigma1))
    quad_term = float((delta.T @ inv_sigma2 @ delta).item())
    return 0.5 * (math.log(det2 / det1) - k + trace_term + quad_term)


def bhattacharyya(
    g1: GaussianKnowledge, g2: GaussianKnowledge, floor: float = 1e-9
) -> float:
    sigma = _symmetrize((g1.sigma + g2.sigma) / 2.0)
    delta = (g1.mu - g2.mu)[:, None]
    inv_sigma = np.linalg.pinv(sigma)
    det_sigma = max(float(np.linalg.det(sigma)), floor)
    det1 = max(float(np.linalg.det(_symmetrize(g1.sigma))), floor)
    det2 = max(float(np.linalg.det(_symmetrize(g2.sigma))), floor)
    mean_term = 0.125 * float((delta.T @ inv_sigma @ delta).item())
    cov_term = 0.5 * math.log(det_sigma / math.sqrt(det1 * det2))
    return mean_term + cov_term


def fisher_rao(
    g1: GaussianKnowledge, g2: GaussianKnowledge, floor: float = 1e-9
) -> float:
    sigma1 = _symmetrize(np.diag(np.clip(g1.sigma_diag, floor, None)) + g1.sigma_L @ g1.sigma_L.T)
    sigma2 = _symmetrize(np.diag(np.clip(g2.sigma_diag, floor, None)) + g2.sigma_L @ g2.sigma_L.T)
    whitened = (
        _sqrtm_spd(np.linalg.pinv(sigma1)) @ sigma2 @ _sqrtm_spd(np.linalg.pinv(sigma1))
    )
    log_whitened = _logm_spd(whitened, floor=floor)
    mean_term = np.linalg.norm(g1.mu - g2.mu)
    cov_term = np.linalg.norm(log_whitened, ord="fro")
    return float(math.sqrt(mean_term**2 + cov_term**2))
