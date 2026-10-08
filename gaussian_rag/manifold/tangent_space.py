from __future__ import annotations

import numpy as np

from .manifold import matrix_inv_sqrt, matrix_sqrt, project_spd


def log_map(base: np.ndarray, point: np.ndarray) -> np.ndarray:
    base_sqrt = matrix_sqrt(base)
    base_inv_sqrt = matrix_inv_sqrt(base)
    whitened = project_spd(base_inv_sqrt @ point @ base_inv_sqrt)
    values, vectors = np.linalg.eigh(whitened)
    log_whitened = (vectors * np.log(values)) @ vectors.T
    return base_sqrt @ log_whitened @ base_sqrt


def flatten_symmetric(matrix: np.ndarray) -> np.ndarray:
    upper = np.triu_indices_from(matrix)
    return matrix[upper]


def tangent_projection(base: np.ndarray, point: np.ndarray) -> np.ndarray:
    return flatten_symmetric(log_map(base, point))
