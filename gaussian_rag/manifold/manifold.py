from __future__ import annotations

import numpy as np


def project_spd(matrix: np.ndarray, floor: float = 1e-6) -> np.ndarray:
    symmetric = (matrix + matrix.T) / 2.0
    values, vectors = np.linalg.eigh(symmetric)
    values = np.clip(values, floor, None)
    return (vectors * values) @ vectors.T


def matrix_sqrt(matrix: np.ndarray, floor: float = 1e-6) -> np.ndarray:
    values, vectors = np.linalg.eigh(project_spd(matrix, floor=floor))
    return (vectors * np.sqrt(values)) @ vectors.T


def matrix_inv_sqrt(matrix: np.ndarray, floor: float = 1e-6) -> np.ndarray:
    values, vectors = np.linalg.eigh(project_spd(matrix, floor=floor))
    return (vectors * (1.0 / np.sqrt(values))) @ vectors.T


def geodesic_interpolate(start: np.ndarray, end: np.ndarray, t: float) -> np.ndarray:
    start_sqrt = matrix_sqrt(start)
    start_inv_sqrt = matrix_inv_sqrt(start)
    transport = start_inv_sqrt @ end @ start_inv_sqrt
    values, vectors = np.linalg.eigh(project_spd(transport))
    powered = (vectors * (values**t)) @ vectors.T
    return project_spd(start_sqrt @ powered @ start_sqrt)
