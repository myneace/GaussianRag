from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Callable

import numpy as np


def elk_score(
    mu_q: np.ndarray,
    sigma_q: np.ndarray,
    mu_d: np.ndarray,
    sigma_d: np.ndarray,
) -> float:
    """Expected Likelihood Kernel score, normalized per dimension.

    Accepts either full covariance matrices with shape ``(d, d)`` or diagonal
    covariance vectors with shape ``(d,)``.
    """
    d = max(mu_q.shape[0], 1)

    def _as_covariance_matrix(sigma: np.ndarray) -> np.ndarray:
        sigma = np.asarray(sigma, dtype=float)
        if sigma.ndim == 1:
            return np.diag(np.clip(np.nan_to_num(sigma, nan=1e-9), 1e-12, None))
        if sigma.ndim == 2:
            sym = np.nan_to_num((sigma + sigma.T) / 2.0, nan=0.0)
            values, vectors = np.linalg.eigh(sym)
            values = np.clip(values, 1e-9, None)
            return vectors @ np.diag(values) @ vectors.T
        raise ValueError("sigma must be a diagonal vector or full covariance matrix")

    sigma_q_matrix = _as_covariance_matrix(sigma_q)
    sigma_d_matrix = _as_covariance_matrix(sigma_d)
    combined_cov = _as_covariance_matrix(sigma_q_matrix + sigma_d_matrix)

    diff = mu_q - mu_d
    values, vectors = np.linalg.eigh(combined_cov)
    values = np.clip(np.nan_to_num(values, nan=1e-9), 1e-9, None)
    projected = vectors.T @ diff
    mahal_term = float(np.sum((projected**2) / values))
    logdet = float(np.sum(np.log(values)))
    return -0.5 * (mahal_term + logdet) / d


def diag_plus_lowrank(
    covariance: np.ndarray, rank: int, floor: float = 1e-6
) -> tuple[np.ndarray, np.ndarray]:
    covariance = (covariance + covariance.T) / 2.0
    diag = np.clip(np.diag(covariance), floor, None)
    residual = covariance - np.diag(diag)
    if rank <= 0:
        return diag, np.zeros((covariance.shape[0], 0))
    values, vectors = np.linalg.eigh(residual)
    values = np.clip(values, 0.0, None)
    top = np.argsort(values)[-rank:]
    top_values = values[top]
    top_vectors = vectors[:, top]
    sigma_l = top_vectors * np.sqrt(top_values)
    return diag, sigma_l


@dataclass(slots=True)
class HeuristicSigmaEstimator:
    encoder: Callable[[str], np.ndarray]
    rank: int = 16
    floor: float = 1e-6

    def _augmentations(self, text: str) -> Sequence[str]:
        tokens = text.split()
        shorter = " ".join(tokens[: max(1, len(tokens) // 2)])
        head = " ".join(tokens[: min(len(tokens), 32)])
        tail = " ".join(tokens[-min(len(tokens), 32) :])
        normalized = " ".join(text.lower().split())
        deduped = " ".join(dict.fromkeys(tokens))
        reversed_tokens = " ".join(reversed(tokens[: min(len(tokens), 64)]))
        return [text, normalized, shorter, head, tail, deduped, reversed_tokens]

    def estimate(self, text: str, mu: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        views = [
            self.encoder(variant) for variant in self._augmentations(text) if variant
        ]
        if len(views) < 2:
            return np.full_like(mu, self.floor), np.zeros((mu.shape[0], 0))
        stacked = np.vstack(views)
        covariance = np.cov(stacked, rowvar=False)
        covariance = np.atleast_2d(covariance)
        return diag_plus_lowrank(covariance, rank=self.rank, floor=self.floor)


@dataclass(slots=True)
class LearnedSigmaEstimator:
    dimension: int
    rank: int = 16
    floor: float = 1e-6
    diag_scale: np.ndarray = field(init=False)
    rank_scale: np.ndarray = field(init=False)
    head: object | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.diag_scale = np.ones(self.dimension, dtype=float)
        self.rank_scale = np.linspace(1.0, 0.25, num=max(self.rank, 1), dtype=float)

    def fit(self, hidden_states: Sequence[np.ndarray]) -> None:
        if not hidden_states:
            return
        stacked = np.vstack([np.asarray(state, dtype=float) for state in hidden_states])
        mean_abs = np.mean(np.abs(stacked), axis=0)
        if mean_abs.shape[0] < self.dimension:
            mean_abs = np.pad(mean_abs, (0, self.dimension - mean_abs.shape[0]))
        learned = np.clip(mean_abs[: self.dimension], self.floor, None)
        self.diag_scale = learned / max(float(np.linalg.norm(learned)), self.floor)
        if self.rank > 0:
            base = np.linspace(
                float(np.max(self.diag_scale)),
                float(np.min(self.diag_scale)),
                num=self.rank,
                dtype=float,
            )
            self.rank_scale = np.clip(base, self.floor, None)

    def estimate(self, hidden_state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.head is not None:
            return self.head.predict(hidden_state)

        hidden_state = np.asarray(hidden_state, dtype=float)
        weights = np.abs(hidden_state[: self.dimension])
        if weights.shape[0] < self.dimension:
            weights = np.pad(weights, (0, self.dimension - weights.shape[0]))
        scaled = weights * self.diag_scale
        sigma_diag = np.clip(
            scaled / max(float(np.linalg.norm(scaled)), self.floor), self.floor, None
        )
        sigma_l = np.zeros((self.dimension, self.rank), dtype=float)
        for column in range(self.rank):
            sigma_l[:, column] = sigma_diag * (
                self.rank_scale[column] / max(float(self.rank), 1.0)
            )
        return sigma_diag, sigma_l


@dataclass(slots=True)
class DissolverSigmaEstimator:
    encoder: Callable[[str], np.ndarray]
    floor: float = 1e-6

    def estimate(
        self, text: str, interpretations: Sequence[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(interpretations) < 2:
            # Fallback to zero variance if dissolver fails or provides only one view
            dimension = self.encoder("").shape[0]
            return np.full(dimension, self.floor), np.zeros((dimension, 0))

        view_embeddings = [self.encoder(view) for view in interpretations]
        stacked = np.vstack(view_embeddings)

        # Using ddof=1 for unbiased sample variance across the interpretations
        variance_diag = np.var(stacked, axis=0, ddof=1)

        # Apply numerical floor
        sigma_diag = np.clip(variance_diag, self.floor, None)

        # Return diagonal and empty low-rank matrix
        return sigma_diag, np.zeros((sigma_diag.shape[0], 0))


@dataclass(slots=True)
class EntityAwareSigmaEstimator:
    """Sigma estimator that scales covariance based on entity density.

    Inspired by the entity-standardisation pass in ai-knowledge-graph:
    when a chunk contains many unique entities it is semantically broad
    (high variance → large Σ); when a chunk is dominated by one repeated
    entity it is focused (low variance → small Σ).  This mirrors the
    GSKG concept where broad concepts have larger covariance matrices and
    specific concepts have tighter ones.

    Entity detection is intentionally lightweight (stop-word-filtered
    capitalised tokens and noun phrases) so it works without an NLP
    dependency.  The result is blended with a heuristic augmentation
    estimate so the node still captures syntactic variance.

    Two public estimation methods:
      ``estimate(text, mu)``        — for DOCUMENTS: uses augmentation covariance
                                      blended with capitalised-entity density.
      ``estimate_query(text, mu)``  — for QUERIES: uses token diversity × length
                                      saturation; bypasses augmentation which
                                      anti-correlates with query length.
    """

    encoder: Callable[[str], np.ndarray]
    rank: int = 8
    floor: float = 1e-6
    # Controls how much entity density affects sigma in estimate().
    # 0.0 → pure heuristic augmentation; 1.0 → full entity-density scaling.
    entity_weight: float = 0.5

    # Stop words used to filter out non-entity tokens
    _STOP_WORDS: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {
                "the",
                "a",
                "an",
                "of",
                "and",
                "or",
                "in",
                "on",
                "at",
                "to",
                "for",
                "with",
                "by",
                "as",
                "is",
                "are",
                "was",
                "were",
                "be",
                "been",
                "being",
                "it",
                "its",
                "this",
                "that",
                "these",
                "those",
            }
        )
    )

    def _extract_entities(self, text: str) -> list[str]:
        """Extract lightweight entity tokens (capitalised non-stop words)."""
        import re

        tokens = re.findall(r"\b[A-Za-z][a-z]*\b", text)
        entities = [
            t.lower()
            for t in tokens
            if t[0].isupper() and t.lower() not in self._STOP_WORDS
        ]
        return entities

    def _entity_scale(self, text: str) -> float:
        """Return a sigma scale factor in (0, 1] based on entity diversity.

        High diversity (many unique entities relative to total) → scale near 1.
        Low diversity (repeated single entity) → scale near floor.
        """
        entities = self._extract_entities(text)
        if not entities:
            return 0.5  # neutral when no entities detected
        unique_ratio = len(set(entities)) / max(len(entities), 1)
        # Map ratio [0, 1] → scale [0.1, 1.0]
        return max(0.1, float(unique_ratio))

    def _heuristic_augmentations(self, text: str) -> list[str]:
        tokens = text.split()
        shorter = " ".join(tokens[: max(1, len(tokens) // 2)])
        head = " ".join(tokens[: min(len(tokens), 32)])
        tail = " ".join(tokens[-min(len(tokens), 32) :])
        normalized = " ".join(text.lower().split())
        return [text, normalized, shorter, head, tail]

    def estimate(self, text: str, mu: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Estimate sigma for a DOCUMENT using entity density + heuristic augmentation."""
        # Heuristic covariance from text augmentations
        views = [self.encoder(v) for v in self._heuristic_augmentations(text) if v]
        if len(views) < 2:
            sigma_diag = np.full_like(mu, self.floor)
            return sigma_diag, np.zeros((mu.shape[0], 0))

        stacked = np.vstack(views)
        covariance = np.cov(stacked, rowvar=False)
        covariance = np.atleast_2d(covariance)
        base_diag, base_L = diag_plus_lowrank(
            covariance, rank=self.rank, floor=self.floor
        )

        # Entity-density scaling
        scale = self._entity_scale(text)
        blend = 1.0 - self.entity_weight + self.entity_weight * scale

        sigma_diag = np.clip(base_diag * blend, self.floor, None)
        sigma_L = base_L * np.sqrt(blend)
        return sigma_diag, sigma_L

    def estimate_query(
        self, text: str, mu: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Estimate sigma for a QUERY using embedding-aligned non-uniform variance.

        Why a separate method from ``estimate``:
        The heuristic augmentation approach has a systematic bias for queries —
        truncating a 3-token query creates a larger embedding shift than truncating
        a 10-token query, so short specific queries incorrectly receive *higher*
        covariance than long ambiguous ones.

        Why non-uniform sigma is essential:
        If sigma_diag is a uniform constant c·𝟏, then ELK's Mahalanobis term
        reduces to ‖µq−µd‖²/(2c²), which is exactly cosine distance (scaled by a
        constant because µq,µd are on the unit sphere).  Uniform sigma makes ELK
        and cosine produce identical rankings — defeating the purpose of Gaussian
        retrieval entirely.  Non-uniform sigma breaks this symmetry: dimensions where
        the query has high activation (semantically salient features) receive wider
        variance (more uncertainty), making ELK sensitive to the doc's covariance
        shape in a way that cosine is not.

        Design:
          1. Diversity scale (D): TTR × length_saturation.  Monotonically increases
             with query length and vocabulary diversity, reversing the systematic
             bias of the heuristic augmentation approach.
          2. Per-dimension scale (σ_k = D × |µ_k| + ε): dimensions where the
             query embedding is large receive wider variance.  This creates a
             semantically meaningful non-uniform covariance aligned with the query.
          3. Entity override: capitalised entity diversity can boost D upward.
        """
        import math

        tokens = text.split()
        n = len(tokens)

        if n <= 1:
            # Degenerate single-token query — near-floor non-uniform sigma
            # proportional to |µ_k| so ELK is still different from cosine.
            sigma_diag = np.clip(np.abs(mu) * 0.01 + self.floor * 100, self.floor, None)
            return sigma_diag, np.zeros((mu.shape[0], 0))

        lower_tokens = text.lower().split()
        ttr = len(set(lower_tokens)) / n  # type-token ratio in (0,1]
        length_sat = 1.0 - math.exp(-n / 4.0)  # length saturation in (0,1)
        diversity = ttr * length_sat  # in (0,1)

        # Entity diversity override when capitalised entities are present
        entities = self._extract_entities(text)
        if entities:
            entity_div = len(set(entities)) / max(len(entities), 1)
            diversity = max(diversity, entity_div * length_sat)

        diversity = max(0.05, min(1.0, diversity))

        # Non-uniform per-dimension sigma: scale each dim by diversity × |µ_k|.
        # Dimensions where the query has high activation (semantically salient
        # features for this query) receive proportionally higher variance,
        # meaning ELK is less certain about retrieval along those directions.
        # A small uniform floor (diversity × 0.01) prevents zero-sigma dimensions.
        # XRC-E Fix: Scale by 1e-2 to match the ~1e-4 empirical variance of documents.
        # Without this, query variance (~1e-2) completely washes out document variance
        # in the combined_var sum, causing ELK to collapse back to Cosine.
        sigma_diag = diversity * (np.abs(mu) + 0.01) * 1e-2
        sigma_diag = np.clip(sigma_diag, self.floor, None)

        return sigma_diag, np.zeros((mu.shape[0], 0))
