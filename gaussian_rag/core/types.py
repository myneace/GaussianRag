from __future__ import annotations

import time

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _confidence_from_trace(trace_sigma: float) -> float:
    return 1.0 / (1.0 + max(trace_sigma, 0.0))


def _uncertainty_label(confidence: float) -> str:
    if confidence >= 0.5:
        return "low"
    if confidence >= 0.2:
        return "medium"
    return "high"


@dataclass(slots=True)
class GaussianKnowledge:
    id: str
    text: str
    mu: np.ndarray
    sigma_diag: np.ndarray
    sigma_L: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.mu = np.asarray(self.mu, dtype=float)
        self.sigma_diag = np.asarray(self.sigma_diag, dtype=float)
        self.sigma_L = np.asarray(self.sigma_L, dtype=float)
        if self.mu.ndim != 1:
            raise ValueError("mu must be a 1D vector")
        if self.sigma_diag.shape != self.mu.shape:
            raise ValueError("sigma_diag must match mu shape")
        if self.sigma_L.ndim != 2:
            raise ValueError("sigma_L must be 2D")
        if self.sigma_L.shape[0] != self.mu.shape[0]:
            raise ValueError("sigma_L row count must match mu dimension")

    @property
    def sigma(self) -> np.ndarray:
        return np.diag(self.sigma_diag) + self.sigma_L @ self.sigma_L.T

    @property
    def trace_sigma(self) -> float:
        return float(np.sum(self.sigma_diag) + np.sum(self.sigma_L * self.sigma_L))

    @property
    def confidence(self) -> float:
        return _confidence_from_trace(self.trace_sigma)

    @property
    def uncertainty_label(self) -> str:
        return _uncertainty_label(self.confidence)


@dataclass(slots=True)
class QueryRepresentation:
    text: str
    mu: np.ndarray
    sigma_diag: np.ndarray | None = None
    sigma_L: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.mu = np.asarray(self.mu, dtype=float)
        if self.mu.ndim != 1:
            raise ValueError("mu must be a 1D vector")
        if self.sigma_diag is not None:
            self.sigma_diag = np.asarray(self.sigma_diag, dtype=float)
        if self.sigma_L is not None:
            self.sigma_L = np.asarray(self.sigma_L, dtype=float)

    @property
    def has_covariance(self) -> bool:
        return self.sigma_diag is not None and self.sigma_L is not None

    def as_gaussian(self) -> GaussianKnowledge:
        sigma_diag = (
            self.sigma_diag if self.sigma_diag is not None else np.zeros_like(self.mu)
        )
        sigma_L = (
            self.sigma_L
            if self.sigma_L is not None
            else np.zeros((self.mu.shape[0], 0))
        )
        return GaussianKnowledge(
            id="query",
            text=self.text,
            mu=self.mu,
            sigma_diag=sigma_diag,
            sigma_L=sigma_L,
            metadata=self.metadata,
        )


@dataclass(slots=True)
class RetrievedChunk:
    knowledge: GaussianKnowledge
    w2_distance: float
    fr_distance: float | None
    elk_score: float | None
    confidence: float
    uncertainty_label: str
    rank: int


# ---------------------------------------------------------------------------
# Disambiguation types
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SenseNode:
    """One disambiguated interpretation of a polysemous term."""

    sense_id: str  # stable: hash(term + domain + version)
    term: str  # original ambiguous token/phrase
    domain: str  # e.g. "technology/corporate"
    confidence: float  # 0.0–1.0
    context_hints: list[str]  # discriminative tokens for this sense
    supporting_evidence: list[str]
    retrieval_query: str  # expanded query string for retrieval
    mu_anchor: np.ndarray | None = None  # embedding anchor for this sense
    sigma_anchor: np.ndarray | None = None  # diagonal covariance for this sense
    sigma_L_anchor: np.ndarray | None = None
    manifold_neighbor_id: str | None = None
    anchor_neighbor_id: str | None = None
    ttl_hours: int = 24
    version: int = 1
    created_at: float = field(default_factory=time.time)

    def is_expired(self) -> bool:
        age_hours = (time.time() - self.created_at) / 3600.0
        return age_hours > self.ttl_hours

    def as_gaussian(self) -> GaussianKnowledge:
        """Convert this sense to a GaussianKnowledge node for geometry calculations."""
        mu = self.mu_anchor if self.mu_anchor is not None else np.zeros(384)
        sigma_diag = (
            self.sigma_anchor if self.sigma_anchor is not None else np.zeros_like(mu)
        )
        sigma_l = (
            self.sigma_L_anchor
            if self.sigma_L_anchor is not None
            else np.zeros((mu.shape[0], 0))
        )
        return GaussianKnowledge(
            id=self.sense_id,
            text=self.retrieval_query,
            mu=mu,
            sigma_diag=sigma_diag,
            sigma_L=sigma_l,
            metadata={"domain": self.domain, "term": self.term},
        )


@dataclass(slots=True)
class MemoryOps:
    """Instructions for the session sense cache after a disambiguation pass."""

    upsert_nodes: list[SenseNode] = field(default_factory=list)
    prune_ids: list[str] = field(default_factory=list)
    context_window: list[str] = field(default_factory=list)  # active sense_ids


@dataclass(slots=True)
class DisambiguationPacket:
    """Master output of the parallel agent disambiguation pipeline."""

    disambiguation_id: str
    original_text: str
    senses: list[SenseNode]
    field_weights: dict[str, float]  # sense_id -> normalised weight
    entropy: float  # Shannon entropy over field_weights
    selected_sense_id: str | None  # None = multi-sense routing
    memory_ops: MemoryOps
    fallback_triggered: bool = False
