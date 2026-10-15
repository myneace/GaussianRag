"""Agent 3: Cross-Context Validator & Field Mapper.

Constructs a semantic field over all SenseNodes by:
  1. Computing pairwise compatibility edges (via bhattacharyya overlap).
  2. Applying Gaussian kernel smoothing to derive field_weights.
  3. Computing Shannon entropy over field_weights.
  4. Flagging CONFLICTED senses (same term, both confidence > 0.45).
  5. Recommending selected_sense_id or signalling multi-sense routing.

Thresholds:
  ENTROPY_SINGLE  ≤ 1.5 bits → confident single-sense routing
  ENTROPY_DUAL    ≤ 2.5 bits → dual-sense routing
  ENTROPY_HIGH    > 2.5 bits → clarification fallback
"""
from __future__ import annotations

import math
from collections import defaultdict

from gaussian_rag.core.gaussian_embedding import GaussianEmbedder
from gaussian_rag.core.information_geometry import bhattacharyya
from gaussian_rag.core.types import SenseNode

ENTROPY_SINGLE = 1.5
ENTROPY_DUAL = 2.5


def _field_entropy(weights: dict[str, float]) -> float:
    """Shannon entropy H = -Σ w_i log2(w_i) over the field weight dict."""
    entropy = 0.0
    for w in weights.values():
        if w > 0:
            entropy -= w * math.log2(w)
    return entropy


def _kernel_smooth(
    raw_weights: dict[str, float],
    compatibility: dict[tuple[str, str], float],
    alpha: float = 0.3,
) -> dict[str, float]:
    """Apply one step of Gaussian kernel smoothing.

    Each sense's weight is updated as:
        w_i = (1-alpha) * w_i + alpha * mean(w_j * compat(i,j) for j != i)
    """
    sense_ids = list(raw_weights.keys())
    smoothed = {}
    for sid in sense_ids:
        neighbors = [
            raw_weights[other] * compatibility.get((sid, other), 0.0)
            for other in sense_ids if other != sid
        ]
        neighbor_mean = sum(neighbors) / max(len(neighbors), 1)
        smoothed[sid] = (1.0 - alpha) * raw_weights[sid] + alpha * neighbor_mean

    # Renormalise
    total = sum(smoothed.values()) or 1.0
    return {sid: w / total for sid, w in smoothed.items()}


def build_field(
    senses: list[SenseNode],
    embedder: GaussianEmbedder,
    smoothing_alpha: float = 0.3,
    conflict_threshold: float = 0.45,
) -> tuple[dict[str, float], list[str], float, str | None]:
    """Agent 3: compute the contextual semantic field.

    Returns:
        field_weights:     sense_id -> normalised weight.
        conflict_flags:    list of sense_ids involved in conflicts.
        entropy:           Shannon entropy of field_weights.
        selected_sense_id: highest-weight sense if entropy ≤ ENTROPY_SINGLE, else None.
    """
    if not senses:
        return {}, [], 0.0, None

    # Start from Agent 2 confidences as raw weights
    raw_weights = {s.sense_id: s.confidence for s in senses}
    total = sum(raw_weights.values()) or 1.0
    raw_weights = {sid: w / total for sid, w in raw_weights.items()}

    # Build sense gaussian map for Bhattacharyya compatibility
    sense_map = {s.sense_id: s for s in senses}
    gaussian_map = {s.sense_id: s.as_gaussian() for s in senses}

    # Pairwise compatibility (Bhattacharyya distance → similarity)
    compatibility: dict[tuple[str, str], float] = {}
    sense_ids = list(gaussian_map.keys())
    for i, sid_a in enumerate(sense_ids):
        for sid_b in sense_ids[i + 1:]:
            try:
                dist = bhattacharyya(gaussian_map[sid_a], gaussian_map[sid_b])
                # Convert distance to similarity in [0, 1]
                sim = math.exp(-dist)
            except Exception:
                sim = 0.0
            # Cross-term: compatible; same-term: penalise (competing senses)
            if sense_map[sid_a].term == sense_map[sid_b].term:
                sim *= 0.1  # reduce similarity for same-term competitors
            compatibility[(sid_a, sid_b)] = sim
            compatibility[(sid_b, sid_a)] = sim

    # Apply smoothing
    field_weights = _kernel_smooth(raw_weights, compatibility, alpha=smoothing_alpha)

    # Conflict detection: same-term senses both above threshold
    term_groups: dict[str, list[str]] = defaultdict(list)
    for s in senses:
        term_groups[s.term].append(s.sense_id)

    conflict_flags: list[str] = []
    for term, ids in term_groups.items():
        conflicted = [sid for sid in ids if field_weights.get(sid, 0) > conflict_threshold]
        if len(conflicted) >= 2:
            conflict_flags.extend(conflicted)

    entropy = _field_entropy(field_weights)

    # Routing recommendation
    if entropy <= ENTROPY_SINGLE:
        selected = max(field_weights, key=field_weights.__getitem__)
    else:
        selected = None

    return field_weights, conflict_flags, entropy, selected
