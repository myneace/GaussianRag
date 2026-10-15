"""clef_disambiguator.py — Non-autoregressive Clef-based ambiguity detection and sense selection.

Replaces the slow LLM-driven ``query_dissolver`` / ``sense_generator`` path with
two millisecond-level Clef decision gates:

1. **Ambiguity gate** (``noul``): "Does this query contain semantic ambiguity?"
   - If probability < 0.5 → unambiguous, return immediately (< 40 ms).

2. **Sense-selection gate** (``choice``): Routes polysemous queries to dominant
   interpretation domains from a lightweight heuristic candidate pool.

Design decisions:
- Candidate domains are derived using fast heuristic noun-phrase extraction
  (no LLM, no embedding) to feed the Clef ``choice`` question.
- ``disambiguate_fast()`` in pipeline.py uses this as a pre-filter.
- Backward-compatible: the synchronous ``disambiguate()`` still works for
  LLM-only mode; ``disambiguate_fast()`` is the new fast path.
"""

from __future__ import annotations

import hashlib
import re
import uuid
import logging
from typing import Any

from gaussian_rag.core.types import DisambiguationPacket, MemoryOps, SenseNode
from gaussian_rag.rag.disambiguation.clef_client import (
    ChoiceQuestion,
    ClefClient,
    NoulQuestion,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Domain heuristics — fast candidate pool (no LLM required)
# ---------------------------------------------------------------------------

_DOMAIN_HINTS: list[tuple[str, str, list[str]]] = [
    ("technology", "technology/software", ["software", "code", "api", "function", "library", "framework", "bug", "error", "database", "server", "cloud", "python", "javascript"]),
    ("finance", "finance/market", ["stock", "market", "price", "investment", "fund", "earnings", "revenue", "profit", "loss", "trading", "portfolio"]),
    ("science", "science/research", ["study", "experiment", "hypothesis", "molecule", "research", "paper", "journal", "discovery", "theory", "model"]),
    ("medicine", "medicine/health", ["patient", "treatment", "drug", "symptom", "diagnosis", "clinical", "therapy", "dosage", "disease", "hospital"]),
    ("legal", "legal/contract", ["law", "contract", "court", "clause", "regulation", "compliance", "lawsuit", "attorney", "liability", "patent"]),
    ("business", "business/operations", ["company", "product", "customer", "sales", "strategy", "marketing", "management", "team", "project", "deadline"]),
    ("general", "general/information", ["what", "how", "why", "when", "where", "which", "explain", "describe", "define", "meaning"]),
]

_MIN_CANDIDATES = 2
_MAX_CANDIDATES = 5


def _extract_candidate_domains(text: str) -> list[tuple[str, str]]:
    """Return (label, domain) pairs from heuristic keyword matching.

    Always includes 'general/information' as a fallback candidate.
    Returns at most ``_MAX_CANDIDATES`` items.
    """
    text_lower = text.lower()
    scored: list[tuple[int, str, str]] = []

    for label, domain, keywords in _DOMAIN_HINTS:
        hits = sum(1 for kw in keywords if kw in text_lower)
        if hits > 0:
            scored.append((hits, label, domain))

    # Sort by hits descending; ensure 'general' is always present
    scored.sort(key=lambda x: x[0], reverse=True)

    seen_domains: set[str] = set()
    candidates: list[tuple[str, str]] = []

    for _, label, domain in scored:
        if len(candidates) >= _MAX_CANDIDATES:
            break
        if domain not in seen_domains:
            candidates.append((label, domain))
            seen_domains.add(domain)

    # Always include general as a fallback
    general_domain = "general/information"
    if general_domain not in seen_domains:
        candidates.append(("general", general_domain))

    # Ensure minimum candidates
    while len(candidates) < _MIN_CANDIDATES:
        candidates.append(("general", "general/information"))

    return candidates[:_MAX_CANDIDATES]


def _make_sense_id(text: str, domain: str) -> str:
    raw = f"clef::{text[:32].lower()}::{domain.lower()}"
    return f"clef_{hashlib.sha256(raw.encode()).hexdigest()[:16]}"


def _build_sense_node(text: str, domain: str, confidence: float) -> SenseNode:
    """Build a SenseNode from a Clef-selected domain."""
    label = domain.split("/")[-1]
    hints = [w for w in re.findall(r'\b[a-z]{4,}\b', text.lower()) if len(w) > 3][:4]
    return SenseNode(
        sense_id=_make_sense_id(text, domain),
        term=text[:32],
        domain=domain,
        confidence=confidence,
        context_hints=hints,
        supporting_evidence=[f"Clef-Flash non-autoregressive gate selected domain '{domain}'"],
        retrieval_query=f"{text} ({label})",
    )


# ---------------------------------------------------------------------------
# ClefDisambiguator
# ---------------------------------------------------------------------------

class ClefDisambiguator:
    """Millisecond-level query ambiguity detector and sense selector.

    Uses Cloudflare Clef-Flash to:
    1. Detect ambiguity via a ``noul`` gate (< 40 ms target).
    2. Select dominant interpretation domains via a ``choice`` gate.

    Designed to be instantiated once and reused across requests (stateless,
    thread-safe for async usage).
    """

    AMBIGUITY_GATE_KEY = "is_ambiguous"
    SENSE_SELECT_KEY = "dominant_domain"

    def __init__(self, clef_client: ClefClient, *, ambiguity_threshold: float = 0.5) -> None:
        """
        Args:
            clef_client: Pre-configured :class:`ClefClient` instance.
            ambiguity_threshold: Noul probability above which a query is
                considered ambiguous. Default 0.5.
        """
        self._client = clef_client
        self._threshold = ambiguity_threshold

    async def check_ambiguity(self, text: str) -> tuple[bool, float]:
        """Run the ambiguity noul gate.

        Returns:
            ``(is_ambiguous, probability)`` — probability is the Clef score for 'yes'.
        """
        response = await self._client.request_decision(
            state=text,
            questions={
                self.AMBIGUITY_GATE_KEY: NoulQuestion(
                    instructions=(
                        "Does this query contain semantic ambiguity, polysemy, or multiple "
                        "distinct possible interpretations that would require different "
                        "knowledge to answer correctly? Answer 'yes' if the query could "
                        "reasonably refer to multiple unrelated domains or concepts."
                    )
                )
            },
        )
        answer = response.answers.get(self.AMBIGUITY_GATE_KEY)
        if answer is None:
            logger.warning("ClefDisambiguator: no ambiguity gate answer returned; defaulting to unambiguous")
            return False, 0.0

        # noul returns a bool answer; probabilities may have 'yes'/'no' keys
        prob_yes = answer.probabilities.get("yes", float(answer.answer) if isinstance(answer.answer, (int, float)) else (1.0 if answer.answer else 0.0))
        is_ambiguous = prob_yes >= self._threshold
        return is_ambiguous, prob_yes

    async def select_senses(
        self,
        text: str,
        max_senses: int = 3,
    ) -> list[SenseNode]:
        """Select dominant interpretation domains via Clef ``choice``.

        Builds a heuristic candidate pool, then uses Clef to select the most
        relevant domain(s). Returns up to ``max_senses`` SenseNodes.

        Args:
            text: The query text.
            max_senses: Maximum number of senses to return (1–5).

        Returns:
            List of :class:`SenseNode` instances (1–max_senses items).
        """
        candidates = _extract_candidate_domains(text)
        criteria = {label: f"Interpretation focused on {domain}" for label, domain in candidates}

        response = await self._client.request_decision(
            state=text,
            questions={
                self.SENSE_SELECT_KEY: ChoiceQuestion(
                    instructions=(
                        "Which interpretation domain best describes the primary meaning "
                        "or intent of this query? Consider the most likely context the "
                        "user is asking about."
                    ),
                    criteria=criteria,
                )
            },
        )

        answer = response.answers.get(self.SENSE_SELECT_KEY)
        if answer is None:
            logger.warning("ClefDisambiguator: no sense-select answer; using top candidate")
            label, domain = candidates[0]
            return [_build_sense_node(text, domain, 1.0)]

        probabilities = answer.probabilities  # {"label": probability, ...}

        # Build SenseNode for each candidate with Clef-calibrated confidence,
        # sorted by probability descending, keeping top max_senses
        label_to_domain = dict(candidates)
        scored: list[tuple[float, str, str]] = []

        for label, domain in candidates:
            prob = probabilities.get(label, 0.0)
            scored.append((prob, label, domain))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:max_senses]

        # Normalize probabilities among selected senses
        total = sum(p for p, _, _ in top) or 1.0
        nodes = [
            _build_sense_node(text, domain, round(prob / total, 4))
            for prob, _label, domain in top
            if prob > 0.0
        ]

        return nodes or [_build_sense_node(text, label_to_domain.get(candidates[0][0], "general/information"), 1.0)]

    async def route(
        self,
        text: str,
        max_senses: int = 3,
    ) -> tuple[bool, list[SenseNode]]:
        """Full single-round-trip routing: ambiguity check → optional sense selection.

        For unambiguous queries, skips sense selection entirely (one Clef call).
        For ambiguous queries, runs sense selection (two Clef calls total).

        Returns:
            ``(is_ambiguous, senses)`` — senses has exactly one item if unambiguous.
        """
        is_ambiguous, _ = await self.check_ambiguity(text)

        if not is_ambiguous:
            # Fast path: single-sense packet
            sense = SenseNode(
                sense_id=_make_sense_id(text, "general"),
                term=text[:32],
                domain="general",
                confidence=1.0,
                context_hints=[],
                supporting_evidence=["Clef gate: unambiguous — single-sense fast path"],
                retrieval_query=text,
            )
            return False, [sense]

        senses = await self.select_senses(text, max_senses=max_senses)
        return True, senses


# ---------------------------------------------------------------------------
# High-level pipeline integration helper
# ---------------------------------------------------------------------------

async def disambiguate_fast(
    text: str,
    clef_client: ClefClient,
    *,
    max_senses: int = 3,
    ambiguity_threshold: float = 0.5,
) -> DisambiguationPacket:
    """Async fast-path disambiguation using Clef-Flash decision gates.

    Replaces the slow LLM-driven pipeline for queries where sub-100ms
    end-to-end latency is required.

    Args:
        text: Query text to disambiguate.
        clef_client: Active :class:`ClefClient` instance.
        max_senses: Maximum senses to generate for ambiguous queries.
        ambiguity_threshold: Noul probability threshold (0.0–1.0).

    Returns:
        :class:`DisambiguationPacket` compatible with the existing router.
    """
    disambiguator = ClefDisambiguator(clef_client, ambiguity_threshold=ambiguity_threshold)
    is_ambiguous, senses = await disambiguator.route(text, max_senses=max_senses)

    # Build field weights from sense confidences
    total_confidence = sum(s.confidence for s in senses) or 1.0
    field_weights = {
        s.sense_id: round(s.confidence / total_confidence, 6) for s in senses
    }

    # Shannon entropy
    import math
    entropy = 0.0
    for w in field_weights.values():
        if w > 0:
            entropy -= w * math.log2(w)

    selected_sense_id = senses[0].sense_id if len(senses) == 1 else None

    return DisambiguationPacket(
        disambiguation_id=str(uuid.uuid4()),
        original_text=text,
        senses=senses,
        field_weights=field_weights,
        entropy=round(entropy, 6),
        selected_sense_id=selected_sense_id,
        memory_ops=MemoryOps(),
        fallback_triggered=False,
    )
