"""clef_passage_gate.py — Unified 4-question Clef passage gate for candidate pool pruning.

Implements GATE-01 and GATE-02:
  - 4-question Clef-Flash decision gate (is_relevant, has_evidence, is_contradictory, quality_score)
  - Concurrent batch evaluation with connection pooling and semaphore bounding
  - Composite gating score calculation penalizing contradiction and rewarding grounded evidence
  - Prunes candidate pools from initial retrieval down to high-signal passages
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from gaussian_rag.core.types import GaussianKnowledge
from gaussian_rag.rag.disambiguation.clef_client import (
    ClefClient,
    NoulQuestion,
    ScoreQuestion,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PassageGatingResult:
    """Detailed evaluation result for a single candidate passage."""

    chunk_id: str
    is_relevant: bool
    has_evidence: bool
    is_contradictory: bool
    quality_score: float  # Scale 1.0 to 3.0
    relevance_prob: float
    evidence_prob: float
    contradiction_prob: float
    composite_score: float
    passed: bool
    raw_answers: dict[str, Any] = field(default_factory=dict)


class ClefPassageGate:
    """Evaluates candidate passages via a 4-question Cloudflare Workers AI Clef gate."""

    def __init__(
        self,
        clef_client: ClefClient,
        concurrency_limit: int = 10,
        min_composite_score: float = 0.15,
        max_contradiction_prob: float = 0.5,
    ) -> None:
        self.clef_client = clef_client
        self.concurrency_limit = concurrency_limit
        self.min_composite_score = min_composite_score
        self.max_contradiction_prob = max_contradiction_prob
        self._semaphore = asyncio.Semaphore(concurrency_limit)

    def _build_questions(self, query: str, passage: str) -> dict[str, Any]:
        return {
            "is_relevant": NoulQuestion(
                instructions="Is this candidate passage relevant to answering the query?"
            ),
            "has_evidence": NoulQuestion(
                instructions="Does this candidate passage contain direct factual evidence or context to answer the query?"
            ),
            "is_contradictory": NoulQuestion(
                instructions="Does this candidate passage contradict the established facts or provide false information?"
            ),
            "quality_score": ScoreQuestion(
                instructions="Rate the overall informational quality and coherence of this candidate passage on a 1-3 scale.",
                criteria=[
                    "1 - Low quality, noisy, or unhelpful",
                    "2 - Moderate informative content",
                    "3 - High quality, authoritative factual evidence",
                ],
            ),
        }

    async def evaluate_passage(
        self, query: str, chunk_id: str, text: str
    ) -> PassageGatingResult:
        """Evaluate a single passage through the 4-question Clef gate."""
        questions = self._build_questions(query, text)
        state = f"Query: {query}\nCandidate Passage: {text}"

        try:
            resp = await self.clef_client.request_decision(state, questions)
            answers = resp.answers

            # Helper to extract probability for positive ('yes'/'true') outcome
            def _prob(ans: Any, default: float) -> float:
                if not ans or not ans.probabilities:
                    return default
                probs = ans.probabilities
                if "yes" in probs:
                    return float(probs["yes"])
                if "true" in probs:
                    return float(probs["true"])
                return default

            # 1. Relevance
            q_rel = answers.get("is_relevant")
            is_relevant = bool(q_rel.answer) if q_rel else True
            p_rel = _prob(q_rel, 1.0 if is_relevant else 0.0)

            # 2. Evidence
            q_evi = answers.get("has_evidence")
            has_evidence = bool(q_evi.answer) if q_evi else True
            p_evi = _prob(q_evi, 1.0 if has_evidence else 0.0)

            # 3. Contradiction
            q_con = answers.get("is_contradictory")
            is_contradictory = bool(q_con.answer) if q_con else False
            p_con = _prob(q_con, 1.0 if is_contradictory else 0.0)

            # 4. Quality
            q_qual = answers.get("quality_score")
            try:
                qual_val = (
                    float(q_qual.answer)
                    if q_qual and q_qual.answer is not None
                    else 2.0
                )
            except (ValueError, TypeError):
                qual_val = 2.0
            norm_qual = max(0.1, min(1.0, qual_val / 3.0))

            # Composite score: rewards relevance & evidence, strongly penalizes contradiction
            composite = p_rel * p_evi * (1.0 - p_con) * norm_qual

            # Rejection criteria:
            # - Contradiction probability exceeds threshold
            # - Irrelevant or failing minimum composite threshold
            passed = (
                (not is_contradictory or p_con < self.max_contradiction_prob)
                and p_con <= self.max_contradiction_prob
                and (is_relevant or p_rel >= 0.4)
                and (composite >= self.min_composite_score)
            )

            raw_dict = {
                k: {"answer": v.answer, "probabilities": v.probabilities}
                for k, v in answers.items()
            }

            return PassageGatingResult(
                chunk_id=chunk_id,
                is_relevant=is_relevant,
                has_evidence=has_evidence,
                is_contradictory=is_contradictory,
                quality_score=qual_val,
                relevance_prob=p_rel,
                evidence_prob=p_evi,
                contradiction_prob=p_con,
                composite_score=composite,
                passed=passed,
                raw_answers=raw_dict,
            )

        except Exception as exc:
            logger.warning(
                "Clef passage gating encountered error for chunk %s: %s; falling through.",
                chunk_id,
                exc,
            )
            # Resilient neutral fallback
            return PassageGatingResult(
                chunk_id=chunk_id,
                is_relevant=True,
                has_evidence=True,
                is_contradictory=False,
                quality_score=2.0,
                relevance_prob=0.5,
                evidence_prob=0.5,
                contradiction_prob=0.0,
                composite_score=0.33,
                passed=True,
                raw_answers={"error": str(exc)},
            )

    async def evaluate_candidates_batch(
        self,
        query: str,
        candidates: list[GaussianKnowledge],
        top_k: int = 8,
    ) -> list[tuple[GaussianKnowledge, PassageGatingResult]]:
        """Evaluate a batch of candidate chunks concurrently and return top_k pruned survivors."""
        if not candidates:
            return []

        async def _gate_chunk(
            cand: GaussianKnowledge,
        ) -> tuple[GaussianKnowledge, PassageGatingResult]:
            async with self._semaphore:
                res = await self.evaluate_passage(query, cand.id, cand.text)
                return cand, res

        results = await asyncio.gather(*[_gate_chunk(c) for c in candidates])

        # Filter passed chunks
        passed = [pair for pair in results if pair[1].passed]

        # If strict filtering pruned too aggressively (all pruned), take top scoring survivors
        if not passed:
            logger.info(
                "All candidates pruned by strict filter; falling back to top composite scores."
            )
            passed = sorted(results, key=lambda p: -p[1].composite_score)
        else:
            passed = sorted(passed, key=lambda p: -p[1].composite_score)

        return passed[:top_k]
