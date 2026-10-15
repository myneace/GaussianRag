"""test_clef_passage_gate.py — Unit and live Cloudflare Workers AI integration tests for ClefPassageGate.

Validates GATE-01:
  - 4-question Clef gate formulation (is_relevant, has_evidence, is_contradictory, quality_score)
  - Composite scoring and thresholding
  - Live execution against Cloudflare Workers AI using real credentials from .env
"""

from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pytest

from gaussian_rag.core.types import GaussianKnowledge
from gaussian_rag.rag.disambiguation.clef_client import ClefClient
from gaussian_rag.rag.retrieval.clef_passage_gate import (
    ClefPassageGate,
    PassageGatingResult,
)


def _ensure_env() -> None:
    """Ensure credentials from .env are populated in os.environ."""
    env_file = Path(__file__).resolve().parents[2] / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


@pytest.fixture(autouse=True)
def load_env() -> None:
    _ensure_env()


def test_passage_gating_result_properties() -> None:
    res = PassageGatingResult(
        chunk_id="chunk_1",
        is_relevant=True,
        has_evidence=True,
        is_contradictory=False,
        quality_score=3.0,
        relevance_prob=0.95,
        evidence_prob=0.90,
        contradiction_prob=0.05,
        composite_score=0.95 * 0.90 * 0.95 * (3.0 / 3.0),
        passed=True,
    )
    assert res.chunk_id == "chunk_1"
    assert res.is_relevant is True
    assert res.has_evidence is True
    assert res.is_contradictory is False
    assert res.quality_score == 3.0
    assert res.passed is True
    assert res.composite_score > 0.7


@pytest.mark.asyncio
async def test_live_clef_passage_gate_relevant_and_contradiction() -> None:
    """Live verification against Cloudflare Workers AI Clef-Flash evaluating real passages."""
    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    api_token = os.environ.get("CLOUDFLARE_API_TOKEN")
    model = os.environ.get("CLOUDFLARE_MODEL", "@cf/cloudflare/clef-flash")

    assert account_id, "CLOUDFLARE_ACCOUNT_ID required in .env"
    assert api_token, "CLOUDFLARE_API_TOKEN required in .env"

    async with ClefClient(
        account_id=account_id,
        api_token=api_token,
        model=model,
        timeout=30.0,
    ) as client:
        gate = ClefPassageGate(clef_client=client, min_composite_score=0.1)

        query = "What is the time complexity of the Woodbury low-rank matrix identity?"

        # 1. Relevant, high-evidence passage
        rel_passage = (
            "The Woodbury matrix identity evaluates low-rank covariance updates in O(d * r^2) "
            "rather than dense O(d^3) time by using capacitance matrix factorization."
        )
        res_rel = await gate.evaluate_passage(query, "doc_woodbury_valid", rel_passage)

        assert res_rel.chunk_id == "doc_woodbury_valid"
        assert res_rel.is_relevant is True, f"Expected relevant, got {res_rel}"
        assert res_rel.relevance_prob >= 0.5
        assert res_rel.passed is True
        assert res_rel.composite_score > 0.1

        # 2. Contradictory passage
        contra_passage = (
            "The Woodbury matrix identity is fundamentally impossible to compute in low-rank time "
            "and requires strictly intractable O(d^6) execution, disproving all matrix speedups."
        )
        res_contra = await gate.evaluate_passage(query, "doc_woodbury_contra", contra_passage)

        assert res_contra.chunk_id == "doc_woodbury_contra"
        # Contradiction flag or contradiction prob should be elevated
        assert res_contra.is_contradictory is True or res_contra.contradiction_prob >= 0.3


@pytest.mark.asyncio
async def test_live_clef_passage_gate_batch_pruning() -> None:
    """Live batch pruning of candidate pool down to top high-signal passages."""
    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    api_token = os.environ.get("CLOUDFLARE_API_TOKEN")

    async with ClefClient(
        account_id=account_id,
        api_token=api_token,
        timeout=30.0,
    ) as client:
        gate = ClefPassageGate(clef_client=client)

        query = "How does coronary artery disease cause myocardial infarction?"
        dim = 16

        def _make_chunk(cid: str, text: str) -> GaussianKnowledge:
            return GaussianKnowledge(
                id=cid,
                text=text,
                mu=np.ones(dim) / np.sqrt(dim),
                sigma_diag=np.ones(dim) * 0.1,
                sigma_L=np.zeros((dim, 0)),
            )

        candidates = [
            _make_chunk(
                "cand_1",
                "Coronary artery disease leads to atherosclerotic plaque rupture, forming a thrombus "
                "that occludes blood flow to the myocardium, resulting in myocardial infarction.",
            ),
            _make_chunk(
                "cand_2",
                "Baking sourdough bread requires water, flour, salt, and active wild yeast fermentation "
                "at ambient room temperature.",
            ),
            _make_chunk(
                "cand_3",
                "Myocardial infarction is myocardial necrosis caused by acute ischemia from coronary plaque rupture.",
            ),
        ]

        pruned = await gate.evaluate_candidates_batch(query, candidates, top_k=2)

        assert len(pruned) >= 1
        survivor_ids = [cand.id for cand, _ in pruned]
        # Sourdough bread (cand_2) must be excluded or ranked lowest
        assert "cand_1" in survivor_ids
        assert survivor_ids[0] in ("cand_1", "cand_3")
        if "cand_2" in survivor_ids:
            # If included, its composite score must be lower than cand_1
            score_map = {cand.id: res.composite_score for cand, res in pruned}
            assert score_map["cand_1"] > score_map["cand_2"]
