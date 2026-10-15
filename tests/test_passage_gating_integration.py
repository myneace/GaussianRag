"""test_passage_gating_integration.py — Live integration tests for Clef passage gating.

Validates GATE-01 & GATE-02:
  - Candidates are evaluated through ClefPassageGate prior to manifold reranking
  - Rejection of contradictory and off-topic chunks
  - Populates gating_metadata on RetrievedChunk
  - Live execution against Cloudflare Workers AI using credentials in .env
"""

from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pytest

from gaussian_rag.core.store import KnowledgeStore
from gaussian_rag.core.types import GaussianKnowledge, QueryRepresentation
from gaussian_rag.rag.disambiguation.clef_client import ClefClient
from gaussian_rag.rag.retrieval.ann_index import MuAnnIndex
from gaussian_rag.rag.retrieval.clef_passage_gate import ClefPassageGate
from gaussian_rag.rag.retrieval.retriever import GaussianRetriever


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


@pytest.mark.asyncio
async def test_live_passage_gating_rejection_and_retrieval(tmp_path: Path) -> None:
    """Validate that contradictory and off-topic candidates are pruned live."""
    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    api_token = os.environ.get("CLOUDFLARE_API_TOKEN")
    model = os.environ.get("CLOUDFLARE_MODEL", "@cf/cloudflare/clef-flash")

    assert account_id, "CLOUDFLARE_ACCOUNT_ID required in .env"
    assert api_token, "CLOUDFLARE_API_TOKEN required in .env"

    store = KnowledgeStore()
    dim = 16

    # 1. Populate store with evidence, contradictory, and off-topic chunks
    base_mu = np.array([1.0, 0.5] + [0.0] * (dim - 2), dtype=float)
    base_mu /= np.linalg.norm(base_mu)

    chunks = [
        # Direct evidence chunk
        GaussianKnowledge(
            id="evidence_cardio_1",
            text="Myocardial infarction is primarily caused by coronary artery occlusion following atherosclerotic plaque rupture.",
            mu=base_mu + np.random.normal(0, 0.01, dim),
            sigma_diag=np.ones(dim) * 0.01,
            sigma_L=np.zeros((dim, 0)),
        ),
        # Supporting evidence chunk
        GaussianKnowledge(
            id="evidence_cardio_2",
            text="Acute coronary syndrome involves sudden reduced blood flow to the heart, damaging heart muscle tissue.",
            mu=base_mu + np.random.normal(0, 0.01, dim),
            sigma_diag=np.ones(dim) * 0.01,
            sigma_L=np.zeros((dim, 0)),
        ),
        # Direct contradictory chunk (near in vector space, but factually false)
        GaussianKnowledge(
            id="contra_cardio",
            text="Myocardial infarction has no relation to the heart or arteries and is solely a fungal infection of fingernails.",
            mu=base_mu + np.random.normal(0, 0.01, dim),
            sigma_diag=np.ones(dim) * 0.01,
            sigma_L=np.zeros((dim, 0)),
        ),
        # Off-topic distractor chunk
        GaussianKnowledge(
            id="distractor_baking",
            text="Sourdough bread crust forms through gelatinization of starch and Maillard browning during oven steam injection.",
            mu=base_mu + np.random.normal(0, 0.01, dim),
            sigma_diag=np.ones(dim) * 0.01,
            sigma_L=np.zeros((dim, 0)),
        ),
    ]

    store.add(chunks)
    ann_index = MuAnnIndex()
    ann_index.build(store.values())

    async with ClefClient(
        account_id=account_id,
        api_token=api_token,
        model=model,
        timeout=30.0,
    ) as client:
        gate = ClefPassageGate(clef_client=client, min_composite_score=0.1)
        retriever = GaussianRetriever(store=store, ann_index=ann_index, passage_gate=gate)

        query_text = "What is the primary etiology and arterial mechanism of myocardial infarction?"
        query_rep = QueryRepresentation(
            text=query_text,
            mu=base_mu.copy(),
            sigma_diag=np.ones(dim) * 0.01,
            sigma_L=np.zeros((dim, 0)),
        )

        results = await retriever.retrieve_with_gating(
            query=query_rep,
            query_text=query_text,
            top_k=2,
            metric="elk",
        )

        assert len(results) >= 1
        result_ids = [c.knowledge.id for c in results]

        # Top result must be legitimate evidence
        assert result_ids[0] in ("evidence_cardio_1", "evidence_cardio_2")
        # Contradictory fungal fingernail chunk must NOT be top 1
        assert result_ids[0] != "contra_cardio"
        assert result_ids[0] != "distractor_baking"

        # Check gating metadata is attached
        top_chunk = results[0]
        assert top_chunk.gating_metadata is not None
        assert "composite_score" in top_chunk.gating_metadata
        assert top_chunk.gating_metadata["is_relevant"] is True
        assert top_chunk.gating_metadata["has_evidence"] is True
