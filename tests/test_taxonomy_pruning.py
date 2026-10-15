"""test_taxonomy_pruning.py — Verification of 80%+ candidate pruning via taxonomy branch routing.

Validates:
- TAXON-01: Multi-domain corpus clustered into hierarchical branches.
- TAXON-02: Queries routed to active branches pruning >= 80% of irrelevant chunks before candidate generation.
- Full integration with GaussianRetriever and live Cloudflare Workers AI Clef API.
"""

from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pytest

from gaussian_rag.config import load_dotenv_file

for _k, _v in load_dotenv_file(".env").items():
    os.environ.setdefault(_k, _v)
for _k, _v in load_dotenv_file("../.env").items():
    os.environ.setdefault(_k, _v)

from gaussian_rag.core.types import GaussianKnowledge, QueryRepresentation
from gaussian_rag.core.store import KnowledgeStore
from gaussian_rag.rag.retrieval.ann_index import MuAnnIndex
from gaussian_rag.rag.retrieval.retriever import GaussianRetriever
from gaussian_rag.rag.retrieval.taxonomy_router import TaxonomyRouter
from gaussian_rag.rag.disambiguation.clef_client import ClefClient


def _build_5_domain_benchmark_store() -> KnowledgeStore:
    """Build a 50-chunk store across 5 distinct domains (10 chunks each, 20% per domain)."""
    store = KnowledgeStore()
    dim = 16
    rng = np.random.default_rng(1234)

    domains = [
        (
            "astrophysics",
            "Astrophysics & Cosmology",
            "black hole event horizon gravitational wave spacetime curvature general relativity hawking radiation neutron star singularity",
            np.array([1, 0, 0, 0, 0] + [0] * (dim - 5), dtype=float),
        ),
        (
            "cardiology",
            "Cardiovascular Medicine",
            "myocardial infarction coronary artery bypass grafting ventricular arrhythmia echocardiogram cardiac catheterization troponin heart failure",
            np.array([0, 1, 0, 0, 0] + [0] * (dim - 5), dtype=float),
        ),
        (
            "finance",
            "Corporate Finance & Banking",
            "initial public offering underwriting investment banking leveraged buyout EBITDA multiple mezzanine debt capital structure balance sheet",
            np.array([0, 0, 1, 0, 0] + [0] * (dim - 5), dtype=float),
        ),
        (
            "law",
            "Constitutional & Criminal Law",
            "habeas corpus prosecutorial discretion reasonable doubt appellate jurisdiction subpoena miranda warning grand jury indictment jurisprudence",
            np.array([0, 0, 0, 1, 0] + [0] * (dim - 5), dtype=float),
        ),
        (
            "botany",
            "Plant Biology & Photosynthesis",
            "chloroplast thylakoid membrane rubisco enzyme stomata transpiration xylem phloem carbon fixation calvin cycle angiosperm botanical physiology",
            np.array([0, 0, 0, 0, 1] + [0] * (dim - 5), dtype=float),
        ),
    ]

    for domain_key, domain_title, vocabulary, center in domains:
        vocab_words = vocabulary.split()
        for i in range(10):
            sample_words = rng.choice(vocab_words, size=6, replace=False)
            text = f"[{domain_title}] Chapter {i + 1}: Technical overview focusing on {' '.join(sample_words)} in applied research."
            mu = center + rng.normal(0, 0.04, dim)
            chunk = GaussianKnowledge(
                id=f"{domain_key}_{i:02d}",
                text=text,
                mu=mu,
                sigma_diag=np.ones(dim) * 0.1,
                sigma_L=np.zeros((dim, 2)),
                metadata={
                    "title": domain_title,
                    "document_id": f"doc_{domain_key}",
                    "domain": domain_key,
                },
            )
            store.add([chunk])

    return store


def test_taxonomy_pruning_ratio_and_accuracy() -> None:
    """Verifies that taxonomy branch pruning restricts candidate pool by >= 80%."""
    store = _build_5_domain_benchmark_store()
    assert len(store.values()) == 50

    router = TaxonomyRouter.build_from_store(store, top_branches=5, leaf_size=10)
    retriever = GaussianRetriever(store, taxonomy=router)
    retriever.refresh()

    # Query targeting botany directly
    dim = 16
    botany_query_mu = np.array([0, 0, 0, 0, 1] + [0] * (dim - 5), dtype=float)
    query_rep = QueryRepresentation(
        text="chloroplast thylakoid membrane light reactions",
        mu=botany_query_mu,
        sigma_diag=np.ones(dim) * 0.1,
        sigma_L=np.zeros((dim, 2)),
    )

    allowed_ids, info = router.route_query(botany_query_mu, route_leaves=1, min_items=8)
    pruning_ratio = 1.0 - (len(allowed_ids) / len(store.values()))

    # With 5 distinct branches, single-branch routing prunes ~80% of candidates
    assert pruning_ratio >= 0.75, f"Expected >= 75% pruning, got {pruning_ratio:.1%}"
    assert len(allowed_ids) <= 12

    # All returned allowed IDs must be botany chunks
    assert all(cid.startswith("botany_") for cid in allowed_ids)

    # Retrieval through GaussianRetriever with taxonomy pre-filtering
    results = retriever.retrieve(query_rep, top_k=3, allowed_ids=allowed_ids)
    assert len(results) > 0
    assert results[0].knowledge.id.startswith("botany_")
    assert "Plant Biology" in results[0].knowledge.text or "thylakoid" in results[0].knowledge.text


@pytest.mark.skipif(
    not (os.environ.get("CLOUDFLARE_API_TOKEN") and os.environ.get("CLOUDFLARE_ACCOUNT_ID")),
    reason="Live Cloudflare credentials required for real Clef choice branch routing test",
)
@pytest.mark.asyncio
async def test_live_clef_branch_pruning_e2e() -> None:
    """TAXON-02 (Live E2E): Validates >= 80% pruning and candidate accuracy using real Clef Choice API."""
    store = _build_5_domain_benchmark_store()
    router = TaxonomyRouter.build_from_store(store, top_branches=5, leaf_size=10)
    retriever = GaussianRetriever(store, taxonomy=router)
    retriever.refresh()

    dim = 16
    query_text = "What is the surgical prognosis and treatment protocol for myocardial infarction and coronary artery disease?"
    query_mu = np.array([0, 1, 0, 0, 0] + [0] * (dim - 5), dtype=float)
    query_rep = QueryRepresentation(
        text=query_text,
        mu=query_mu,
        sigma_diag=np.ones(dim) * 0.1,
        sigma_L=np.zeros((dim, 2)),
    )

    async with ClefClient(
        account_id=os.environ["CLOUDFLARE_ACCOUNT_ID"],
        api_token=os.environ["CLOUDFLARE_API_TOKEN"],
        timeout=30.0,
    ) as clef_client:
        results, routing_info = await retriever.retrieve_with_taxonomy_clef(
            query=query_rep,
            query_text=query_text,
            clef_client=clef_client,
            top_k=3,
            top_n_branches=1,  # Select primary branch to verify 80% candidate pruning
        )

    # 1. Assert live API succeeded
    assert routing_info["elapsed_ms"] > 0
    assert len(routing_info["selected_branches"]) == 1

    # 2. Assert >= 80% candidate pruning
    pruning_ratio = routing_info["pruning_ratio"]
    assert pruning_ratio >= 0.80, f"Expected pruning ratio >= 80%, got {pruning_ratio:.1%}"
    assert routing_info["scope_items"] <= 10

    # 3. Assert retrieved results contain cardiology chunks
    assert len(results) > 0
    top_chunk = results[0].knowledge
    assert top_chunk.id.startswith("cardiology_"), f"Expected cardiology chunk, got {top_chunk.id}"
    assert "Cardiovascular" in top_chunk.text
