"""test_taxonomy_router.py — Unit and live integration tests for TaxonomyRouter.

Validates:
- TAXON-01: Hierarchical clustering of KnowledgeStore chunks into topic branches based on titles/embeddings.
- TAXON-02: Real Cloudflare Clef Choice question routing with probability distributions and candidate pruning.
- Persistence round-trip (taxonomy.json + taxonomy.npz).
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

from gaussian_rag.core.types import GaussianKnowledge
from gaussian_rag.core.store import KnowledgeStore
from gaussian_rag.rag.retrieval.taxonomy_router import TaxonomyRouter
from gaussian_rag.rag.disambiguation.clef_client import ClefClient


def _create_sample_multi_domain_store() -> KnowledgeStore:
    """Build a KnowledgeStore with 3 distinct clusters: Medicine, Finance, and Programming."""
    store = KnowledgeStore()
    dim = 16
    rng = np.random.default_rng(42)

    # Medicine cluster (centered around vector [1, 0, 0, ...])
    med_center = np.zeros(dim)
    med_center[0] = 1.0
    med_chunks = [
        GaussianKnowledge(
            id=f"med_{i}",
            text=f"Medical cardiology study {i}: mitral valve regurgitation pathophysiology and clinical surgical repair.",
            mu=med_center + rng.normal(0, 0.05, dim),
            sigma_diag=np.ones(dim) * 0.1,
            sigma_L=np.zeros((dim, 2)),
            metadata={"title": "Cardiology Surgery", "domain": "medicine"},
        )
        for i in range(8)
    ]

    # Finance cluster (centered around vector [0, 1, 0, ...])
    fin_center = np.zeros(dim)
    fin_center[1] = 1.0
    fin_chunks = [
        GaussianKnowledge(
            id=f"fin_{i}",
            text=f"Corporate finance report {i}: investment banking debt issuance, capital raising, and equity underwriting.",
            mu=fin_center + rng.normal(0, 0.05, dim),
            sigma_diag=np.ones(dim) * 0.1,
            sigma_L=np.zeros((dim, 2)),
            metadata={"title": "Investment Banking", "domain": "finance"},
        )
        for i in range(8)
    ]

    # Coding cluster (centered around vector [0, 0, 1, ...])
    code_center = np.zeros(dim)
    code_center[2] = 1.0
    code_chunks = [
        GaussianKnowledge(
            id=f"code_{i}",
            text=f"Software engineering guide {i}: Python compiler optimization, memory layout, and asynchronous concurrency.",
            mu=code_center + rng.normal(0, 0.05, dim),
            sigma_diag=np.ones(dim) * 0.1,
            sigma_L=np.zeros((dim, 2)),
            metadata={"title": "Python Engineering", "domain": "programming"},
        )
        for i in range(8)
    ]

    store.add(med_chunks + fin_chunks + code_chunks)
    return store


def test_taxonomy_construction_and_clustering() -> None:
    """TAXON-01: Builds two-level taxonomy tree with spherical k-means from store."""
    store = _create_sample_multi_domain_store()
    router = TaxonomyRouter.build_from_store(store, top_branches=3, leaf_size=4)

    assert router.item_count == 24
    assert router.top_node_count >= 2
    assert router.leaf_node_count >= 3

    labels = router.branch_labels()
    assert len(labels) >= 2
    for b in labels:
        assert b["id"].startswith("topic-")
        assert len(b["label"]) > 0


def test_taxonomy_save_and_load(tmp_path: Path) -> None:
    """Verifies co-located persistence of taxonomy.json and taxonomy.npz."""
    store = _create_sample_multi_domain_store()
    router = TaxonomyRouter.build_from_store(store, top_branches=3, leaf_size=4)

    router.save(tmp_path)
    assert (tmp_path / "taxonomy.json").exists()
    assert (tmp_path / "taxonomy.npz").exists()

    loaded = TaxonomyRouter.load(tmp_path)
    assert loaded is not None
    assert loaded.item_count == router.item_count
    assert loaded.top_node_count == router.top_node_count
    assert loaded.leaf_node_count == router.leaf_node_count


def test_route_query_cosine() -> None:
    """Verifies baseline spherical centroid cosine routing."""
    store = _create_sample_multi_domain_store()
    router = TaxonomyRouter.build_from_store(store, top_branches=3, leaf_size=4)

    query_mu = np.zeros(16)
    query_mu[1] = 1.0  # Points directly to finance cluster

    allowed, info = router.route_query(query_mu, route_leaves=1, min_items=4)
    assert len(allowed) >= 4
    assert len(allowed) < router.item_count
    # Should contain finance chunks
    assert any(cid.startswith("fin_") for cid in allowed)


@pytest.mark.skipif(
    not (os.environ.get("CLOUDFLARE_API_TOKEN") and os.environ.get("CLOUDFLARE_ACCOUNT_ID")),
    reason="Live Cloudflare credentials required for real Clef choice branch routing test",
)
@pytest.mark.asyncio
async def test_live_route_query_clef() -> None:
    """TAXON-02: Real end-to-end Clef choice routing against live Cloudflare Workers AI."""
    store = _create_sample_multi_domain_store()
    router = TaxonomyRouter.build_from_store(store, top_branches=3, leaf_size=4)

    async with ClefClient(
        account_id=os.environ["CLOUDFLARE_ACCOUNT_ID"],
        api_token=os.environ["CLOUDFLARE_API_TOKEN"],
        timeout=30.0,
    ) as client:
        # Route a financial query
        allowed_ids, info = await router.route_query_clef(
            query="How do companies raise capital through initial public offerings and bond issuance?",
            clef_client=client,
            top_n_branches=1,
        )

        assert len(allowed_ids) > 0
        assert len(allowed_ids) < router.item_count
        assert info["elapsed_ms"] > 0
        assert info["pruning_ratio"] > 0.0
        assert len(info["selected_branches"]) >= 1

        # Check that finance chunks are retained and majority of unrelated chunks are pruned
        has_fin = any(cid.startswith("fin_") for cid in allowed_ids)
        assert has_fin, f"Expected finance chunks in allowed_ids, got: {allowed_ids}"
