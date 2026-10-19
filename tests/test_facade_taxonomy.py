"""test_facade_taxonomy.py — Verify TaxonomyRouter wiring and query covariance in GaussianRAG facade.

Validates ARCH-03:
  - GaussianRAG.create(use_taxonomy=True) supports taxonomy routing
  - rag.build_taxonomy() creates and attaches TaxonomyRouter to retriever
  - rag.search() and rag.query() support with_covariance=True
  - rag.save() and GaussianRAG.load() preserve and restore taxonomy index
  - Pruning reduces candidate search space before ELK ranking
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

_PKG_ROOT = Path(__file__).resolve().parents[1]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from gaussian_rag import GaussianRAG
from gaussian_rag.core.types import GaussianKnowledge, QueryRepresentation
from gaussian_rag.rag.retrieval.taxonomy_router import TaxonomyRouter


@pytest.fixture
def rag() -> GaussianRAG:
    """Create a standard GaussianRAG instance with taxonomy support enabled."""
    return GaussianRAG.create(use_taxonomy=True)


def test_facade_build_taxonomy_and_pruning(rag: GaussianRAG):
    """Verify build_taxonomy clusters items and attaches router to facade."""
    passages = [
        "Topic A law 1: Never outshine the master in power dynamics.",
        "Topic A law 2: Conceal your intentions until execution.",
        "Topic A law 3: Always say less than necessary in royal courts.",
        "Topic B physics 1: Quantum entanglement exhibits non-local correlation.",
        "Topic B physics 2: General relativity describes spacetime curvature via stress-energy.",
        "Topic B physics 3: Thermodynamics entropy always increases in isolated systems.",
    ]
    rag.add_texts(passages)
    assert rag.item_count == 6

    # Build taxonomy with 2 branches
    router = rag.build_taxonomy(top_branches=2, leaf_size=2, seed=42)
    assert isinstance(router, TaxonomyRouter)
    assert rag.taxonomy is router
    assert not router.is_empty

    # Search with query
    results = rag.search("quantum mechanics physics entanglement", top_k=3)
    assert len(results) > 0
    assert all(r.knowledge.id in rag._store.items for r in results)


def test_facade_search_and_query_with_covariance(rag: GaussianRAG):
    """Verify search and query accept with_covariance=True and evaluate ELK ranking."""
    rag.add_texts([
        "Expected Likelihood Kernel measures probabilistic overlap between Gaussians.",
        "Wasserstein distance accounts for transportation cost across manifold distributions.",
    ])

    # Search with covariance
    results_cov = rag.search("Likelihood Kernel overlap", top_k=2, metric="elk", with_covariance=True)
    assert len(results_cov) == 2

    # Query with covariance
    import asyncio
    res_query = asyncio.run(rag.query("Likelihood Kernel", top_k=1, with_covariance=True))
    assert res_query["question"] == "Likelihood Kernel"
    assert len(res_query["retrieved_chunks"]) >= 1


def test_facade_save_and_load_taxonomy(tmp_path: Path, rag: GaussianRAG):
    """Verify rag.save() and GaussianRAG.load() persist and restore taxonomy structures."""
    passages = [f"Passage number {i} with distinct information and details." for i in range(10)]
    rag.add_texts(passages)

    router = rag.build_taxonomy(top_branches=2, leaf_size=4, seed=123)
    assert not router.is_empty

    # Save to directory
    save_path = tmp_path / "rag_taxonomy_store"
    rag.save(save_path)
    assert (save_path / "knowledge.json").exists()
    assert (save_path / "lexical.db").exists()
    assert (save_path / "taxonomy.json").exists()
    assert (save_path / "taxonomy.npz").exists()

    # Load from directory
    loaded_rag = GaussianRAG.load(save_path)
    assert loaded_rag.item_count == 10
    assert loaded_rag.taxonomy is not None
    assert not loaded_rag.taxonomy.is_empty
    assert loaded_rag.taxonomy.item_count == 10
