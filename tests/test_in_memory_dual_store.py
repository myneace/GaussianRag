"""test_in_memory_dual_store.py — Verify lockstep in-memory SQLite dual-store indexing.

Validates ARCH-01:
  - KnowledgeStore() initializes with active LexicalStore(":memory:")
  - store.add() indexes chunks into SQLite FTS5 in lockstep
  - GaussianRetriever.get_hybrid_candidates() yields BM25 candidates in memory without disk save
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_PKG_ROOT = Path(__file__).resolve().parents[1]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from gaussian_rag.core.store import KnowledgeStore
from gaussian_rag.core.lexical_store import LexicalStore
from gaussian_rag.core.types import GaussianKnowledge
from gaussian_rag.rag.retrieval.retriever import GaussianRetriever


def _create_sample_chunk(cid: str, text: str, dim: int = 16) -> GaussianKnowledge:
    return GaussianKnowledge(
        id=cid,
        text=text,
        mu=np.zeros(dim),
        sigma_diag=np.ones(dim) * 0.1,
        sigma_L=np.zeros((dim, 2)),
        metadata={"document_id": "doc1"},
    )


def test_in_memory_lexical_initialization():
    """Verify ARCH-01: KnowledgeStore defaults to an active in-memory LexicalStore."""
    store = KnowledgeStore()
    assert store.lexical is not None
    assert isinstance(store.lexical, LexicalStore)
    assert store.search_lexical("anything") == []


def test_in_memory_lockstep_add_and_search():
    """Verify store.add() populates SQLite FTS5 lexical index in lockstep."""
    store = KnowledgeStore()
    c1 = _create_sample_chunk("c1", "Nicolas Fouquet was the superintendent of finances under Louis XIV.")
    c2 = _create_sample_chunk("c2", "Jean-Baptiste Colbert orchestrated the downfall of Fouquet at Vaux-le-Vicomte.")
    c3 = _create_sample_chunk("c3", "The astronomy laboratory calibrated the telescope aperture for stellar parallax.")

    store.add([c1, c2, c3])
    assert len(store.items) == 3

    # Immediate in-memory BM25 lexical search
    hits = store.search_lexical("Fouquet")
    hit_ids = [cid for cid, score in hits]
    assert "c1" in hit_ids
    assert "c2" in hit_ids
    assert "c3" not in hit_ids


def test_in_memory_retriever_hybrid_candidates():
    """Verify retriever.get_hybrid_candidates() gets BM25 hits in pure in-memory mode."""
    store = KnowledgeStore()
    c1 = _create_sample_chunk("law1", "Never outshine the master. Always make those above you feel superior.")
    c2 = _create_sample_chunk("law2", "Never put too much trust in friends, learn how to use enemies.")
    store.add([c1, c2])

    retriever = GaussianRetriever(store)
    retriever.refresh()

    query_mu = np.zeros(16)
    candidates = retriever.get_hybrid_candidates(query_mu, query_text="master", top_k=5)
    cand_ids = [c.id for c in candidates]
    assert "law1" in cand_ids


def test_in_memory_save_and_load_persistence(tmp_path: Path):
    """Verify saving to disk syncs SQLite lexical.db and load() restores both stores."""
    store = KnowledgeStore()
    c1 = _create_sample_chunk("node1", "Quantum computing leverages superposition and quantum entanglement.")
    store.add([c1])

    store.save(tmp_path)
    assert (tmp_path / "knowledge.json").exists()
    assert (tmp_path / "lexical.db").exists()

    loaded = KnowledgeStore.load(tmp_path)
    assert len(loaded.items) == 1
    assert loaded.lexical is not None
    hits = loaded.search_lexical("entanglement")
    assert len(hits) >= 1
    assert hits[0][0] == "node1"
