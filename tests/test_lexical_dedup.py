"""test_lexical_dedup.py — Verify FTS5 lexical store deduplication and upsert.

Validates ARCH-02:
  - LexicalStore.add_passages() removes duplicate entries on matching chunk_ids
  - Repeated KnowledgeStore.save() calls maintain constant row count equal to len(store.items)
  - BM25 candidate scores remain consistent across repeated saves
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_PKG_ROOT = Path(__file__).resolve().parents[1]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from gaussian_rag.core.lexical_store import LexicalStore
from gaussian_rag.core.store import KnowledgeStore
from gaussian_rag.core.types import GaussianKnowledge


def _make_chunk(cid: str, text: str, dim: int = 16) -> GaussianKnowledge:
    return GaussianKnowledge(
        id=cid,
        text=text,
        mu=np.zeros(dim),
        sigma_diag=np.ones(dim) * 0.1,
        sigma_L=np.zeros((dim, 2)),
        metadata={"document_id": "doc_test"},
    )


def test_lexical_store_upsert_dedup():
    """Verify LexicalStore.add_passages() upserts matching chunk_ids without inflating row count."""
    lex = LexicalStore(":memory:")
    passages = [
        ("p1", "doc1", "Law 1: Never outshine the master."),
        ("p2", "doc1", "Law 2: Never put too much trust in friends."),
    ]
    lex.add_passages(passages)
    assert lex.count() == 2

    # Re-insert p1 with updated wording
    updated = [("p1", "doc1", "Law 1: Always make superiors look brilliant.")]
    lex.add_passages(updated)
    assert lex.count() == 2, f"Expected 2 rows after upsert, got {lex.count()}"

    # Search confirms updated text is indexed and old text is gone
    hits_new = lex.search("brilliant")
    assert len(hits_new) == 1
    assert hits_new[0][0] == "p1"

    hits_old = lex.search("outshine")
    assert len(hits_old) == 0


def test_repeated_knowledge_store_saves_constant_count(tmp_path: Path):
    """Verify ARCH-02: Repeated calls to store.save() do not bloat SQLite FTS5 index."""
    store = KnowledgeStore()
    chunks = [
        _make_chunk(f"c_{i}", f"Document passage number {i} with key technical terminology.")
        for i in range(5)
    ]
    store.add(chunks)
    assert store.count_lexical() == 5

    # Save 4 times consecutively to the same directory
    for _ in range(4):
        store.save(tmp_path)
        assert store.count_lexical() == 5

    # Verify directly against SQLite on-disk database
    disk_lex = LexicalStore(tmp_path / "lexical.db")
    assert disk_lex.count() == 5
    disk_lex.close()


def test_bm25_ranking_consistency_across_saves(tmp_path: Path):
    """Verify BM25 ranking scores remain stable and identical after repeated saves."""
    store = KnowledgeStore()
    c1 = _make_chunk("c1", "Deep learning probabilistic models optimize Gaussian Wasserstein geometry.")
    c2 = _make_chunk("c2", "Information geometry models manifold metrics using Woodbury matrix formulas.")
    store.add([c1, c2])

    store.save(tmp_path)
    initial_scores = store.search_lexical("geometry")

    # Save 3 additional times
    for _ in range(3):
        store.save(tmp_path)

    subsequent_scores = store.search_lexical("geometry")
    assert len(initial_scores) == len(subsequent_scores)
    for (id1, s1), (id2, s2) in zip(initial_scores, subsequent_scores):
        assert id1 == id2
        assert pytest.approx(s1, rel=1e-5) == s2
