"""test_dual_store_sync.py — Integration tests verifying lockstep dual-store synchronization.

Covers:
- STORE-01: KnowledgeStore.save() writes both knowledge.json and lexical.db with identical IDs.
- STORE-01 (pipeline): IngestionPipeline.ingest_document() yields chunks that, after
  KnowledgeStore.save(), appear in both geometric items and lexical FTS5 index.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from gaussian_rag.core.types import GaussianKnowledge
from gaussian_rag.core.store import KnowledgeStore
from gaussian_rag.core.lexical_store import LexicalStore
from gaussian_rag.creator.ingestion import IngestionPipeline


def test_dual_store_lockstep_persistence(tmp_path: Path) -> None:
    """STORE-01: KnowledgeStore.save() writes knowledge.json and lexical.db with identical chunk IDs."""
    store_dir = tmp_path / "dual_store"
    store = KnowledgeStore()

    chunks = [
        GaussianKnowledge(
            id="chunk_alpha",
            text="Cloudflare Workers AI Clef-Flash offers non-autoregressive decision classification.",
            mu=np.ones(16),
            sigma_diag=np.ones(16) * 0.1,
            sigma_L=np.zeros((16, 2)),
            metadata={"document_id": "doc_1"},
        ),
        GaussianKnowledge(
            id="chunk_beta",
            text="Expected Likelihood Kernel accelerates manifold retrieval using low-rank Woodbury formulas.",
            mu=np.zeros(16),
            sigma_diag=np.ones(16) * 0.2,
            sigma_L=np.zeros((16, 2)),
            metadata={"document_id": "doc_2"},
        ),
    ]

    store.add(chunks)
    assert len(store.values()) == 2

    # Save to disk
    store.save(store_dir)

    # Verify both artifacts exist
    assert (store_dir / "knowledge.json").exists()
    assert (store_dir / "lexical.db").exists()

    # Verify loading populates both geometric items and lexical searcher
    loaded = KnowledgeStore.load(store_dir)
    assert len(loaded.values()) == 2
    assert loaded.lexical is not None

    # Verify lexical search returns identical chunk IDs
    results = loaded.search_lexical("Cloudflare Clef", top_k=5)
    assert len(results) > 0
    matched_id = results[0][0]
    assert matched_id == "chunk_alpha"

    # Verify geometric item exists for the matched ID
    matched_geo = loaded.get([matched_id])
    assert len(matched_geo) == 1
    assert matched_geo[0].id == "chunk_alpha"
    assert "Clef-Flash" in matched_geo[0].text


def test_ingestion_pipeline_dual_store_lockstep(tmp_path: Path) -> None:
    """STORE-01 (pipeline): Every chunk produced by IngestionPipeline appears in both stores.

    Uses _build_embedder() (which reads config.yaml, same as production) to embed two
    sample documents end-to-end and verifies lockstep synchronization between the
    geometric KnowledgeStore and the SQLite FTS5 LexicalStore after save/load.

    Requires sentence-transformers (the standard local embedding dependency).
    Skips automatically if the package is not installed.
    """
    # Skip if sentence-transformers isn't installed — consistent with the rest of
    # the pipeline test suite which calls build_demo_pipeline() / _build_embedder().
    pytest.importorskip("sentence_transformers")

    from gaussian_rag.main import _build_embedder

    store_dir = tmp_path / "pipeline_dual"

    embedder = _build_embedder()
    pipeline = IngestionPipeline(embedder=embedder)

    sample_docs = [
        ("doc_gaussian", "Gaussian embeddings model uncertainty as covariance ellipsoids in latent space."),
        ("doc_fts5", "SQLite FTS5 full-text search enables sub-millisecond BM25 ranked retrieval."),
    ]

    store = KnowledgeStore()
    for doc_id, text in sample_docs:
        chunks = pipeline.ingest_document(text, document_id=doc_id, metadata={"source": "test"})
        store.add(chunks)

    # Both documents must have produced at least one chunk each
    assert len(store.values()) >= len(sample_docs)

    # Save triggers dual-store write
    store.save(store_dir)

    assert (store_dir / "knowledge.json").exists(), "knowledge.json must be written"
    assert (store_dir / "lexical.db").exists(), "lexical.db must be written"

    # Load and cross-validate IDs
    loaded = KnowledgeStore.load(store_dir)
    assert loaded.lexical is not None, "LexicalStore must be loaded from lexical.db"

    geometric_ids = {chunk.id for chunk in loaded.values()}
    assert len(geometric_ids) > 0

    # Every geometric ID must have a corresponding passage in the lexical index
    for gid in geometric_ids:
        text_from_lexical = loaded.lexical.get_passage_text(gid)
        assert text_from_lexical is not None, (
            f"Chunk ID '{gid}' found in KnowledgeStore but missing from LexicalStore"
        )

    # Lexical search must return IDs that exist in the geometric store
    results = loaded.search_lexical("Gaussian embeddings covariance", top_k=5)
    assert len(results) > 0, "Lexical search must return at least one result"
    for candidate_id, _score in results:
        assert candidate_id in geometric_ids, (
            f"Lexical result ID '{candidate_id}' not found in KnowledgeStore"
        )
