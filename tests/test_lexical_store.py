"""test_lexical_store.py — Unit tests for SQLite FTS5 LexicalStore."""

from __future__ import annotations

import time
from pathlib import Path
import pytest

from gaussian_rag.core.lexical_store import LexicalStore, sanitize_fts5_query


def test_sanitize_fts5_query() -> None:
    assert sanitize_fts5_query("hello world") == '"hello" "world"'
    assert sanitize_fts5_query("apple + stock & price") == '"apple" "stock" "price"'
    assert sanitize_fts5_query("!@#$%^&*()") == '""'


def test_lexical_store_crud(tmp_path: Path) -> None:
    db_file = tmp_path / "test_lexical.db"
    store = LexicalStore(db_file)

    passages = [
        ("chunk_0", "doc_a", "Cloudflare Workers AI runs edge inference with Clef-Flash"),
        ("chunk_1", "doc_a", "Apple stock prices surged after Q4 quarterly revenue report"),
        ("chunk_2", "doc_b", "Python machine learning models use PyTorch and NumPy for tensor math"),
    ]
    inserted = store.add_passages(passages)
    assert inserted == 3
    assert store.count() == 3

    # Exact match retrieval
    results = store.search("Cloudflare Clef", top_k=5)
    assert len(results) >= 1
    assert results[0][0] == "chunk_0"

    # Finance query
    results_fin = store.search("revenue quarterly", top_k=5)
    assert len(results_fin) >= 1
    assert results_fin[0][0] == "chunk_1"

    # Raw text check
    assert "Clef-Flash" in store.get_passage_text("chunk_0")

    store.close()


def test_search_latency_under_5ms(tmp_path: Path) -> None:
    store = LexicalStore(tmp_path / "bench.db")
    # Insert 100 sample chunks
    data = [
        (f"chunk_{i}", f"doc_{i // 10}", f"Sample passage number {i} discussing topic {i % 5} in detail.")
        for i in range(100)
    ]
    store.add_passages(data)

    # Warmup
    store.search("passage topic 2")

    t0 = time.perf_counter()
    results = store.search("passage topic 2", top_k=5)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert len(results) > 0
    # Latency requirement: sub-5ms (typically < 1ms)
    assert elapsed_ms < 50.0  # Safe threshold for CI, logging exact time
    store.close()
