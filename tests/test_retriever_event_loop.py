"""test_retriever_event_loop.py — Verify parallel multi-sense retrieval inside running event loops.

Validates BUG-04:
  - retrieve_multi_sense_parallel detects running event loops and offloads to ThreadPoolExecutor
  - No RuntimeError is raised when invoked from inside an active asyncio event loop
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import numpy as np
import pytest

_PKG_ROOT = Path(__file__).resolve().parents[1]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from gaussian_rag.core.store import KnowledgeStore
from gaussian_rag.core.types import GaussianKnowledge, QueryRepresentation
from gaussian_rag.rag.retrieval.retriever import GaussianRetriever


def _create_synthetic_store(dim: int = 16, count: int = 10) -> KnowledgeStore:
    rng = np.random.default_rng(123)
    store = KnowledgeStore()
    nodes = [
        GaussianKnowledge(
            id=f"doc_{i}",
            text=f"Synthetic document {i} with informative text content.",
            mu=rng.normal(0, 1.0, dim),
            sigma_diag=np.ones(dim) * 0.1,
            sigma_L=np.zeros((dim, 2)),
            metadata={"doc_index": i},
        )
        for i in range(count)
    ]
    store.add(nodes)
    return store


def test_retrieve_multi_sense_parallel_synchronous():
    """Verify retrieve_multi_sense_parallel executes cleanly in standard sync execution."""
    store = _create_synthetic_store()
    retriever = GaussianRetriever(store)
    retriever.refresh()

    rng = np.random.default_rng(456)
    sense_a = QueryRepresentation(text="Query A", mu=rng.normal(0, 1.0, 16))
    sense_b = QueryRepresentation(text="Query B", mu=rng.normal(0, 1.0, 16))
    senses = [("sense_a", 0.6, sense_a), ("sense_b", 0.4, sense_b)]

    provenance, fused = retriever.retrieve_multi_sense_parallel(senses, top_k=3, adaptive=False)
    assert len(fused) == 3
    assert "sense_a" in provenance
    assert "sense_b" in provenance


@pytest.mark.asyncio
async def test_retrieve_multi_sense_parallel_inside_running_loop():
    """Verify BUG-04: invoking retrieve_multi_sense_parallel inside an active loop succeeds without RuntimeError."""
    loop = asyncio.get_running_loop()
    assert loop.is_running()

    store = _create_synthetic_store()
    retriever = GaussianRetriever(store)
    retriever.refresh()

    rng = np.random.default_rng(789)
    sense_a = QueryRepresentation(text="Async Query A", mu=rng.normal(0, 1.0, 16))
    sense_b = QueryRepresentation(text="Async Query B", mu=rng.normal(0, 1.0, 16))
    senses = [("sense_a", 0.5, sense_a), ("sense_b", 0.5, sense_b)]

    # This call previously raised RuntimeError: asyncio.run() cannot be called from a running event loop
    provenance, fused = retriever.retrieve_multi_sense_parallel(senses, top_k=4, adaptive=False)
    assert len(fused) == 4
    assert len(provenance["sense_a"]) > 0
    assert len(provenance["sense_b"]) > 0
