"""test_facade_query.py — Verify rag.query signature, QueryResult, and generator execution.

Validates ARCH-04:
  - GaussianRAG.create(use_generator=True) automatically instantiates a generator
  - rag.query() and rag.query_sync() produce non-empty answer text when use_generator=True
  - QueryResult supports dictionary access, property access, and str(result) == result.answer
  - rag.query_sync() and rag.query(sync=True) execute synchronously without unhandled event loop errors
  - rag.generator setter allows assigning custom generator implementations
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

_PKG_ROOT = Path(__file__).resolve().parents[1]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from gaussian_rag import GaussianRAG, QueryResult
from gaussian_rag.core.types import RetrievedChunk


@pytest.fixture
def rag_with_generator() -> GaussianRAG:
    rag = GaussianRAG.create(use_generator=True)
    rag.add_texts([
        "Law 1: Never outshine the master. Make those above you feel comfortably superior.",
        "Law 2: Never put too much trust in friends; learn how to use enemies.",
    ])
    return rag


def test_create_use_generator_initializes_generator(rag_with_generator: GaussianRAG):
    """Verify use_generator=True instantiates an active generator."""
    assert rag_with_generator.generator is not None
    assert hasattr(rag_with_generator.generator, "generate")


@pytest.mark.asyncio
async def test_async_query_returns_query_result(rag_with_generator: GaussianRAG):
    """Verify async query returns QueryResult with non-empty answer."""
    res = await rag_with_generator.query("How to handle superiors?", top_k=2)
    assert isinstance(res, QueryResult)
    assert isinstance(res, dict)
    assert res.question == "How to handle superiors?"
    assert res["question"] == "How to handle superiors?"
    assert len(res.retrieved_chunks) >= 1
    assert len(res["retrieved_chunks"]) >= 1
    assert len(res.context) > 0
    assert len(res["context"]) > 0
    assert len(res.answer) > 0
    assert len(res["answer"]) > 0
    assert str(res) == res.answer


def test_query_sync_execution(rag_with_generator: GaussianRAG):
    """Verify synchronous query_sync returns QueryResult with answer."""
    res = rag_with_generator.query_sync("Tell me about friends and enemies", top_k=2)
    assert isinstance(res, QueryResult)
    assert res.question == "Tell me about friends and enemies"
    assert len(res.answer) > 0
    assert "enemies" in res.answer.lower() or "friends" in res.answer.lower()


def test_query_sync_flag_delegation(rag_with_generator: GaussianRAG):
    """Verify calling rag.query(..., sync=True) executes synchronously."""
    res = rag_with_generator.query("How to behave?", top_k=2, sync=True)
    assert isinstance(res, QueryResult)
    assert len(res.answer) > 0


def test_custom_generator_assignment():
    """Verify custom generator assignment and execution."""
    rag = GaussianRAG.create(use_generator=False)
    rag.add_texts(["Artificial intelligence accelerates information processing."])

    class CustomMockGenerator:
        def generate(self, query: str, chunks: list[RetrievedChunk]) -> dict[str, str]:
            return {
                "context": "\n".join(c.knowledge.text for c in chunks),
                "answer": f"Mock answer for '{query}' based on {len(chunks)} chunks.",
            }

    rag.generator = CustomMockGenerator()
    res = rag.query_sync("What does AI accelerate?", top_k=1)
    assert res.answer == "Mock answer for 'What does AI accelerate?' based on 1 chunks."
