"""Gaussian RAG — probabilistic retrieval-augmented generation on statistical manifolds.

Public API
----------
The easiest entry point is the high-level :class:`GaussianRAG` facade::

    from gaussian_rag import GaussianRAG

    rag = GaussianRAG.create()
    rag.add_texts(["Never outshine the master."])
    results = rag.search("How to behave around powerful people?")

For lower-level control, import individual components directly:

    from gaussian_rag import (
        GaussianEmbedder,
        KnowledgeStore,
        GaussianRetriever,
        UniversalIngestor,
        GaussianKnowledge,
        QueryRepresentation,
        RetrievedChunk,
    )
"""
from __future__ import annotations

# ── Thread-safety patch for Windows OpenBLAS (must be first) ──────────────────
import functools
import threading
import numpy as np

_orig_eigh = np.linalg.eigh
_eigh_lock = threading.Lock()


@functools.wraps(_orig_eigh)
def _thread_safe_eigh(*args, **kwargs):
    with _eigh_lock:
        return _orig_eigh(*args, **kwargs)


np.linalg.eigh = _thread_safe_eigh

# ── Version ───────────────────────────────────────────────────────────────────
__version__ = "0.1.0"

# ── Core types ────────────────────────────────────────────────────────────────
from gaussian_rag.core.types import (
    GaussianKnowledge,
    QueryRepresentation,
    RetrievedChunk,
    SenseNode,
    DisambiguationPacket,
    MemoryOps,
)

# ── Core components ───────────────────────────────────────────────────────────
from gaussian_rag.core.gaussian_embedding import GaussianEmbedder
from gaussian_rag.core.store import KnowledgeStore

# ── Retrieval ─────────────────────────────────────────────────────────────────
from gaussian_rag.rag.retrieval.retriever import GaussianRetriever
from gaussian_rag.rag.retrieval.taxonomy_router import TaxonomyRouter

# ── Ingestion & Corpus Generation ─────────────────────────────────────────────
from gaussian_rag.creator.universal_ingestor import UniversalIngestor
from gaussian_rag.creator.text_normalizer import TextNormalizer
from gaussian_rag.creator.corpus_generator import CorpusGenerator, CorpusDocument
from gaussian_rag.creator.corpus_cache import CorpusCache
from gaussian_rag.creator.llm_assistant import OpenAICorpusAssistant

from gaussian_rag.rag.facade import GaussianRAG, QueryResult

# ── Optional cloud (imported lazily — don't fail if httpx not configured) ─────
def _lazy_clef():
    from gaussian_rag.rag.disambiguation.clef_client import ClefClient
    from gaussian_rag.rag.retrieval.clef_passage_gate import ClefPassageGate
    return ClefClient, ClefPassageGate


__all__ = [
    # Facade
    "GaussianRAG",
    "QueryResult",
    # Ingestion & Corpus Generation
    "UniversalIngestor",
    "TextNormalizer",
    "CorpusGenerator",
    "CorpusDocument",
    "CorpusCache",
    "OpenAICorpusAssistant",
    # Core
    "GaussianEmbedder",
    "KnowledgeStore",
    # Retrieval
    "GaussianRetriever",
    "TaxonomyRouter",
    # Types
    "GaussianKnowledge",
    "QueryRepresentation",
    "RetrievedChunk",
    "SenseNode",
    "DisambiguationPacket",
    "MemoryOps",
    # Version
    "__version__",
]
