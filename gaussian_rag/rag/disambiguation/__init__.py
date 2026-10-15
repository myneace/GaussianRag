"""
gaussian_rag.disambiguation
============================
Parallel multi-agent disambiguation pipeline for polysemy-resilient RAG.

Entry point: ``disambiguate()`` in ``pipeline.py``.

Clef decision client: ``ClefClient`` in ``clef_client.py``.
"""

from gaussian_rag.rag.disambiguation.clef_client import (  # noqa: F401
    ChoiceQuestion,
    ClefClient,
    ClefClientError,
    ClefDecisionResponse,
    ClefRequestError,
    ClefRetryExhaustedError,
    NoulQuestion,
    ScoreQuestion,
)
from gaussian_rag.rag.disambiguation.clef_disambiguator import (  # noqa: F401
    ClefDisambiguator,
    disambiguate_fast,
)
