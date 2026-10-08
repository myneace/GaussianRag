"""Disambiguation Pipeline — public entry point.

Usage::

    from gaussian_rag.rag.disambiguation.pipeline import disambiguate
    from gaussian_rag.rag.generation.context_builder import (
        build_context,
        build_multi_sense_context,
    )

    packet = disambiguate(
        text="What is the latest Apple earnings report?",
        embedder=embedder,
        retriever=retriever,
        cache=cache,
    )

    sense_results, fallback = route(packet, embedder, retriever, top_k=5)

    if len(sense_results) == 1:
        sense_id, chunks = next(iter(sense_results.items()))
        ctx = build_context(text, chunks)
    else:
        ctx = build_multi_sense_context(text, sense_results)
"""

from __future__ import annotations

import uuid
from typing import Any, Callable

from gaussian_rag.core.gaussian_embedding import GaussianEmbedder
from gaussian_rag.core.types import DisambiguationPacket, MemoryOps
from gaussian_rag.rag.sense_cache import SenseCache

from .query_dissolver import dissolve_query
from .domain_grounder import ground_senses
from .field_mapper import build_field
from .memory_packager import package_memory, prune_ids_for_weights
from .sense_generator import generate_senses


def disambiguate(
    text: str,
    embedder: GaussianEmbedder,
    cache: SenseCache,
    *,
    retriever: Any | None = None,
    llm_caller: Callable[[str], str] | None = None,
    min_confidence: float = 0.2,
    smoothing_alpha: float = 0.3,
    min_upsert_confidence: float = 0.3,
    context_boost: float = 0.15,
) -> DisambiguationPacket:
    """Run the refined disambiguation pipeline with Query Dissolution.

    Args:
        text:                    Input query or passage to disambiguate.
        embedder:                Active GaussianEmbedder instance.
        cache:                   Active SenseCache (session context store).
        retriever:               Optional retriever to ground senses in manifold.
        llm_caller:              Function calling the LLM for taxonomy generation.
        min_confidence:          Prune senses below this after Agent 1.
        smoothing_alpha:         Kernel smoothing intensity for Agent 3.
        min_upsert_confidence:   Agent 4 threshold for writing to cache.
        context_boost:           Confidence boost for session-recurring senses.

    Returns:
        DisambiguationPacket ready for ``router.route()``.
    """
    # ------------------------------------------------------------------
    # Agent 0: Query Dissolver (Ambiguity Routing Refinement)
    # ------------------------------------------------------------------
    query_senses = []
    ambiguity_score = 1
    if llm_caller:
        ambiguity_score, query_senses = dissolve_query(text, llm_caller)

    # If the query-level dissolver found high ambiguity (> 3) and generated
    # interpretations, we prioritize these over term-level senses.
    if ambiguity_score > 3 and len(query_senses) >= 2:
        raw_senses = query_senses
    else:
        # ------------------------------------------------------------------
        # Agent 1: Polysemy & Sense Generator (Fallback to Term-Level)
        # ------------------------------------------------------------------
        raw_senses = generate_senses(
            text, llm_caller=llm_caller, min_confidence=min_confidence
        )

    # If still no ambiguous interpretations detected, return a trivial single-sense packet
    if not raw_senses:
        from gaussian_rag.core.types import SenseNode

        generic = SenseNode(
            sense_id="generic_" + uuid.uuid4().hex[:8],
            term=text[:32],
            domain="general",
            confidence=1.0,
            context_hints=[],
            supporting_evidence=[],
            retrieval_query=text,
        )
        ops = MemoryOps()
        return DisambiguationPacket(
            disambiguation_id=str(uuid.uuid4()),
            original_text=text,
            senses=[generic],
            field_weights={generic.sense_id: 1.0},
            entropy=0.0,
            selected_sense_id=generic.sense_id,
            memory_ops=ops,
            fallback_triggered=False,
        )

    # ------------------------------------------------------------------
    # Agent 2: Domain Context Grounder
    # ------------------------------------------------------------------
    grounded_senses = ground_senses(text, raw_senses, embedder, retriever=retriever)

    # ------------------------------------------------------------------
    # Agent 3: Cross-Context Validator & Field Mapper
    # ------------------------------------------------------------------
    field_weights, _conflict_flags, entropy, selected_sense_id = build_field(
        grounded_senses, embedder, smoothing_alpha=smoothing_alpha
    )

    # ------------------------------------------------------------------
    # Agent 4: Memory & RAG Context Packager
    # ------------------------------------------------------------------
    enriched_senses, memory_ops = package_memory(
        grounded_senses,
        field_weights,
        cache,
        min_upsert_confidence=min_upsert_confidence,
        context_boost=context_boost,
    )

    field_weights, _conflict_flags, entropy, selected_sense_id = build_field(
        enriched_senses, embedder, smoothing_alpha=smoothing_alpha
    )
    memory_ops.prune_ids = prune_ids_for_weights(enriched_senses, field_weights)

    # Apply memory ops to the live cache
    cache.upsert(memory_ops.upsert_nodes)
    cache.prune_expired()
    cache.prune_by_ids(memory_ops.prune_ids)

    return DisambiguationPacket(
        disambiguation_id=str(uuid.uuid4()),
        original_text=text,
        senses=enriched_senses,
        field_weights=field_weights,
        entropy=entropy,
        selected_sense_id=selected_sense_id,
        memory_ops=memory_ops,
        fallback_triggered=False,
    )
