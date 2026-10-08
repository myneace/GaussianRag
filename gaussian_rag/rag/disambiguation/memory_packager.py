"""Agent 4: Memory & RAG Context Packager.

Responsibilities:
  - Apply session-context continuity boost to senses matching active cache.
  - Produce MemoryOps (upsert list, prune list, context_window snapshot).
  - Build stable sense_ids and enriched retrieval queries with contrastive hints.
"""

from __future__ import annotations

from gaussian_rag.core.types import MemoryOps, SenseNode
from gaussian_rag.rag.sense_cache import SenseCache


def _contrastive_query(sense: SenseNode, competing: list[SenseNode]) -> str:
    """Return the base retrieval_query without string-based NOT-filtering.

    Appending 'NOT (competing_hint)' actually hurts vector embeddings because
    embedding models (and our heuristic embedder) will encode the literal tokens
    of the competing hints, ironically pulling the query closer to the competing
    domain in the latent space.

    Instead, contrastive filtering is handled strictly via search_excluding()
    in the ANN index logic.
    """
    return sense.retrieval_query


def _normalized_query_identity(text: str) -> str:
    return " ".join(text.lower().split())


def prune_ids_for_weights(
    senses: list[SenseNode], field_weights: dict[str, float], threshold: float = 0.1
) -> list[str]:
    return [
        sense.sense_id
        for sense in senses
        if field_weights.get(sense.sense_id, 0.0) < threshold
    ]


def package_memory(
    senses: list[SenseNode],
    field_weights: dict[str, float],
    cache: SenseCache,
    min_upsert_confidence: float = 0.3,
    context_boost: float = 0.15,
) -> tuple[list[SenseNode], MemoryOps]:
    """Agent 4: apply session context, build MemoryOps, enrich queries.

    Args:
        senses:                  Refined SenseNodes from Agents 2/3.
        field_weights:           Smoothed weights from Agent 3.
        cache:                   Live SenseCache instance.
        min_upsert_confidence:   Only upsert nodes above this threshold.
        context_boost:           Confidence delta for senses in active window.

    Returns:
        (enriched_senses, memory_ops)
    """
    # --- Session context boost -------------------------------------------
    active_window = cache.context_window()
    enriched: list[SenseNode] = []

    for sense in senses:
        # Check if this term already has an active sense in cache
        cached = cache.get_by_term(sense.term)
        boosted_confidence = sense.confidence
        if any(
            c.sense_id == sense.sense_id
            or (
                c.domain == sense.domain
                and c.version == sense.version
                and _normalized_query_identity(c.retrieval_query)
                == _normalized_query_identity(sense.retrieval_query)
            )
            for c in cached
        ):
            # Same sense recurs across turns → continuity boost
            boosted_confidence = min(sense.confidence + context_boost, 0.95)

        # (Contrastive query enrichment logic removed — handled by ANN logic)
        enriched.append(
            SenseNode(
                sense_id=sense.sense_id,
                term=sense.term,
                domain=sense.domain,
                confidence=boosted_confidence,
                context_hints=sense.context_hints,
                supporting_evidence=sense.supporting_evidence,
                retrieval_query=sense.retrieval_query,  # contrastive handled via ANN
                mu_anchor=sense.mu_anchor,
                sigma_anchor=sense.sigma_anchor,
                sigma_L_anchor=sense.sigma_L_anchor,
                manifold_neighbor_id=sense.manifold_neighbor_id,
                anchor_neighbor_id=sense.anchor_neighbor_id,
                ttl_hours=sense.ttl_hours,
                version=sense.version,
                created_at=sense.created_at,
            )
        )

    # --- Build MemoryOps -------------------------------------------------
    upsert = [s for s in enriched if s.confidence >= min_upsert_confidence]

    # Prune: low-weight senses that are superseded (weight < 0.1 and not top)
    prune_ids = prune_ids_for_weights(enriched, field_weights)

    ops = MemoryOps(
        upsert_nodes=upsert,
        prune_ids=prune_ids,
        context_window=active_window,
    )

    return enriched, ops
