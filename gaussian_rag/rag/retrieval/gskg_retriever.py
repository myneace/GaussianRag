"""GSKG Multi-Hop Retrieval implementation.

This module implements the graph-based retrieval logic defined in
docs/concepts/gaussian_knowledge_graph.md.

It uses the entities extracted during ingestion as implicit edges, allowing
retrieval to 'hop' from semantically similar nodes to related entity-nodes
across the information manifold.
"""

from __future__ import annotations
import logging
from typing import List, Set, Dict

from gaussian_rag.core.types import (
    QueryRepresentation,
    RetrievedChunk,
    GaussianKnowledge,
)
from gaussian_rag.rag.retrieval.retriever import GaussianRetriever
from gaussian_rag.core.information_geometry import wasserstein2

logger = logging.getLogger(__name__)


def retrieve_gskg_multi_hop(
    retriever: GaussianRetriever,
    query: QueryRepresentation,
    top_k: int = 5,
    max_hops: int = 1,
    expansion_factor: int = 3,
    metric: str = "wasserstein2",
    manifold_pruning_threshold: float = 2.5,
) -> list[RetrievedChunk]:
    """Perform multi-hop retrieval over the Gaussian Semantic Knowledge Graph.

    Args:
        retriever: The GaussianRetriever instance.
        query: The query representation.
        top_k: Final number of chunks to return.
        max_hops: Number of entity-based expansion hops.
        expansion_factor: How many neighbors to consider per seed node.
        metric: Distance metric for pruning and ranking.
        manifold_pruning_threshold: Max distance from query to keep a neighbor.
    """
    logger.info(f"[GSKG] Starting {max_hops}-hop retrieval for query: '{query.text}'")

    # --- Step 1: Initial Seed Retrieval ---
    seeds = retriever.retrieve(query, top_k=top_k, adaptive=True)
    if not seeds:
        return []

    all_retrieved_ids = {r.knowledge.id for r in seeds}
    results = list(seeds)

    current_hop_nodes = seeds

    for hop in range(max_hops):
        logger.info(f"[GSKG] Hop {hop + 1} expansion...")

        # --- Step 2: Extract Entities from current nodes ---
        hop_entities: Set[str] = set()
        for res in current_hop_nodes:
            entities = res.knowledge.metadata.get("entities", [])
            hop_entities.update(entities)

        if not hop_entities:
            logger.info("[GSKG] No entities found for expansion. Stopping hops.")
            break

        logger.info(
            f"[GSKG] Found {len(hop_entities)} unique entities in current frontier."
        )

        # --- Step 3: Find Neighbors (nodes sharing these entities) ---
        neighbors: List[GaussianKnowledge] = []
        # Optimization: This is a linear scan for now.
        # In production, we'd have an Entity-Node index.
        for item in retriever.store.values():
            if item.id in all_retrieved_ids:
                continue

            item_entities = item.metadata.get("entities", [])
            # Check for overlap
            if any(e in hop_entities for e in item_entities):
                neighbors.append(item)

        if not neighbors:
            logger.info("[GSKG] No new neighbors found via entity overlap.")
            break

        logger.info(f"[GSKG] Discovered {len(neighbors)} potential neighbor nodes.")

        # --- Step 4: Manifold Pruning & Scoring ---
        # We only keep neighbors that aren't too far from the original query
        # on the information manifold (to prevent semantic drift).
        new_hop_results: List[RetrievedChunk] = []
        query_gaussian = query.as_gaussian()

        for neighbor in neighbors:
            dist = wasserstein2(query_gaussian, neighbor)
            if dist <= manifold_pruning_threshold:
                chunk = RetrievedChunk(
                    knowledge=neighbor,
                    w2_distance=dist,
                    fr_distance=None,  # Only W2 for now
                    elk_score=None,
                    confidence=neighbor.confidence,
                    uncertainty_label=neighbor.uncertainty_label,
                    rank=0,  # Will be re-ranked later
                )
                new_hop_results.append(chunk)
                all_retrieved_ids.add(neighbor.id)

        if not new_hop_results:
            logger.info("[GSKG] All potential neighbors pruned by manifold threshold.")
            break

        # Sort by distance and take the best based on expansion factor
        new_hop_results.sort(key=lambda x: x.w2_distance)
        best_neighbors = new_hop_results[: len(current_hop_nodes) * expansion_factor]

        logger.info(
            f"[GSKG] Hop {hop + 1} added {len(best_neighbors)} nodes after pruning."
        )
        results.extend(best_neighbors)
        current_hop_nodes = best_neighbors

    # --- Step 5: Final Re-ranking ---
    # Sort all results by distance (Semantic Proximity)
    results.sort(key=lambda x: x.w2_distance)

    # Final slice and rank assignment
    final_results = results[:top_k]
    for i, res in enumerate(final_results, 1):
        res.rank = i

    logger.info(f"[GSKG] Retrieval complete. Returning {len(final_results)} nodes.")
    return final_results
