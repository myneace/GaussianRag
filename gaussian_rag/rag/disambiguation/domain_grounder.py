"""Agent 2: Domain Context Grounder.

Refines the raw SenseNodes from Agent 1 by:
  - Using GaussianEmbedder.encode_sense() to anchor each sense in manifold space.
  - Adjusting confidence using Wasserstein-2 distance between the sense embedding
    and a full-text query embedding (senses closer to the query get a boost).
  - Renormalising confidences per term so they sum ≤ 1.0.

The ``ground_senses`` function is the Agent 2 entry point.
"""

from __future__ import annotations

from typing import Any

from gaussian_rag.core.gaussian_embedding import GaussianEmbedder
from gaussian_rag.core.information_geometry import wasserstein2
from gaussian_rag.core.types import GaussianKnowledge, SenseNode


def _proximity_boost(
    query_gaussian: GaussianKnowledge,
    sense_gaussian: GaussianKnowledge,
    scale: float = 0.2,
) -> float:
    """Return a small positive boost for senses close to the query in manifold.

    Senses whose embedding is far from the query in Wasserstein distance get a
    smaller boost (or none), keeping the prior confidence largely intact when
    the query provides little discriminating signal.
    """
    distance = wasserstein2(query_gaussian, sense_gaussian)
    # Invert: small distance → large boost (up to `scale`)
    return scale / (1.0 + distance)


def ground_senses(
    text: str,
    senses: list[SenseNode],
    embedder: GaussianEmbedder,
    proximity_scale: float = 0.2,
    retriever: Any = None,
) -> list[SenseNode]:
    """Agent 2: attach Gaussian anchors and refine confidence scores.

    If a retriever is provided, it searches the manifold for the closest
    interpretation node to 'ground' the query sense in the existing knowledge.
    """
    if not senses:
        return []

    # Encode the raw query as a Gaussian reference point
    query_repr = embedder.encode_query(text, with_covariance=True)
    query_gaussian = query_repr.as_gaussian()

    # Encode each sense and compute proximity boost
    sense_gaussians: dict[str, GaussianKnowledge] = {}
    boosted: dict[str, float] = {}

    for sense in senses:
        g = embedder.encode_sense(sense)
        sense_gaussians[sense.sense_id] = g
        boost = _proximity_boost(query_gaussian, g, scale=proximity_scale)
        boosted[sense.sense_id] = min(sense.confidence + boost, 0.99)

    # Renormalise per term so confidences sum ≤ 1.0
    from collections import defaultdict

    term_groups: dict[str, list[SenseNode]] = defaultdict(list)
    for sense in senses:
        term_groups[sense.term].append(sense)

    refined: list[SenseNode] = []
    for group in term_groups.values():
        total = sum(boosted[s.sense_id] for s in group) or 1.0
        for sense in group:
            norm_confidence = min(boosted[sense.sense_id] / total, 0.99)
            g = sense_gaussians[sense.sense_id]

            # Relation to Manifold
            neighbor_id = None
            anchor_id = None
            if retriever is not None:
                # Search for closest interpretation node
                q_obj = embedder.encode_query(sense.retrieval_query)
                matches = retriever.retrieve(q_obj, top_k=1)
                if matches:
                    top = matches[0].knowledge
                    ntype = top.metadata.get("node_type")
                    if ntype == "interpretation":
                        neighbor_id = top.id
                        anchor_id = top.metadata.get("anchor_id")
                    elif ntype == "anchor":
                        anchor_id = top.id

            refined.append(
                SenseNode(
                    sense_id=sense.sense_id,
                    term=sense.term,
                    domain=sense.domain,
                    confidence=norm_confidence,
                    context_hints=sense.context_hints,
                    supporting_evidence=sense.supporting_evidence,
                    retrieval_query=sense.retrieval_query,
                    mu_anchor=g.mu,
                    sigma_anchor=g.sigma_diag,
                    sigma_L_anchor=g.sigma_L,
                    manifold_neighbor_id=neighbor_id,
                    anchor_neighbor_id=anchor_id,
                    ttl_hours=sense.ttl_hours,
                    version=sense.version,
                    created_at=sense.created_at,
                )
            )

    return refined
