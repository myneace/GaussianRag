from __future__ import annotations

from gaussian_rag.core.gaussian_embedding import GaussianEmbedder
from gaussian_rag.main import build_demo_pipeline

from ..rag.generation.generator import SimpleGenerator


def run(query: str) -> dict[str, str]:
    retriever = build_demo_pipeline()
    query_representation = GaussianEmbedder().encode_query(query)
    results = retriever.retrieve_with_diversity(query_representation, top_k=3)
    return SimpleGenerator().generate(query, results)
