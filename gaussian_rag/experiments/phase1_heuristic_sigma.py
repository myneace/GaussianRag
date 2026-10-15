from __future__ import annotations

from gaussian_rag.config import load_project_config
from gaussian_rag.core.gaussian_embedding import GaussianEmbedder
from gaussian_rag.main import build_demo_pipeline

from ..evaluation.benchmarks import demo_benchmark
from ..evaluation.metrics import ndcg_at_k, recall_at_k


def run() -> tuple[dict[str, float], list[dict]]:
    """Run full GaussianRAG pipeline on the demo benchmark.

    Returns
    -------
    tuple[dict[str, float], list[dict]]
        (aggregate metrics dict, per-query detail rows)
    """
    retriever = build_demo_pipeline()
    # Build the embedder using the same config that build_demo_pipeline() uses,
    # so the embedding backend and dimension always match the store's item vectors.
    cfg = load_project_config().get("encoder", {})
    embedder = GaussianEmbedder(
        dimension=cfg.get("dimension", 384),
        sigma_mode=cfg.get("sigma_mode", "heuristic"),
        rank=cfg.get("low_rank", 8),
        embedding_backend=cfg.get("embedding_backend", "local"),
        on_mismatch=cfg.get("on_mismatch", "error"),
    )
    recalls: list[float] = []
    ndcgs: list[float] = []
    details: list[dict] = []
    for sample in demo_benchmark():
        q_text = str(sample["query"])
        query = embedder.encode_query(q_text, with_covariance=True)
        results = retriever.retrieve(query, top_k=3)
        ids = [item.knowledge.id for item in results]
        relevant_ids = [str(item) for item in sample["relevant_ids"]]
        r3 = recall_at_k(ids, relevant_ids, 3)
        n3 = ndcg_at_k(ids, relevant_ids, 3)
        recalls.append(r3)
        ndcgs.append(n3)
        details.append({
            "query": q_text[:50],
            "recall@3": round(r3, 4),
            "ndcg@3": round(n3, 4),
            "top_results": ids,
            "relevant": relevant_ids,
        })
    agg = {
        "recall@3": round(sum(recalls) / len(recalls), 4),
        "ndcg@3": round(sum(ndcgs) / len(ndcgs), 4),
    }
    return agg, details
