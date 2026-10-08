from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from collections.abc import Sequence

from gaussian_rag.config import load_project_config, load_provider_env, load_runtime_env
from gaussian_rag.core.gaussian_embedding import GaussianEmbedder
from gaussian_rag.core.types import RetrievedChunk
from gaussian_rag.evaluation.benchmarks import demo_benchmark
from gaussian_rag.evaluation.metrics import ndcg_at_k, recall_at_k
from gaussian_rag.rag.generation.generator import (
    ProviderBackedGenerator,
    SimpleGenerator,
)
from gaussian_rag.creator.ingestion import IngestionPipeline
from gaussian_rag.core.store import KnowledgeStore
from gaussian_rag.rag.retrieval.retriever import GaussianRetriever
from gaussian_rag.rag.sense_cache import SenseCache
from gaussian_rag.rag.disambiguation.pipeline import disambiguate
from gaussian_rag.rag.disambiguation.router import route
from gaussian_rag.rag.generation.context_builder import build_multi_sense_context
from gaussian_rag.rag.generation.provider import MistralChatClient
from gaussian_rag.rag.retrieval.visualizer import generate_visualization


def build_demo_pipeline() -> GaussianRetriever:
    config = load_project_config()
    encoder_config = config["encoder"]
    embedder = GaussianEmbedder(
        dimension=encoder_config["dimension"],
        sigma_mode=encoder_config["sigma_mode"],
        rank=encoder_config["low_rank"],
        embedding_backend=encoder_config.get("embedding_backend", "local"),
        on_mismatch=encoder_config.get("on_mismatch", "error"),
    )
    pipeline = IngestionPipeline(embedder)
    store = KnowledgeStore()
    docs = [
        (
            "medical-1",
            "Mitral valve regurgitation describes backward blood flow from the left ventricle to the left atrium.",
        ),
        (
            "medical-2",
            "A systolic murmur is typically heard during the contraction phase of the heart.",
        ),
        (
            "medical-3",
            "Treatment for mitral regurgitation depends on symptoms, imaging, and comorbidities.",
        ),
        (
            "legal-1",
            "Legal precedent can be ambiguous when multiple jurisdictions interpret similar facts differently.",
        ),
        (
            "legal-2",
            "Common law systems rely heavily on the principle of stare decisis.",
        ),
        (
            "legal-3",
            "The jurisdiction of the court was questioned due to the venue of the crime.",
        ),
        (
            "tech-1",
            "Apple Inc. is an American multinational technology company headquartered in Cupertino, California.",
        ),
        ("tech-2", "The iPhone 15 features a titanium frame and the A17 Pro chip."),
        (
            "tech-3",
            "Microsoft and Apple have been competitors in the personal computer market for decades.",
        ),
        ("nature-1", "The apple is a sweet, edible fruit produced by an apple tree."),
        (
            "nature-2",
            "Apples are rich in fiber and vitamin C, making them a healthy snack.",
        ),
        ("nature-3", "The harvest of apples usually occurs in late summer or autumn."),
        (
            "finance-1",
            "A bank is a financial institution that accepts deposits from the public and creates credit.",
        ),
        (
            "finance-2",
            "The central bank raised interest rates to combat rising inflation.",
        ),
        (
            "finance-3",
            "Investment banks help companies raise capital through initial public offerings.",
        ),
        (
            "geography-1",
            "The bank of the river was eroded after the heavy spring floods.",
        ),
        (
            "geography-2",
            "Fishermen often gather along the river bank during the salmon run.",
        ),
        (
            "geography-3",
            "The stream flows gently past the grassy bank into the large lake.",
        ),
        (
            "coding-1",
            "Python is a high-level, general-purpose programming language known for readability.",
        ),
        (
            "coding-2",
            "The Python ecosystem includes powerful libraries like NumPy and PyTorch.",
        ),
        (
            "coding-3",
            "Writing clean code in Python requires following PEP 8 guidelines.",
        ),
        (
            "snakes-1",
            "The python is a genus of constricting snakes found in Asia and Africa.",
        ),
        (
            "snakes-2",
            "Pythons kill their prey by wrapping their bodies around them and squeezing.",
        ),
        (
            "snakes-3",
            "The reticulated python is one of the world's longest snake species.",
        ),
    ]
    for doc_id, text in docs:
        store.add(pipeline.ingest_document(text, document_id=doc_id))
    retriever = GaussianRetriever(store)
    retriever.refresh()
    return retriever


def _build_embedder() -> GaussianEmbedder:
    config = load_project_config()
    encoder_config = config["encoder"]
    return GaussianEmbedder(
        dimension=encoder_config["dimension"],
        sigma_mode=encoder_config["sigma_mode"],
        rank=encoder_config["low_rank"],
        embedding_backend=encoder_config.get("embedding_backend", "local"),
        on_mismatch=encoder_config.get("on_mismatch", "error"),
    )


def _metric_requires_query_covariance(metric: str) -> bool:
    return metric in {"elk", "fisher_rao"}


def _serialize_results(results: Sequence[RetrievedChunk]) -> list[dict[str, object]]:
    return [
        {
            "rank": item.rank,
            "id": item.knowledge.id,
            "confidence": item.confidence,
            "uncertainty_label": item.uncertainty_label,
            "w2_distance": item.w2_distance,
            "fr_distance": item.fr_distance,
            "elk_score": item.elk_score,
            "text": item.knowledge.text,
        }
        for item in results
    ]


def _build_retriever(
    store_path: str | None = None, *, demo: bool = False
) -> GaussianRetriever:
    if demo:
        return build_demo_pipeline()
    if store_path is None:
        raise ValueError(
            "A persisted store is required. Run ingest first or pass --store explicitly."
        )
    knowledge_file = Path(store_path) / "knowledge.json"
    if not knowledge_file.exists():
        raise ValueError(
            f"Persisted store not found at {knowledge_file}. Run ingest first or pass a valid --store path."
        )
    store = KnowledgeStore.load(store_path)
    retriever = GaussianRetriever(store)
    retriever.refresh()
    return retriever


def _run_query(
    text: str,
    top_k: int,
    metric: str,
    store_path: str | None = None,
    *,
    demo: bool = False,
) -> list[dict[str, object]]:
    retriever = _build_retriever(store_path, demo=demo)
    query_representation = _build_embedder().encode_query(
        text, with_covariance=_metric_requires_query_covariance(metric)
    )
    results = retriever.retrieve(query_representation, top_k=top_k, metric=metric)
    return _serialize_results(results)


def _run_answer(
    text: str,
    top_k: int,
    metric: str,
    store_path: str,
) -> dict[str, object]:
    retriever = _build_retriever(store_path)
    query_representation = _build_embedder().encode_query(
        text, with_covariance=_metric_requires_query_covariance(metric)
    )
    retrieved = retriever.retrieve(query_representation, top_k=top_k, metric=metric)
    generation_config = load_project_config()["generation"]
    if generation_config["provider"] != "mistral":
        raise ValueError(
            "The current answer path supports only generation.provider=mistral."
        )
    if generation_config["mode"] != "provider":
        raise ValueError(
            "The answer command requires generation.mode=provider for real provider-backed generation."
        )
    provider_env = load_provider_env()
    if provider_env["model"] == "mistral-small-latest":
        provider_env["model"] = generation_config["model"]
    generator = ProviderBackedGenerator(generation_config, provider_env)
    generated = generator.generate(text, retrieved)
    return {
        "query": text,
        "retrieved": _serialize_results(retrieved),
        "answer": generated["answer"],
        "prompt": generated["prompt"],
    }


def _run_polysemy_query(
    text: str,
    top_k: int,
    metric: str,
    store_path: str | None = None,
) -> dict[str, object]:
    retriever = _build_retriever(store_path)
    embedder = _build_embedder()

    # Init SenseCache locally for now; could be loaded from store_path in future
    cache_dir = Path(store_path) if store_path else Path(".planning")
    cache = SenseCache.load(cache_dir)

    provider_env = load_provider_env()
    client = MistralChatClient(provider_env)

    # Wrapper to enforce JSON mode if supported by the MistralChatClient,
    # or just use the default generate which follows the prompt's JSON instructions.
    def llm_caller(prompt: str) -> str:
        return client.generate(prompt)

    packet = disambiguate(
        text,
        embedder,
        cache,
        retriever=retriever,
        llm_caller=llm_caller,
        min_confidence=0.05,
    )
    cache.save(cache_dir)

    sense_results, fallback = route(
        packet, embedder, retriever, top_k=top_k, metric=metric
    )

    if len(sense_results) == 1:
        # Fall back to standard context build for single sense
        from gaussian_rag.rag.generation.context_builder import build_context

        ctx = build_context(text, next(iter(sense_results.values())))
    else:
        ctx = build_multi_sense_context(
            text,
            sense_results,
            sense_domains={sense.sense_id: sense.domain for sense in packet.senses},
            max_total_chunks=top_k,
        )

    return {
        "query": text,
        "senses_detected": len(packet.senses),
        "entropy": packet.entropy,
        "fallback_triggered": fallback,
        "selected_sense_id": packet.selected_sense_id,
        "context_prompt": ctx,
    }


def _run_ingest(input_path: str, output_path: str) -> dict[str, object]:
    embedder = _build_embedder()

    llm_caller = None
    try:
        provider_env = load_provider_env()
    except ValueError:
        provider_env = None

    if provider_env is not None:
        client = MistralChatClient(provider_env)

        def llm_caller_func(prompt: str) -> str:
            return client.generate(prompt)

        llm_caller = llm_caller_func

    pipeline = IngestionPipeline(
        embedder,
        llm_caller=llm_caller,
        extract_entities=True,
        ai_chunking=(llm_caller is not None),
    )
    store = KnowledgeStore()
    source = Path(input_path)
    files = [source] if source.is_file() else sorted(source.glob("**/*.txt"))
    for index, file_path in enumerate(files):
        text = file_path.read_text(encoding="utf-8")
        store.add(
            pipeline.ingest_document(
                text,
                document_id=file_path.stem or f"doc-{index}",
                metadata={"source_path": str(file_path)},
            )
        )
    store.save(output_path)
    return {
        "input": str(source),
        "output": output_path,
        "documents": len(files),
        "chunks": len(store.values()),
    }


def _run_evaluate(benchmark: str, split: str, top_k: int) -> dict[str, object]:
    if benchmark != "demo":
        raise ValueError(f"Unsupported benchmark: {benchmark}")
    if split != "test":
        raise ValueError(f"Unsupported split: {split}")
    retriever = build_demo_pipeline()
    embedder = _build_embedder()
    recalls: list[float] = []
    ndcgs: list[float] = []
    for sample in demo_benchmark():
        query = embedder.encode_query(sample["query"], with_covariance=True)
        results = retriever.retrieve(query, top_k=top_k)
        ids = [item.knowledge.id for item in results]
        recalls.append(recall_at_k(ids, sample["relevant_ids"], top_k))
        ndcgs.append(ndcg_at_k(ids, sample["relevant_ids"], top_k))
    return {
        "benchmark": benchmark,
        "split": split,
        f"recall@{top_k}": sum(recalls) / len(recalls),
        f"ndcg@{top_k}": sum(ndcgs) / len(ndcgs),
    }


def _run_visualize(store_path: str, output_path: str) -> dict[str, object]:
    html_path = generate_visualization(store_path, output_path)
    return {
        "status": "success",
        "visualizer_path": html_path,
        "message": f"Visualization generated at {html_path}",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run GaussianRAG production-oriented commands"
    )
    subparsers = parser.add_subparsers(dest="command")
    runtime_env = load_runtime_env()

    ingest_parser = subparsers.add_parser("ingest")
    ingest_parser.add_argument("--input", required=True)
    ingest_parser.add_argument("--output", default=runtime_env["store_path"])

    query_parser = subparsers.add_parser("query")
    query_parser.add_argument("--text", required=True)
    query_parser.add_argument("--top-k", type=int, default=runtime_env["top_k"])
    query_parser.add_argument("--store", default=runtime_env["store_path"])
    query_parser.add_argument(
        "--metric", choices=["elk", "wasserstein2", "fisher_rao"], default="elk"
    )

    answer_parser = subparsers.add_parser("answer")
    answer_parser.add_argument("--text", required=True)
    answer_parser.add_argument("--top-k", type=int, default=runtime_env["top_k"])
    answer_parser.add_argument("--store", default=runtime_env["store_path"])
    answer_parser.add_argument(
        "--metric", choices=["elk", "wasserstein2", "fisher_rao"], default="elk"
    )

    polysemy_parser = subparsers.add_parser("polysemy-query")
    polysemy_parser.add_argument("--text", required=True)
    polysemy_parser.add_argument("--top-k", type=int, default=runtime_env["top_k"])
    polysemy_parser.add_argument("--store", default=runtime_env["store_path"])
    polysemy_parser.add_argument(
        "--metric", choices=["elk", "wasserstein2", "fisher_rao"], default="elk"
    )

    demo_query_parser = subparsers.add_parser("demo-query")
    demo_query_parser.add_argument("--text", required=True)
    demo_query_parser.add_argument("--top-k", type=int, default=runtime_env["top_k"])
    demo_query_parser.add_argument(
        "--metric", choices=["elk", "wasserstein2", "fisher_rao"], default="elk"
    )

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--benchmark", default="demo")
    evaluate_parser.add_argument("--split", default="test")
    evaluate_parser.add_argument("--top-k", type=int, default=3)

    subparsers.add_parser("ablation")

    visualize_parser = subparsers.add_parser("visualize")
    visualize_parser.add_argument("--store", default=runtime_env["store_path"])
    visualize_parser.add_argument("--output", default=".visualizer")

    parser.add_argument("legacy_query", nargs="?")
    args = parser.parse_args()

    if args.command == "ingest":
        print(json.dumps(_run_ingest(args.input, args.output), indent=2))
        return
    if args.command == "query":
        print(
            json.dumps(
                _run_query(args.text, args.top_k, args.metric, args.store), indent=2
            )
        )
        return
    if args.command == "answer":
        print(
            json.dumps(
                _run_answer(args.text, args.top_k, args.metric, args.store), indent=2
            )
        )
        return
    if args.command == "polysemy-query":
        print(
            json.dumps(
                _run_polysemy_query(args.text, args.top_k, args.metric, args.store),
                indent=2,
            )
        )
        return
    if args.command == "demo-query":
        print(
            json.dumps(
                _run_query(args.text, args.top_k, args.metric, demo=True), indent=2
            )
        )
        return
    if args.command == "evaluate":
        print(
            json.dumps(_run_evaluate(args.benchmark, args.split, args.top_k), indent=2)
        )
        return
    if args.command == "ablation":
        from gaussian_rag.experiments.ablation_runs import run as run_ablation

        payload = [asdict(result) for result in run_ablation()]
        print(json.dumps(payload, indent=2))
        return
    if args.command == "visualize":
        print(json.dumps(_run_visualize(args.store, args.output), indent=2))
        return

    if args.legacy_query:
        print(
            json.dumps(
                _run_query(args.legacy_query, runtime_env["top_k"], "elk"),
                indent=2,
            )
        )
        return
    parser.print_help()


if __name__ == "__main__":
    main()
