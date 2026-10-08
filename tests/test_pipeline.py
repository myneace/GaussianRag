from pathlib import Path
import subprocess
from types import SimpleNamespace
from typing import cast

import numpy as np
from _pytest.monkeypatch import MonkeyPatch

from gaussian_rag.config import load_project_config, load_provider_env
from gaussian_rag.core.gaussian_embedding import GaussianEmbedder
from gaussian_rag.core.types import (
    DisambiguationPacket,
    GaussianKnowledge,
    MemoryOps,
    QueryRepresentation,
    SenseNode,
    RetrievedChunk,
)
from gaussian_rag.experiments.phase1_heuristic_sigma import run as run_phase1
from gaussian_rag.experiments.phase2_learned_sigma import run as run_phase2
from gaussian_rag.experiments.phase3_geometry import run as run_phase3
from gaussian_rag.experiments.phase4_llm import run as run_phase4
from gaussian_rag.experiments.ablation_runs import run as run_ablation
from gaussian_rag.main import (
    _run_answer,
    _run_ingest,
    _run_query,
    _build_embedder,
    build_demo_pipeline,
)
from gaussian_rag.rag.disambiguation.router import route
from gaussian_rag.rag.generation.provider import MistralChatClient
from gaussian_rag.rag.sense_cache import SenseCache
from gaussian_rag.rag.retrieval.retriever import GaussianRetriever


def test_demo_pipeline_retrieves_results() -> None:
    retriever = build_demo_pipeline()
    assert retriever.store.values()


def test_phase1_metrics_are_bounded() -> None:
    results, _ = run_phase1()
    assert 0.0 <= results["recall@3"] <= 1.0
    assert 0.0 <= results["ndcg@3"] <= 1.0
    # At least one relevant doc must be retrieved in a non-trivial demo pipeline
    assert results["recall@3"] > 0.0, (
        "recall@3 is exactly 0 — demo pipeline returns no relevant results"
    )


def test_phase4_generation_builds_answer() -> None:
    payload = run_phase4("ambiguous treatment guidance")
    assert "Evidence summary" in payload["answer"]


def test_phase2_training_score_is_non_negative() -> None:
    score = run_phase2("uncertain scientific evidence with conflicting studies")
    assert score >= 0.0
    # score is average Wasserstein distance between augmented views; for a
    # non-trivial embedding two distinct views must have *some* separation
    assert score > 0.0, (
        "augmentation_consistency_loss is exactly 0 — augmented views are identical"
    )


def test_phase3_geometry_score_is_non_negative() -> None:
    score = run_phase3()
    assert score >= 0.0
    # coverage_score is mean pairwise Wasserstein distance; a multi-chunk store
    # must have strictly positive diversity
    assert score > 0.0, (
        "coverage_score is exactly 0 — store has fewer than 2 chunks or all chunks are identical"
    )


def test_ablation_runner_returns_named_results() -> None:
    results = run_ablation()
    assert results
    assert all(result.name for result in results)


def test_project_config_loads_expected_defaults() -> None:
    config = load_project_config()
    assert config["encoder"]["dimension"] == 384
    assert config["retrieval"]["metric"] == "elk"


def test_sense_cache_round_trip_preserves_anchors(tmp_path: Path) -> None:
    cache = SenseCache()
    cache.upsert(
        [
            SenseNode(
                sense_id="sense-1",
                term="apple",
                domain="technology/corporate",
                confidence=0.8,
                context_hints=["iphone"],
                supporting_evidence=["earnings"],
                retrieval_query="apple earnings iphone",
                mu_anchor=np.array([1.0, 2.0, 3.0]),
                sigma_anchor=np.array([0.1, 0.2, 0.3]),
                sigma_L_anchor=np.array([[0.01], [0.02], [0.03]]),
                ttl_hours=24,
                version=2,
            )
        ]
    )

    cache.save(tmp_path)
    loaded = SenseCache.load(tmp_path)

    nodes = loaded.get_by_term("apple")
    assert len(nodes) == 1
    assert nodes[0].mu_anchor is not None
    assert nodes[0].sigma_anchor is not None
    assert nodes[0].sigma_L_anchor is not None
    assert np.allclose(nodes[0].mu_anchor, np.array([1.0, 2.0, 3.0]))
    assert np.allclose(nodes[0].sigma_anchor, np.array([0.1, 0.2, 0.3]))
    assert np.allclose(nodes[0].sigma_L_anchor, np.array([[0.01], [0.02], [0.03]]))


def test_dual_sense_route_preserves_per_sense_results() -> None:
    """Router dual-sense path must return per-sense provenance using the real retriever."""
    retriever = build_demo_pipeline()
    embedder = _build_embedder()

    # Build two real sense nodes pointing at the demo store's actual embedding space.
    # sense-a → a medical/clinical query; sense-b → a legal query.
    # Both mu_anchors come from real encode_query calls so they sit on the manifold.
    sense_a_qr = embedder.encode_query("mitral valve regurgitation treatment")
    sense_b_qr = embedder.encode_query("legal precedent jurisdiction")

    def _sense_node(sense_id: str, term: str, domain: str, qr) -> SenseNode:
        return SenseNode(
            sense_id=sense_id,
            term=term,
            domain=domain,
            confidence=0.5,
            context_hints=[],
            supporting_evidence=[],
            retrieval_query=qr.text if hasattr(qr, "text") else term,
            mu_anchor=qr.mu,
            sigma_anchor=qr.sigma_diag,
            sigma_L_anchor=qr.sigma_L,
        )

    packet = DisambiguationPacket(
        disambiguation_id="d1",
        original_text="Apple trial",
        senses=[
            _sense_node("sense-a", "mitral", "medicine", sense_a_qr),
            _sense_node("sense-b", "legal", "law", sense_b_qr),
        ],
        field_weights={"sense-a": 0.55, "sense-b": 0.45},
        entropy=2.0,  # triggers DUAL-SENSE path (1.5 < 2.0 ≤ 2.5)
        selected_sense_id=None,
        memory_ops=MemoryOps(),
    )

    sense_results, fallback = route(
        packet,
        embedder,
        retriever,
        top_k=2,
    )

    # The router must NOT trigger fallback for this entropy level
    assert fallback is False
    # Both senses must have provenance entries
    assert "sense-a" in sense_results
    assert "sense-b" in sense_results
    # Each sense must have retrieved at least one real chunk
    assert len(sense_results["sense-a"]) >= 1
    assert len(sense_results["sense-b"]) >= 1
    assert all(item.elk_score is not None for item in sense_results["sense-a"])
    assert all(item.elk_score is not None for item in sense_results["sense-b"])


def test_retrieve_multi_sense_enforces_total_budget_and_deduplicates() -> None:
    """retrieve_multi_sense must respect top_k budget using the real retriever."""
    retriever = build_demo_pipeline()
    embedder = _build_embedder()

    # Build two query representations from the real embedding space
    qr_a = embedder.encode_query("mitral valve regurgitation blood flow")
    qr_b = embedder.encode_query("conflicting scientific studies effect size")

    sense_results, fused = retriever.retrieve_multi_sense(
        [
            ("sense-a", 0.6, qr_a),
            ("sense-b", 0.4, qr_b),
        ],
        top_k=1,
        adaptive=False,
        fusion="interleave",
    )

    # top_k=1 non-adaptive: fused list must be capped at 1
    assert len(fused) == 1, f"Expected 1 fused result, got {len(fused)}"
    # Both senses must still have provenance entries
    assert "sense-a" in sense_results
    assert "sense-b" in sense_results


def test_query_dissolver_ids_are_stable_for_repeat_outputs() -> None:
    from gaussian_rag.rag.disambiguation.query_dissolver import dissolve_query

    payload = {
        "ambiguity_score": 5,
        "reasoning": "ambiguous",
        "interpretations": [
            {
                "domain": "technology/corporate",
                "text": "apple earnings and legal proceedings",
                "confidence": 0.7,
            },
            {
                "domain": "agriculture/botany",
                "text": "apple cultivar and orchard disease",
                "confidence": 0.3,
            },
        ],
    }

    def llm_caller(_prompt: str) -> str:
        return __import__("json").dumps(payload)

    first_score, first_nodes = dissolve_query("Apple trial", llm_caller)
    second_score, second_nodes = dissolve_query("Apple trial", llm_caller)

    assert first_score == second_score == 5
    assert [node.sense_id for node in first_nodes] == [
        node.sense_id for node in second_nodes
    ]


def test_memory_packager_boosts_matching_rewrite_identity_when_query_ids_are_stable() -> (
    None
):
    from gaussian_rag.rag.disambiguation.memory_packager import package_memory

    cache = SenseCache()
    cache.upsert(
        [
            SenseNode(
                sense_id="query_dissolve_cached",
                term="Apple trial",
                domain="technology/corporate",
                confidence=0.4,
                context_hints=[],
                supporting_evidence=[],
                retrieval_query="apple earnings and legal proceedings",
            )
        ]
    )

    senses = [
        SenseNode(
            sense_id="query_dissolve_new",
            term="Apple trial",
            domain="technology/corporate",
            confidence=0.5,
            context_hints=[],
            supporting_evidence=[],
            retrieval_query="apple earnings and legal proceedings",
        )
    ]
    enriched, _ops = package_memory(senses, {"query_dissolve_new": 1.0}, cache)

    assert enriched[0].confidence > 0.5


def test_memory_packager_does_not_boost_different_rewrite_same_domain() -> None:
    from gaussian_rag.rag.disambiguation.memory_packager import package_memory

    cache = SenseCache()
    cache.upsert(
        [
            SenseNode(
                sense_id="query_dissolve_cached",
                term="Apple trial",
                domain="technology/corporate",
                confidence=0.4,
                context_hints=[],
                supporting_evidence=[],
                retrieval_query="apple earnings and legal proceedings",
            )
        ]
    )

    senses = [
        SenseNode(
            sense_id="query_dissolve_new",
            term="Apple trial",
            domain="technology/corporate",
            confidence=0.5,
            context_hints=[],
            supporting_evidence=[],
            retrieval_query="app store antitrust case",
        )
    ]
    enriched, _ops = package_memory(senses, {"query_dissolve_new": 1.0}, cache)

    assert enriched[0].confidence == 0.5


def test_disambiguate_recomputes_field_after_memory_boost(
    monkeypatch: MonkeyPatch,
) -> None:
    from gaussian_rag.rag.disambiguation import pipeline as pipeline_module

    sense_a = SenseNode(
        sense_id="sense-a",
        term="apple",
        domain="technology/corporate",
        confidence=0.51,
        context_hints=[],
        supporting_evidence=[],
        retrieval_query="apple earnings",
        mu_anchor=np.array([1.0, 0.0]),
        sigma_anchor=np.array([0.1, 0.1]),
    )
    sense_b = SenseNode(
        sense_id="sense-b",
        term="apple",
        domain="agriculture/botany",
        confidence=0.4,
        context_hints=[],
        supporting_evidence=[],
        retrieval_query="apple orchard",
        mu_anchor=np.array([0.0, 1.0]),
        sigma_anchor=np.array([0.1, 0.1]),
    )

    monkeypatch.setattr(
        pipeline_module, "generate_senses", lambda *args, **kwargs: [sense_a, sense_b]
    )
    monkeypatch.setattr(
        pipeline_module, "ground_senses", lambda *args, **kwargs: [sense_a, sense_b]
    )

    cache = SenseCache()
    cache.upsert(
        [
            SenseNode(
                sense_id="sense-b",
                term="apple",
                domain="agriculture/botany",
                confidence=0.4,
                context_hints=[],
                supporting_evidence=[],
                retrieval_query="apple orchard",
            )
        ]
    )

    packet = pipeline_module.disambiguate(
        "Apple trial",
        cast(GaussianEmbedder, object()),
        cache,
        llm_caller=lambda _prompt: "{}",
    )

    assert packet.selected_sense_id == "sense-b"
    assert packet.field_weights["sense-b"] > packet.field_weights["sense-a"]


def test_encode_sense_scales_low_rank_covariance_consistently(
    monkeypatch: MonkeyPatch,
) -> None:
    base = GaussianKnowledge(
        id="sense-a",
        text="apple earnings",
        mu=np.array([0.0, 0.0]),
        sigma_diag=np.array([4.0, 2.0]),
        sigma_L=np.array([[2.0], [1.0]]),
        metadata={},
    )

    monkeypatch.setattr(GaussianEmbedder, "encode", lambda self, *args, **kwargs: base)
    embedder = object.__new__(GaussianEmbedder)
    sense = SenseNode(
        sense_id="sense-a",
        term="apple",
        domain="technology/corporate",
        confidence=1.0,
        context_hints=[],
        supporting_evidence=[],
        retrieval_query="apple earnings",
    )

    scaled = GaussianEmbedder.encode_sense(embedder, sense)

    assert np.allclose(scaled.sigma_diag, base.sigma_diag * 0.5)
    assert np.allclose(scaled.sigma, base.sigma * 0.5)


def test_build_multi_sense_context_uses_sense_domains_and_global_budget() -> None:
    from gaussian_rag.rag.generation.context_builder import build_multi_sense_context

    shared = GaussianKnowledge(
        id="shared",
        text="shared evidence",
        mu=np.array([0.0, 0.0]),
        sigma_diag=np.array([0.1, 0.1]),
        sigma_L=np.zeros((2, 0)),
        metadata={"domain": "shared"},
    )
    chunk = RetrievedChunk(
        knowledge=shared,
        w2_distance=0.1,
        fr_distance=None,
        elk_score=None,
        confidence=shared.confidence,
        uncertainty_label=shared.uncertainty_label,
        rank=1,
    )

    ctx = build_multi_sense_context(
        "Apple trial",
        {"sense-a": [chunk], "sense-b": [chunk]},
        sense_domains={
            "sense-a": "technology/corporate",
            "sense-b": "agriculture/botany",
        },
        max_total_chunks=1,
    )

    assert "[Sense A: technology/corporate]" in ctx
    assert "[Sense B: agriculture/botany]" in ctx
    assert ctx.count("[Chunk ") == 1


def test_provider_env_loads_mistral_keys_from_file(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                'MISTRAL_API_KEY="file-key"',
                'MISTRAL_BASE_URL="https://api.mistral.ai/v1/chat/completions"',
                'MISTRAL_MODEL="mistral-small-latest"',
            ]
        ),
        encoding="utf-8",
    )
    provider_env = load_provider_env(env_file)
    assert provider_env["api_key"] == "file-key"


def test_provider_client_parses_mistral_response(monkeypatch: MonkeyPatch) -> None:
    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {
                "choices": [
                    {
                        "message": {
                            "content": "provider output",
                        }
                    }
                ]
            }

    def fake_post(*args: object, **kwargs: object) -> FakeResponse:
        return FakeResponse()

    monkeypatch.setattr("httpx.post", fake_post)
    client = MistralChatClient(
        {
            "api_key": "test-key",
            "base_url": "https://api.mistral.ai/v1/chat/completions",
            "model": "mistral-small-latest",
        }
    )
    assert client.generate("prompt") == "provider output"


def test_fisher_rao_metric_populates_distance_field() -> None:
    retriever = build_demo_pipeline()
    results = GaussianRetriever(retriever.store)
    results.refresh()
    ranked = results.retrieve(
        __import__(
            "gaussian_rag.core.gaussian_embedding", fromlist=["GaussianEmbedder"]
        )
        .GaussianEmbedder()
        .encode_query("jurisdictional ambiguity", with_covariance=True),
        top_k=3,
        metric="fisher_rao",
    )
    assert ranked
    assert all(item.fr_distance is not None for item in ranked)


def test_elk_metric_populates_overlap_scores() -> None:
    store = KnowledgeStore()
    store.add(
        [
            GaussianKnowledge(
                id="near",
                text="near",
                mu=np.array([0.0, 0.0]),
                sigma_diag=np.array([0.1, 0.1]),
                sigma_L=np.zeros((2, 0)),
            ),
            GaussianKnowledge(
                id="far",
                text="far",
                mu=np.array([5.0, 5.0]),
                sigma_diag=np.array([0.1, 0.1]),
                sigma_L=np.zeros((2, 0)),
            ),
        ]
    )
    retriever = GaussianRetriever(store)
    retriever.refresh()
    ranked = retriever.retrieve(
        QueryRepresentation(
            text="query",
            mu=np.array([0.0, 0.0]),
            sigma_diag=np.array([0.1, 0.1]),
            sigma_L=np.zeros((2, 0)),
        ),
        top_k=2,
        metric="elk",
        adaptive=False,
    )

    assert ranked
    assert all(item.elk_score is not None for item in ranked)
    assert ranked[0].knowledge.id == "near"


def test_ingest_and_query_store_round_trip(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "alpha.txt").write_text(
        "Precedent ambiguity depends on jurisdiction and venue.", encoding="utf-8"
    )
    store_dir = tmp_path / "store"
    summary = _run_ingest(str(docs_dir), str(store_dir))
    assert summary["documents"] == 1
    results = _run_query("jurisdictional ambiguity", 2, "wasserstein2", str(store_dir))
    assert results


def test_query_requires_persisted_store() -> None:
    try:
        _run_query("jurisdictional ambiguity", 2, "wasserstein2")
    except ValueError as exc:
        assert "persisted store is required" in str(exc)
    else:
        raise AssertionError("query without a store should fail")


def test_query_missing_store_has_clear_error(tmp_path: Path) -> None:
    missing_store = tmp_path / "missing-store"
    try:
        _run_query("jurisdictional ambiguity", 2, "wasserstein2", str(missing_store))
    except ValueError as exc:
        assert "Run ingest first" in str(exc)
    else:
        raise AssertionError("query with a missing store should fail clearly")


def test_answer_uses_provider_backed_generator(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "alpha.txt").write_text(
        "Legal ambiguity depends on jurisdiction and venue.", encoding="utf-8"
    )
    store_dir = tmp_path / "store"
    _run_ingest(str(docs_dir), str(store_dir))

    from gaussian_rag.rag.generation.generator import ProviderBackedGenerator

    def fake_generate(
        self: ProviderBackedGenerator, query: str, retrieved: list[object]
    ) -> dict[str, str]:
        return {"context": "ctx", "prompt": "prompt", "answer": "provider answer"}

    monkeypatch.setattr(ProviderBackedGenerator, "generate", fake_generate)
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    payload = _run_answer("jurisdictional ambiguity", 2, "wasserstein2", str(store_dir))
    assert payload["answer"] == "provider answer"


def test_answer_runs_through_provider_client_contract(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "alpha.txt").write_text(
        "Legal ambiguity depends on jurisdiction and venue.", encoding="utf-8"
    )
    store_dir = tmp_path / "store"
    _run_ingest(str(docs_dir), str(store_dir))

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {
                "choices": [
                    {
                        "message": {
                            "content": "live provider answer",
                        }
                    }
                ]
            }

    def fake_post(*args: object, **kwargs: object) -> FakeResponse:
        return FakeResponse()

    monkeypatch.setattr("httpx.post", fake_post)
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    payload = _run_answer("jurisdictional ambiguity", 2, "wasserstein2", str(store_dir))
    assert payload["answer"] == "live provider answer"


def test_answer_requires_provider_mode(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "alpha.txt").write_text(
        "Legal ambiguity depends on jurisdiction and venue.", encoding="utf-8"
    )
    store_dir = tmp_path / "store"
    _run_ingest(str(docs_dir), str(store_dir))

    monkeypatch.setattr(
        "gaussian_rag.main.load_project_config",
        lambda: {
            "encoder": {
                "backend": "hashing-demo",
                "dimension": 128,
                "sigma_mode": "heuristic",
                "low_rank": 8,
            },
            "chunking": {"max_tokens": 128, "overlap_tokens": 16},
            "retrieval": {
                "ann_candidates": 10,
                "top_k": 5,
                "diversity_threshold": 0.3,
                "metric": "wasserstein2",
            },
            "generation": {
                "mode": "template-demo",
                "provider": "mistral",
                "model": "mistral-small-latest",
                "timeout_seconds": 30.0,
            },
            "evaluation": {
                "ambiguity_buckets": {"medium_entropy": 2.5, "high_entropy": 4.0}
            },
        },
    )
    try:
        _run_answer("jurisdictional ambiguity", 2, "wasserstein2", str(store_dir))
    except ValueError as exc:
        assert "generation.mode=provider" in str(exc)
    else:
        raise AssertionError("answer should fail when generation.mode is not provider")


def test_ablation_cli_runs() -> None:
    result = subprocess.run(
        [
            "uv",
            "run",
            "--no-sync",
            "python",
            "-m",
            "gaussian_rag.main",
            "ablation",
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    assert "heuristic_sigma" in result.stdout


def test_demo_query_cli_runs() -> None:
    result = subprocess.run(
        [
            "uv",
            "run",
            "--no-sync",
            "python",
            "-m",
            "gaussian_rag.main",
            "demo-query",
            "--text",
            "ambiguous treatment guidance",
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    assert "doc-1-chunk-0" in result.stdout


def test_ingest_and_query_cli_flow_runs(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "alpha.txt").write_text(
        "Legal ambiguity depends on jurisdiction and venue.", encoding="utf-8"
    )
    store_dir = tmp_path / "store"
    repo_root = Path(__file__).resolve().parents[1]

    ingest = subprocess.run(
        [
            "uv",
            "run",
            "--no-sync",
            "python",
            "-m",
            "gaussian_rag.main",
            "ingest",
            "--input",
            str(docs_dir),
            "--output",
            str(store_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=repo_root,
    )
    assert '"documents": 1' in ingest.stdout

    query = subprocess.run(
        [
            "uv",
            "run",
            "--no-sync",
            "python",
            "-m",
            "gaussian_rag.main",
            "query",
            "--store",
            str(store_dir),
            "--text",
            "jurisdictional ambiguity",
            "--top-k",
            "2",
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=repo_root,
    )
    assert "alpha-chunk-0" in query.stdout


def test_evaluate_cli_runs() -> None:
    result = subprocess.run(
        [
            "uv",
            "run",
            "--no-sync",
            "python",
            "-m",
            "gaussian_rag.main",
            "evaluate",
            "--benchmark",
            "demo",
            "--split",
            "test",
            "--top-k",
            "3",
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    assert '"benchmark": "demo"' in result.stdout


def test_answer_cli_uses_env_and_store(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "alpha.txt").write_text(
        "Legal ambiguity depends on jurisdiction and venue.", encoding="utf-8"
    )
    store_dir = tmp_path / "store"
    repo_root = Path(__file__).resolve().parents[1]

    subprocess.run(
        [
            "uv",
            "run",
            "--no-sync",
            "python",
            "-m",
            "gaussian_rag.main",
            "ingest",
            "--input",
            str(docs_dir),
            "--output",
            str(store_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=repo_root,
    )

    # Write env file and provider stub inside try/finally so any failure
    # (including unexpected exceptions before the subprocess call) still
    # cleans up the real repo .env — preventing test pollution.
    env_file = repo_root / ".env"
    provider_stub = repo_root / "tests" / "provider_stub.py"
    try:
        env_file.write_text(
            "\n".join(
                [
                    'MISTRAL_API_KEY="cli-test-key"',
                    'MISTRAL_BASE_URL="https://api.mistral.ai/v1/chat/completions"',
                    'MISTRAL_MODEL="mistral-small-latest"',
                    f'GAUSSIAN_RAG_STORE_PATH="{store_dir}"',
                    'GAUSSIAN_RAG_TOP_K="2"',
                ]
            ),
            encoding="utf-8",
        )
        provider_stub.write_text(
            "import json\n"
            "import httpx\n"
            "class _FakeResponse:\n"
            "    def raise_for_status(self):\n"
            "        return None\n"
            "    def json(self):\n"
            "        return {'choices': [{'message': {'content': 'cli provider answer'}}]}\n"
            "httpx.post = lambda *args, **kwargs: _FakeResponse()\n",
            encoding="utf-8",
        )
        result = subprocess.run(
            [
                "uv",
                "run",
                "--no-sync",
                "python",
                "-c",
                "import tests.provider_stub; import gaussian_rag.main as m; import sys; sys.argv=['gaussian_rag.main','answer','--text','jurisdictional ambiguity']; m.main()",
            ],
            check=True,
            capture_output=True,
            text=True,
            cwd=repo_root,
        )
        assert "cli provider answer" in result.stdout
    finally:
        if provider_stub.exists():
            provider_stub.unlink()
        if env_file.exists():
            env_file.unlink()


# ---------------------------------------------------------------------------
# Task 6 — coverage for training helpers that previously had no tests
# ---------------------------------------------------------------------------


def test_sigma_head_predict_returns_valid_diag_and_low_rank() -> None:
    from gaussian_rag.training.sigma_head import SigmaHead

    dim, rank = 16, 4
    head = SigmaHead(dimension=dim, rank=rank)
    hidden = np.random.default_rng(0).standard_normal(dim)
    diag, low_rank = head.predict(hidden)

    # Shape contracts
    assert diag.shape == (dim,), f"diag shape {diag.shape} != ({dim},)"
    assert low_rank.shape == (dim, rank), (
        f"low_rank shape {low_rank.shape} != ({dim}, {rank})"
    )
    # diag must be strictly positive (floor clipped)
    assert float(np.min(diag)) > 0.0, "diag contains non-positive values"
    # diag must be unit-normed by construction
    assert abs(float(np.linalg.norm(diag)) - 1.0) < 1e-6, "diag is not unit-normed"


def test_contrastive_w2_loss_pushes_positive_closer_than_margin() -> None:
    from gaussian_rag.training.losses import contrastive_w2_loss
    from gaussian_rag.core.types import GaussianKnowledge

    dim = 8

    def _make(mu_offset: float) -> GaussianKnowledge:
        mu = np.zeros(dim)
        mu[0] = mu_offset
        sigma_diag = np.ones(dim) * 0.1
        sigma_L = np.zeros((dim, 2))
        return GaussianKnowledge(
            id=f"g{mu_offset}",
            text="",
            mu=mu,
            sigma_diag=sigma_diag,
            sigma_L=sigma_L,
        )

    anchor = _make(0.0)
    positive = _make(0.1)  # close
    negative = _make(5.0)  # far

    loss_with_neg = contrastive_w2_loss(anchor, positive, [negative], margin=0.5)
    loss_no_neg = contrastive_w2_loss(anchor, positive, [], margin=0.5)

    # With a distant negative the margin is not violated → loss should be 0
    assert loss_with_neg == 0.0, (
        f"Expected 0 loss when negative is far, got {loss_with_neg}"
    )
    # Without negatives, loss equals the raw positive distance
    assert loss_no_neg > 0.0, "Loss with no negatives should equal positive distance"


def test_covariance_regularization_loss_is_positive_and_scales_with_inputs() -> None:
    from gaussian_rag.training.losses import covariance_regularization_loss

    dim, rank = 8, 4
    sigma_diag = np.ones(dim)
    sigma_L = np.ones((dim, rank))

    loss_base = covariance_regularization_loss(sigma_diag, sigma_L)
    loss_scaled = covariance_regularization_loss(sigma_diag * 2, sigma_L * 2)

    assert loss_base > 0.0, "Regularization loss must be strictly positive"
    assert loss_scaled > loss_base, "Doubling inputs must increase regularization loss"
