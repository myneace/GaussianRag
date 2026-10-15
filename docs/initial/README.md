# GaussianRAG — Probabilistic Retrieval-Augmented Generation on Information Manifolds

> *Treating knowledge not as points, but as distributions.*

---

## What Is This?

Standard RAG systems embed knowledge chunks as **single points** in vector space and retrieve them using cosine similarity. This discards a fundamental property of knowledge: **uncertainty**.

GaussianRAG represents every knowledge chunk as a **Gaussian distribution** $\mathcal{N}(\mu, \Sigma)$ in an information manifold, where:

- $\mu$ encodes the **central semantic meaning**
- $\Sigma$ encodes **directional uncertainty** — how ambiguous, multi-faceted, or contested the knowledge is

Retrieval then happens not by cosine similarity, but by **distributional overlap on the statistical manifold** using the Expected Likelihood Kernel (ELK), with Wasserstein-2 and Fisher-Rao retained as alternate geometric analysis modes.

---

## Core Thesis

> Standard RAG treats knowledge as points. We treat it as distributions on an information manifold. This recovers directional semantic uncertainty, enables geometry-aware retrieval, and produces measurably better results on ambiguous queries — the exact queries where RAG currently fails most.

---

## Documentation Index

### Root Documentation

| File | Description |
|---|---|
| [README.md](./README.md) | This file — project overview |
| [USAGE.md](./USAGE.md) | **How to run** — install, add documents, run pipeline, run tests |
| [RESEARCH_QUESTIONS.md](./RESEARCH_QUESTIONS.md) | The 20 backbone research questions |
| [THEORY.md](./THEORY.md) | Mathematical foundations and information geometry |
| [PIPELINE.md](./PIPELINE.md) | Full system pipeline — end to end |
| [CODEBASE.md](./CODEBASE.md) | Project structure and module descriptions |
| [EXPERIMENTS.md](./EXPERIMENTS.md) | What we are testing, benchmarks, evaluation protocol |
| [DECISIONS.md](./DECISIONS.md) | Key design decisions and rationale |

### Concept Specifications (`docs/concepts/`)

| File | Description |
|---|---|
| [docs/concepts/README.md](../concepts/README.md) | Index of all concept documents |
| [polysemy_rag_spec.md](../concepts/polysemy_rag_spec.md) | Production spec: polysemy-resilient parallel multi-agent RAG |
| [polysemy_rag_gap_analysis.md](../concepts/polysemy_rag_gap_analysis.md) | Historical gap-analysis snapshot for the spec's early implementation phase |
| [gaussian_knowledge_graph.md](../concepts/gaussian_knowledge_graph.md) | GSKG: nodes as Gaussian distributions on a manifold |
| [agents_implementation.md](../concepts/agents_implementation.md) | Original agent brainstorm (superseded by polysemy\_rag\_spec) |

---

## Quick Summary of the Approach

```
Document → Chunk → Encode as N(μ, Σ) → Store on Manifold
                                                  ↓
Query → Encode as N(μ_q, Σ_q) → Retrieve via ELK overlap
                                                  ↓
         Weight by confidence → Augment LLM context → Generate
```

---

## Why This Matters

Current RAG failures cluster around:
- **Ambiguous queries** — flat retrieval doesn't know what the query *could* mean
- **Contested knowledge** — conflicting chunks are treated equally
- **Vague documents** — long, multi-faceted chunks are poorly represented as a single point

Gaussian distributions directly address all three.

---

## Target Domains

The gains are largest in domains with inherent uncertainty:

- **Medical** — differential diagnoses, guideline conflicts
- **Legal** — precedent ambiguity, jurisdictional variance
- **Scientific** — hypothesis vs. established fact
- **Finance** — forecast distributions, risk scenarios

---

## Status

> 🔬 Research prototype — pre-publication

## Prototype Scope in This Repository

The repository contains an executable prototype in `gaussian_rag/`.

- **Phase 0** adds a fully-implemented `gaussian_rag/rag/disambiguation/` module: a 5-agent LLM-driven polysemy resolution pipeline (Agent 0 plus Agents 1–4) that routes ambiguous queries to single-sense, dual-sense, or clarification fallback retrieval paths. The `gaussian_rag/rag/sense_cache.py` module provides session-scoped `SenseNode` storage with TTL.
- **Phase 1** is implemented as a runnable Gaussian retrieval system with a config-selected embedding backend over ingested local corpora, with ELK overlap as the default live ranker and W2/FR retained as alternate metrics.
- **Phase 2** includes the implemented PyTorch `SigmaHead`, training loop, and online radiation refinement path, while still remaining prototype-shaped rather than benchmark-hardened production infrastructure.
- **Phase 3** includes Fisher-Rao scoring, manifold helpers, and coverage/drift proxies rather than full FR indexing infrastructure.
- **Phase 4** supports env-backed provider generation for real answer calls, while keeping demo-only commands explicit.
- **Phase 5** includes experiment runners, tests, config, and notebook scaffolding for the research workflow.

This means the code is executable and phase-shaped, but retrieval is still a prototype local-store flow rather than a benchmark-validated production stack.

## Running the Prototype

### 1. Configure env-backed provider access

```bash
uv sync
cp .env.example .env
```

Fill `.env` with real local values before using `answer`. The current supported provider-backed answer path is Mistral only.

### 2. Persisted local corpus flow

```bash
uv run python -m gaussian_rag.main ingest --input my_docs --output .gaussian_rag_store
uv run python -m gaussian_rag.main query --store .gaussian_rag_store --text "ambiguity in legal precedent" --top-k 3
uv run python -m gaussian_rag.main answer --store .gaussian_rag_store --text "Summarize the ambiguity in legal precedent" --top-k 3
```

`query --store ...` searches the persisted local store written by `ingest`. `answer --store ...` uses that retrieved context plus the env-backed provider configuration to generate a real model response.

### 3. Built-in demo-only flow

```bash
uv run python -m gaussian_rag.main demo-query --text "ambiguous treatment guidance" --top-k 3
uv run python -m gaussian_rag.main evaluate --benchmark demo --split test
uv run python -m gaussian_rag.main ablation
```

Run the commands from the repository root so `config/config.yaml` is discovered correctly.

The `ingest` command accepts a `.txt` file or a directory of `.txt` files, writes a persistent prototype store, and `query --store ...` can search that ingested corpus.

The production-oriented commands are `ingest`, `query`, and `answer`. They operate on the persisted local store path from `--store` or the default `.gaussian_rag_store` path.

`demo-query`, `evaluate`, and `ablation` are demo-only: they use the built-in demo benchmark and demo corpus rather than the ingested local store.

Default query behavior constructs a query Gaussian for ELK and Fisher-Rao paths so the live retrieval rule matches the paper's query-bubble formulation.

The `answer` command is intentionally strict: it requires `generation.mode=provider`, `generation.provider=mistral`, and valid `MISTRAL_*` settings in `.env` or the environment. It does not silently fall back to template output.

### Visualize Knowledge Field

Generate a 3D interactive topological map of the knowledge field based on ingested semantic means and uncertainty covariance:

```bash
uv run python -m gaussian_rag.main visualize --store .gaussian_rag_store --output .visualizer
```
