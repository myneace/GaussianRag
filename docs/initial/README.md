# Gaussian-Clef RAG — Probabilistic Retrieval-Augmented Generation on Information Manifolds

> *Treating knowledge not as points, but as distributions on statistical manifolds — accelerated to sub-100ms production response times.*

---

## What Is This?

Standard RAG systems embed knowledge chunks as **single points** in flat vector spaces and retrieve them using cosine similarity. This discards a fundamental property of knowledge: **uncertainty**.

Gaussian-Clef RAG represents every knowledge chunk as a **multivariate Gaussian distribution** $\mathcal{N}(\mu, \Sigma)$ in an information manifold:

- $\mu \in \mathbb{R}^d$ encodes the **central semantic meaning**.
- $\Sigma \in \text{SPD}(d)$ encodes **directional uncertainty** — representing ambiguity, multi-faceted aspects, and semantic spread ($\Sigma = \text{diag}(v) + LL^T$).

Retrieval operates through **Expected Likelihood Kernel (ELK)** overlap on the statistical manifold, accelerated via low-rank Woodbury inversion and non-autoregressive Cloudflare Clef decision routing.

---

## Core Thesis

> Standard RAG treats knowledge as points. We treat it as distributions on an information manifold. By combining Woodbury matrix acceleration, embedded SQLite FTS5 lexical pre-filtering, and non-autoregressive Clef decision gates, Gaussian RAG achieves sub-100ms end-to-end response times while preserving statistical manifold geometry.

---

## Documentation Index

### Primary Guides

| File | Description |
|---|---|
| [USAGE.md](./USAGE.md) | **Developer & Usage Guide** — Standalone library facade, CLI commands, data generator, and dashboard |
| [CODEBASE.md](./CODEBASE.md) | **Codebase Architecture** — Module structure, dependencies, and component responsibilities |
| [PIPELINE.md](./PIPELINE.md) | **Pipeline Specification** — 6-stage sub-100ms retrieval and dual-store universal ingestion |
| [DECISIONS.md](./DECISIONS.md) | **Design Decisions** — Architectural decision records (DD-01 through DD-15) |
| [THEORY.md](./THEORY.md) | **Mathematical Foundations** — Information geometry, SPD manifolds, and Woodbury ELK derivation |
| [EXPERIMENTS.md](./EXPERIMENTS.md) | **Experiments & Benchmarks** — Latency tests, *48 Laws of Power* exact-match verification, ablation |
| [RESEARCH_QUESTIONS.md](./RESEARCH_QUESTIONS.md) | **Research Questions** — 20 backbone theoretical and empirical questions |
| [../ABOUT.md](../ABOUT.md) | **PyTorch SigmaHead** — Learned covariance estimation and continual online radiation learning |

### Concept Specifications (`docs/concepts/`)

| File | Description |
|---|---|
| [docs/concepts/README.md](../concepts/README.md) | Index of all concept documents |
| [polysemy_rag_spec.md](../concepts/polysemy_rag_spec.md) | Production specification for polysemy-resilient multi-agent RAG |
| [agent_dissolver.md](../concepts/agent_dissolver.md) | Semantic ingestion engine and manifold filament generation |
| [multi_agent_disambiguation.md](../concepts/multi_agent_disambiguation.md) | Parallel semantic inference and routing pipeline |
| [gaussian_knowledge_graph.md](../concepts/gaussian_knowledge_graph.md) | Gaussian Semantic Knowledge Graphs (GSKG) concept notes |

---

## Quick Summary of the 6-Stage Retrieval Engine

```
User Query
    │
    ▼
[S1] Clef Ambiguity Gate ────── Fast non-autoregressive noul/choice gate (<40ms)
    │
    ▼
[S2] Taxonomy Routing ───────── Prunes 60-80% candidate branches in hierarchical tree
    │
    ▼
[S3] Hybrid Dual-Filter ─────── SQLite FTS5 BM25 + μ-ANN lockstep selection (<2ms)
    │
    ▼
[S4] Clef Passage Gate ──────── Parallel 4-question contradiction/relevance filter
    │
    ▼
[S5] Woodbury ELK Rerank ────── O(d * r^2) exact Gaussian overlap calculation (<3ms)
    │
    ▼
[S6] Provider Generation ────── Calibrated answer synthesis with uncertainty indicators
```

---

## Quick Start (Python Facade)

```python
from gaussian_rag import GaussianRAG

# Create local instance with zero cloud dependencies required
rag = GaussianRAG.create()

# Ingest texts or raw messy text
rag.add_texts([
    "Never outshine the master. Make those above you feel comfortably superior.",
    "Conceal your intentions. Keep people off-balance and in the dark.",
])

# Search with sub-100ms Woodbury ELK
results = rag.search("How to handle superiors?", top_k=3)
for chunk in results:
    print(f"[{chunk.rank}] {chunk.knowledge.text} (confidence: {chunk.confidence:.3f})")
```
