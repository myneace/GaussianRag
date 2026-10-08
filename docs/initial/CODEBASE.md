# Codebase Structure — GaussianRAG

> Every module maps to a specific part of the pipeline. This document describes what each file does, what it depends on, and what it exposes.

---

## Design Principles

The codebase is split into three first-class sub-packages that reflect the two distinct halves of the system:

| Sub-package | Responsibility |
|---|---|
| `gaussian_rag/core/` | Shared foundation — data types, geometry, embeddings, store |
| `gaussian_rag/creator/` | **Knowledge Base Creator** — chunking, ingestion, drift measurement |
| `gaussian_rag/rag/` | **RAG Engine** — retrieval, disambiguation, generation, sense cache |

The Creator and RAG engine only communicate through the `core/` store contract (`KnowledgeStore`). Neither package imports from the other.

---

## Directory Tree

```
repo-root/
├── config/
│   └── config.yaml                      # Runtime configuration loaded from repo root
├── docs/
│   └── concepts/                        # Architecture specs and concept notes
│       ├── README.md
│       ├── polysemy_rag_spec.md         # Production spec: polysemy-resilient multi-agent RAG
│       ├── polysemy_rag_gap_analysis.md # Spec-to-codebase gap mapping
│       ├── gaussian_knowledge_graph.md  # GSKG architecture concept
│       └── agents_implementation.md    # Brainstorm notes (superseded by polysemy_rag_spec)
├── gaussian_rag/
│   ├── __init__.py
│   ├── main.py                          # CLI entrypoint — routes ingest / query / answer / visualize
│   ├── config.py                        # Loads config/config.yaml + provider .env
│   │
│   ├── core/                            # ── Shared foundation ──────────────────────────────
│   │   ├── types.py                     # GaussianKnowledge, SenseNode, RetrievedChunk, etc.
│   │   ├── gaussian_embedding.py        # Text → N(μ, Σ)
│   │   ├── covariance.py               # Heuristic + learned Σ estimators
│   │   ├── information_geometry.py      # W2, Fisher-Rao, KL, Bhattacharyya, Bures
│   │   └── store.py                     # KnowledgeStore — persistence contract used by both halves
│   │
│   ├── creator/                         # ── Knowledge Base Creator ──────────────────────────
│   │   ├── chunker.py                   # Sentence-aware token-bounded chunk splitter
│   │   ├── ingestion.py                 # IngestionPipeline — chunk → embed → store
│   │   └── drift.py                     # W2-based knowledge drift measurement
│   │
│   ├── rag/                             # ── RAG Engine ──────────────────────────────────────
│   │   ├── sense_cache.py               # Session-scoped SenseNode TTL store
│   │   │
│   │   ├── retrieval/                   # Candidate retrieval and ranking
│   │   │   ├── ann_index.py             # μ-space approximate nearest-neighbour index
│   │   │   ├── manifold_ranker.py       # Exact ELK / W2 / FR re-ranking
│   │   │   ├── diversity.py             # W2-based diversity filter
│   │   │   ├── uncertainty_weighter.py  # Confidence-weighted scoring
│   │   │   ├── retriever.py             # GaussianRetriever — orchestrates retrieve / multi-sense
│   │   │   └── visualizer.py            # Interactive 3D Plotly knowledge-field visualisation
│   │   │
│   │   ├── disambiguation/              # 5-agent polysemy resolution
│   │   │   ├── query_dissolver.py       # Agent 0 — ambiguity detection + interpretation draft
│   │   │   ├── sense_generator.py       # Agent 1 — prompt assembly / sense shaping
│   │   │   ├── domain_grounder.py       # Agent 2 — manifold-grounded confidence scoring
│   │   │   ├── field_mapper.py          # Agent 3 — Bhattacharyya edges, Shannon entropy
│   │   │   ├── memory_packager.py       # Agent 4 — SenseCache ops, contrastive hints
│   │   │   ├── router.py                # Entropy gate → single / dual / fallback path
│   │   │   └── pipeline.py              # disambiguate() — top-level entry point
│   │   │
│   │   └── generation/                  # Context building and LLM answer generation
│   │       ├── context_builder.py       # build_context / build_multi_sense_context
│   │       ├── prompt_templates.py      # Uncertainty-aware prompt template
│   │       ├── provider.py              # MistralChatClient — env-backed HTTP call
│   │       └── generator.py             # SimpleGenerator + ProviderBackedGenerator
│   │
│   ├── manifold/                        # SPD helpers and coverage utilities
│   ├── training/                        # Sigma-head fitting, contrastive losses, trainer
│   ├── evaluation/                      # Demo benchmark loader and retrieval metrics
│   └── experiments/                     # Executable phase experiments (phase1–4, ablation)
├── scripts/
│   ├── polysemy_demo.py                 # End-to-end disambiguation + retrieval demo
│   └── validate_manifold.py             # W2-based manifold audit tool
├── tests/
│   └── test_pipeline.py                 # Pipeline-focused test suite
├── pyproject.toml
└── README.md
```

---

## Module Details

### `core/types.py`

The central data types. Everything else builds on these.

```python
@dataclass
class GaussianKnowledge:
    id: str
    text: str
    mu: np.ndarray          # shape: (d,)
    sigma_diag: np.ndarray  # shape: (d,)    diagonal component of Σ
    sigma_L: np.ndarray     # shape: (d, r)  low-rank component of Σ
    confidence: float       # scalar: 1 / (1 + tr(Σ))
    metadata: dict

@dataclass
class SenseNode:
    sense_id: str           # stable hash of (term, domain, retrieval_query)
    term: str
    domain: str
    confidence: float
    context_hints: list[str]
    supporting_evidence: list[str]
    retrieval_query: str
    mu_anchor: np.ndarray | None
    sigma_anchor: np.ndarray | None
    sigma_L_anchor: np.ndarray | None
    ttl_hours: float        # TTL for SenseCache expiry
    version: int
    created_at: float

@dataclass
class RetrievedChunk:
    knowledge: GaussianKnowledge
    w2_distance: float
    fr_distance: float | None
    elk_score: float | None
    confidence: float
    uncertainty_label: str  # "high" | "medium" | "low"
    rank: int
```

---

### `core/store.py`

Shared persistence contract used by both the Creator (write path) and the RAG engine (read path). Neither half imports from the other — only from `core`.

```python
class KnowledgeStore:
    def add(self, chunks: list[GaussianKnowledge]) -> None
    def get(self, ids: list[str]) -> list[GaussianKnowledge]
    def values(self) -> list[GaussianKnowledge]
    def save(self, directory: str | Path) -> None
    @classmethod
    def load(cls, directory: str | Path) -> "KnowledgeStore"
```

---

### `core/gaussian_embedding.py`

Encodes text into $\mathcal{N}(\mu, \Sigma)$.

```python
class GaussianEmbedder:
    def encode(self, text: str) -> GaussianKnowledge
    def encode_batch(self, texts: list[str]) -> list[GaussianKnowledge]
    def encode_query(self, query: str, with_covariance: bool = False) -> QueryRepresentation
    def encode_sense(self, sense: SenseNode) -> GaussianKnowledge
```

**Dependencies:** `core/covariance.py`, `training/sigma_head.py`

---

### `core/information_geometry.py`

All distance metrics between Gaussians.

```python
def wasserstein2(g1: GaussianKnowledge, g2: GaussianKnowledge) -> float
def fisher_rao(g1: GaussianKnowledge, g2: GaussianKnowledge) -> float
def kl_divergence(g1: GaussianKnowledge, g2: GaussianKnowledge) -> float
def bhattacharyya(g1: GaussianKnowledge, g2: GaussianKnowledge) -> float
def bures_metric(S1: np.ndarray, S2: np.ndarray) -> float
```

**Note:** `fisher_rao` uses diagonal approximation for tractability.

---

### `creator/ingestion.py`

IngestionPipeline — the write path into a KnowledgeStore.

```python
class IngestionPipeline:
    def __init__(self, embedder: GaussianEmbedder, config: ChunkingConfig)
    def ingest_document(self, text: str, document_id: str) -> list[GaussianKnowledge]
    def ingest_directory(self, input_dir: Path) -> list[GaussianKnowledge]
```

**Dependencies:** `creator/chunker.py`, `core/gaussian_embedding.py`, `core/store.py`

---

### `creator/chunker.py`

```python
def split_into_chunks(text: str, *, max_tokens: int = 128, overlap: int = 16) -> list[str]
```

Sentence-aware splitter with sliding overlap. Chunking granularity directly affects Σ quality.

---

### `creator/drift.py`

Measures W2-based knowledge drift between two store snapshots.

```python
def measure_drift(previous: list[GaussianKnowledge], current: list[GaussianKnowledge]) -> list[DriftMeasurement]
def mean_drift(previous: list[GaussianKnowledge], current: list[GaussianKnowledge]) -> float
```

---

### `rag/sense_cache.py`

Session-scoped TTL store for SenseNode objects. Mirrors the KnowledgeStore interface but operates on senses rather than knowledge chunks.

```python
class SenseCache:
    def upsert(self, nodes: list[SenseNode]) -> None
    def get_by_term(self, term: str) -> list[SenseNode]
    def context_window(self) -> list[str]   # active sense_ids
    def prune_expired(self) -> int
    def save(self, directory: str | Path) -> None
    @classmethod
    def load(cls, directory: str | Path) -> "SenseCache"
```

---

### `rag/retrieval/retriever.py`

Main retrieval interface. Orchestrates ANN → exact ELK/W2/FR ranking → diversity filter.

```python
class GaussianRetriever:
    def retrieve(self, query: QueryRepresentation, top_k: int = 5, metric: str = "elk") -> list[RetrievedChunk]
    def retrieve_with_diversity(self, query: QueryRepresentation, top_k: int = 5) -> list[RetrievedChunk]
    def retrieve_multi_sense(
        self,
        senses: list[tuple[str, float, QueryRepresentation]],
        top_k: int = 5,
        metric: str = "elk",
        fusion: str = "interleave",
    ) -> tuple[dict[str, list[RetrievedChunk]], list[RetrievedChunk]]
```

---

### `rag/retrieval/visualizer.py`

Interactive 3D PCA + Plotly visualisation of the Gaussian knowledge field.

```python
def generate_visualization(store_path: str | Path, output_dir: str | Path) -> str
```

---

### `rag/disambiguation/pipeline.py`

Top-level entry point for the 5-agent polysemy resolution pipeline.

```python
def disambiguate(
    text: str,
    embedder: GaussianEmbedder,
    cache: SenseCache,
    retriever: GaussianRetriever | None = None,
    llm_caller: Callable[[str], str] | None = None,
) -> DisambiguationPacket
```

**Agent chain:** `query_dissolver` → `sense_generator` → `domain_grounder` → `field_mapper` → `memory_packager` → `router`

---

### `rag/generation/generator.py`

```python
class SimpleGenerator:
    def generate(self, query: str, retrieved: list[RetrievedChunk]) -> dict[str, str]

class ProviderBackedGenerator:
    def generate(self, query: str, retrieved: list[RetrievedChunk]) -> dict[str, str]
```

`ProviderBackedGenerator` calls `MistralChatClient` using env-configured credentials and refuses to silently fall back to template output.

---

## Dependency Graph

```
core/types.py  (GaussianKnowledge, SenseNode, RetrievedChunk, DisambiguationPacket, …)
    ↑
core/information_geometry.py ← core/covariance.py
    ↑                                ↑
core/gaussian_embedding.py     training/sigma_head.py ← training/losses.py ← training/trainer.py
    ↑
core/store.py
  ↗              ↖
creator/          rag/
  chunker.py       sense_cache.py
  ingestion.py     retrieval/
  drift.py           ann_index.py → manifold_ranker.py → diversity.py → retriever.py
                   disambiguation/
                     query_dissolver → domain_grounder → field_mapper → memory_packager → router → pipeline
                   generation/
                     context_builder → generator
```

The `creator/` and `rag/` packages share **no direct imports**. All data flows through `core/store.py`.

---

## Tech Stack

| Component | Library | Version |
|---|---|---|
| Text encoding | Local MiniLM or Gemini Cloud (config-driven) | bundled config + provider deps |
| ANN index | NumPy linear scan (`rag/retrieval/ann_index.py`) | bundled |
| Manifold ops | `numpy`, `scipy` | latest |
| Σ estimation | Heuristic or learned (`core/covariance.py`) | bundled |
| Disambiguation | 5-agent pipeline (`rag/disambiguation/`) | bundled |
| LLM generation | Env-backed Mistral (`rag/generation/provider.py`) | bundled + `httpx` |
| Benchmarks | Demo benchmark scaffold (`evaluation/`) | bundled |
| Config | `pyyaml` | latest |
| Notebooks | `jupyter`, `matplotlib` | latest |

---

## Entry Points

```bash
# Ingest .txt documents into a local store
uv run python -m gaussian_rag.main ingest --input my_docs --output .gaussian_rag_store

# Query an ingested store
uv run python -m gaussian_rag.main query --store .gaussian_rag_store --text "What is the mechanism of action?" --top-k 5

# Generate a provider-backed answer over an ingested store
uv run python -m gaussian_rag.main answer --store .gaussian_rag_store --text "Summarize the mechanism of action" --top-k 5

# Generate an interactive 3D visualisation of the knowledge field
uv run python -m gaussian_rag.main visualize --store .gaussian_rag_store --output .visualizer

# Run demo-only retrieval / evaluation flows
uv run python -m gaussian_rag.main demo-query --text "ambiguous treatment guidance" --top-k 5
uv run python -m gaussian_rag.main evaluate --benchmark demo --split test

# Run ablation
uv run python -m gaussian_rag.main ablation
```

`ingest` runs through `creator/ingestion.py` → `core/store.py` (write).  
`query` and `answer` load the persisted store via `core/store.py` and route through `rag/retrieval/retriever.py` (read).  
`answer` calls `rag/generation/generator.py` and requires `MISTRAL_API_KEY` in the environment or `.env`.
