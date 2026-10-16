# Codebase Structure — Gaussian-Clef RAG

> Every module maps to a specific part of the dual-store, probabilistic manifold retrieval, and universal corpus generation architecture. This document describes what each file does, what it depends on, and what it exposes.

---

## Design Principles

The codebase is split into three first-class sub-packages reflecting the operational layers of the system:

| Sub-package | Responsibility | Primary Classes |
|---|---|---|
| `gaussian_rag/core/` | **Foundational Manifold & Storage** — mathematical representations, geometry, low-rank covariance estimators, dual-store contracts | `KnowledgeStore`, `LexicalStore`, `GaussianEmbedder`, `GaussianKnowledge`, `WoodburyMatrix` |
| `gaussian_rag/creator/` | **Universal Ingestion & Corpus Generation** — text normalization, OCR/HTML cleanup, adaptive structure detection, content caching, format-agnostic ingestion | `UniversalIngestor`, `CorpusGenerator`, `TextNormalizer`, `CorpusCache`, `OpenAICorpusAssistant` |
| `gaussian_rag/rag/` | **High-Speed RAG Engine** — library facade, Cloudflare Clef routing, taxonomy pruning, SQLite FTS5 pre-filtering, passage gating, Woodbury ELK reranking, generation | `GaussianRAG`, `GaussianRetriever`, `ClefClient`, `ClefPassageGate`, `TaxonomyRouter`, `ManifoldRanker` |

The Creator and RAG engines communicate through clean dual-store contracts: `KnowledgeStore` (for $\mathcal{N}(\mu, \Sigma)$ embeddings) and `LexicalStore` (for SQLite FTS5 BM25 pre-filtering).

---

## Directory Tree

```
GaussianRag/
├── config/
│   └── config.yaml                      # Core pipeline configuration (encoder, retrieval, chunking, Clef)
├── dashboard/                           # React + Vite 3D Manifold visualization interface
│   ├── src/                             # Three.js / Canvas 3D ellipsoid rendering & SSE hooks
│   └── package.json
├── data/
│   ├── generate_corpus.py               # Flexible multi-source corpus generator CLI
│   └── training/                        # Contrastive training corpus for SigmaHead
├── docs/                                # Comprehensive documentation hub
│   ├── README.md                        # Documentation index and navigation map
│   ├── ABOUT.md                         # PyTorch SigmaHead & continual learning
│   ├── initial/                         # Core architecture, USAGE, PIPELINE, DECISIONS, THEORY
│   └── concepts/                        # Multi-agent disambiguation specifications
├── gaussian_rag/
│   ├── __init__.py                      # Public API exports (GaussianRAG, stores, embedders, normalizers)
│   ├── main.py                          # CLI entrypoint — ingest / query / answer / evaluate / visualize
│   ├── config.py                        # YAML and environment variable configuration loader
│   ├── py.typed                         # PEP 561 static typing marker
│   │
│   ├── core/                            # ── Shared mathematical foundation & storage ───────
│   │   ├── types.py                     # GaussianKnowledge, SenseNode, RetrievedChunk, etc.
│   │   ├── gaussian_embedding.py        # Text → N(μ, Σ) representation generator
│   │   ├── covariance.py                # Diagonal + low-rank Σ estimators & Woodbury acceleration
│   │   ├── information_geometry.py      # ELK, W2, Fisher-Rao, KL, Bhattacharyya, Bures metrics
│   │   ├── store.py                     # KnowledgeStore — dense distribution persistence
│   │   └── lexical_store.py             # LexicalStore — SQLite FTS5 BM25 lexical index
│   │
│   ├── creator/                         # ── Knowledge Base Creation & Ingestion ───────────
│   │   ├── text_normalizer.py           # TextNormalizer — OCR hyphenation, HTML, unicode, whitespace
│   │   ├── corpus_generator.py          # CorpusGenerator — structure auto-detection (laws, FAQ, transcripts)
│   │   ├── corpus_cache.py              # CorpusCache — SHA256 content-addressed generation cache
│   │   ├── llm_assistant.py             # OpenAICorpusAssistant — AI-assisted parsing helper
│   │   ├── universal_ingestor.py        # UniversalIngestor — format-agnostic ingestion adapter
│   │   ├── chunker.py                   # Sentence-aware token-bounded chunk splitter
│   │   └── ingestion.py                 # IngestionPipeline (legacy single-store adapter)
│   │
│   ├── rag/                             # ── RAG Engine & Library Facade ────────────────────
│   │   ├── facade.py                    # GaussianRAG — high-level 3-line public facade
│   │   ├── clef_client.py               # ClefClient — Cloudflare Workers AI Clef client
│   │   ├── sense_cache.py               # Session-scoped SenseNode TTL store
│   │   │
│   │   ├── retrieval/                   # Sub-100ms candidate retrieval and ranking
│   │   │   ├── taxonomy_router.py       # TaxonomyRouter — hierarchical document cluster pruning
│   │   │   ├── clef_passage_gate.py     # ClefPassageGate — 4-question contradiction/relevance filter
│   │   │   ├── ann_index.py             # μ-space approximate nearest-neighbour index
│   │   │   ├── manifold_ranker.py       # Vectorized Woodbury ELK and W2 re-ranking
│   │   │   ├── diversity.py             # W2-based diversity filter
│   │   │   ├── uncertainty_weighter.py  # Confidence-weighted scoring
│   │   │   ├── retriever.py             # GaussianRetriever — hybrid dual-store candidate selection
│   │   │   └── visualizer.py            # Interactive 3D Plotly knowledge-field visualizer
│   │   │
│   │   ├── disambiguation/              # 5-agent polysemy resolution pipeline
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
│   ├── manifold/                        # SPD cone helpers, tangent projections, coverage
│   ├── training/                        # SigmaHead PyTorch models, contrastive losses, trainer
│   └── evaluation/                      # Benchmark loaders and retrieval metrics (nDCG, Recall)
├── dashboard_server.py                  # FastAPI telemetry backend broadcasting live stage timings
├── test_48_laws_e2e.py                  # Live 48 Laws of Power end-to-end benchmark
├── tests/                               # Comprehensive unit and integration test suite
├── pyproject.toml                       # Standalone packaging metadata and dependencies
└── README.md                            # Project overview and quick start guide
```

---

## Core Module Specifications

### 1. `gaussian_rag/rag/facade.py` (`GaussianRAG`)

The primary public entrypoint providing a unified high-level interface:

```python
class GaussianRAG:
    @classmethod
    def create(cls, store_dir: str | Path | None = None, dimension: int = 384, ...) -> "GaussianRAG"
    def add_texts(self, texts: list[str], metadata: list[dict] | None = None) -> list[GaussianKnowledge]
    def add_raw_text(self, raw_text: str, title: str | None = None, ...) -> list[GaussianKnowledge]
    def add_markdown(self, markdown_text: str, document_id: str | None = None) -> list[GaussianKnowledge]
    def add_records(self, records: list[dict], text_field: str = "text", ...) -> list[GaussianKnowledge]
    def add_file(self, file_path: str | Path) -> list[GaussianKnowledge]
    def add_directory(self, dir_path: str | Path) -> list[GaussianKnowledge]
    def search(self, query: str, top_k: int = 5, metric: str = "elk") -> list[RetrievedChunk]
    def query(self, question: str, top_k: int = 5) -> str
    def save(self, directory: str | Path) -> None
    @classmethod
    def load(cls, directory: str | Path) -> "GaussianRAG"
```

### 2. `gaussian_rag/creator/universal_ingestor.py` (`UniversalIngestor`)

Universal format-agnostic adapter indexing into `KnowledgeStore` and `LexicalStore` in lockstep:

- Parses strings, Markdown heading trees, JSON/JSONL dictionaries, and CSV files.
- Automatically synchronizes $\mathcal{N}(\mu, \Sigma)$ nodes and SQLite FTS5 rows with identical chunk IDs.

### 3. `gaussian_rag/creator/corpus_generator.py` & `text_normalizer.py`

Processes arbitrary, unformatted, messy pasted text into clean corpora:

- **`TextNormalizer`**: Cleans HTML tags/entities, fixes OCR line-break hyphenation (`word-\nbreak`), unwraps soft wraps, removes running headers/footers, and normalizes unicode whitespace.
- **`CorpusGenerator`**: Adaptively detects document structure (`numbered_section`, `qa_pair`, `transcript`, `semantic_chunk`) and serializes to JSONL, clean Markdown with breadcrumbs, and plain text.
- **`CorpusCache`**: Content-addressed SHA-256 cache avoiding redundant parsing and embedding.

### 4. `gaussian_rag/core/lexical_store.py` (`LexicalStore`)

Embedded SQLite FTS5 full-text search engine:

- Operates in WAL mode with busy timeout handling.
- `passages_fts` table indexes chunk IDs, titles, and raw text.
- Provides BM25 lexical candidate pre-filtering in <2ms with Boolean-aware query sanitization.

### 5. `gaussian_rag/core/covariance.py` & Low-Rank ELK Acceleration

Eliminates the $O(d^3)$ bottleneck in Expected Likelihood Kernel evaluation:

- Implements `WoodburyMatrix` for $(\text{diag}(v) + LL^T)^{-1}$ inversion in $O(d \cdot r^2)$.
- Evaluates log determinants via the **Sylvester Determinant Theorem** in $O(r^3)$ instead of full matrix factorizations.

### 6. `gaussian_rag/rag/retrieval/retriever.py` (`GaussianRetriever`)

Hybrid dual-store candidate selection and reranking:

- Retrieves top lexical hits from `LexicalStore` and top semantic hits from ANN index.
- Fuses candidate pools to guarantee keyword/entity recall.
- Dispatches candidates through `ClefPassageGate` before `ManifoldRanker` Woodbury scoring.

### 7. `gaussian_rag/rag/clef_client.py` & `clef_passage_gate.py`

Cloudflare Workers AI Clef client:

- Evaluates non-autoregressive decision questions (`choice`, `noul`, `score`) in parallel batches.
- Prunes noisy, irrelevant, and contradictory passages down to high-signal survivors.

---

## Dependency Graph

```
gaussian_rag/core/types.py
    ↑
gaussian_rag/core/information_geometry.py ← gaussian_rag/core/covariance.py (Woodbury)
    ↑                                               ↑
gaussian_rag/core/gaussian_embedding.py             gaussian_rag/core/lexical_store.py (SQLite FTS5)
    ↑                                               ↑
gaussian_rag/core/store.py ─────────────────────────┘
    ↑
    ├── creator/
    │     ├── text_normalizer.py
    │     ├── corpus_generator.py ← corpus_cache.py & llm_assistant.py
    │     └── universal_ingestor.py (dual-indexes KnowledgeStore + LexicalStore)
    │
    └── rag/
          ├── clef_client.py
          ├── retrieval/
          │     ├── taxonomy_router.py
          │     ├── clef_passage_gate.py
          │     ├── ann_index.py
          │     ├── manifold_ranker.py (Woodbury ELK)
          │     └── retriever.py (Hybrid Dual-Store)
          └── facade.py (GaussianRAG)
```

---

## Execution Entry Points

| Tool / Command | File | Description |
|---|---|---|
| **Python Facade** | `from gaussian_rag import GaussianRAG` | 3-line high-level Python API for application integration |
| **Corpus Generator CLI** | `data/generate_corpus.py` | Command-line tool for raw text normalization, presets, and AI structuring |
| **Telemetry Server** | `dashboard_server.py` | FastAPI SSE server streaming real-time stage latencies |
| **3D Dashboard** | `dashboard/` | Interactive Vite/React 3D manifold visualizer |
| **CLI Commands** | `gaussian_rag/main.py` | Subcommands: `ingest`, `query`, `answer`, `polysemy-query`, `visualize` |
| **End-to-End Test** | `test_48_laws_e2e.py` | Full validation script on *The 48 Laws of Power* |
| **Test Suite** | `pytest` | Comprehensive unit and integration tests |
