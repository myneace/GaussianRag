# Gaussian-Clef RAG Documentation Hub

Welcome to the comprehensive documentation for **Gaussian-Clef RAG** — a production-grade probabilistic retrieval-augmented generation engine operating on statistical manifolds with sub-100ms Cloudflare Clef decision routing, Woodbury-accelerated Expected Likelihood Kernels (ELK), embedded SQLite FTS5 lexical pre-filtering, and universal corpus generation.

---

## 🧭 Documentation Map

| Guide | Description | Key Topics |
|---|---|---|
| [**Codebase Architecture**](initial/CODEBASE.md) | Module-by-module breakdown of repository structure | `core/`, `creator/`, `rag/`, `facade.py`, `lexical_store.py`, `universal_ingestor.py`, `corpus_generator.py` |
| [**Usage Guide**](initial/USAGE.md) | Developer guide and execution walkthroughs | `GaussianRAG` 3-line API, `UniversalIngestor`, `data/generate_corpus.py` CLI, 3D dashboard server |
| [**Pipeline Specification**](initial/PIPELINE.md) | Full architectural breakdown of runtime retrieval & ingestion | 6-stage sub-100ms pipeline: Clef ambiguity gate, taxonomy routing, FTS5 pre-filter, passage gate, Woodbury ELK, generation |
| [**Design Decisions**](initial/DECISIONS.md) | Architectural decision records (ADRs) | DD-01 through DD-14: Woodbury acceleration, Clef-Flash routing, SQLite FTS5 lockstep indexing, facade API |
| [**Mathematical Foundations**](initial/THEORY.md) | Information geometry theory | Multivariate Gaussian representations $\mathcal{N}(\mu, \Sigma)$, SPD manifolds, ELK, Sylvester theorem |
| [**Learned Covariance (`ABOUT.md`)**](ABOUT.md) | PyTorch `SigmaHead` deep dive | Contrastive self-supervised learning, query radiation, background fine-tuning daemon |
| [**Research Questions**](initial/RESEARCH_QUESTIONS.md) | The 20 backbone scientific questions | Theoretical and empirical questions guiding manifold retrieval research |
| [**Experiments & Benchmarks**](initial/EXPERIMENTS.md) | Empirical evaluation protocol | Latency benchmarks, *48 Laws of Power* exact match testing, ablation studies |
| [**Concept Blueprints**](concepts/README.md) | Detailed specifications for polysemy resolution | Polysemy spec, agent dissolver, multi-agent disambiguation, GSKG notes |

---

## ⚡ Architecture at a Glance

### 1. Dual-Store Ingestion
Every knowledge chunk is ingested simultaneously into:
- **`KnowledgeStore`**: Dense Gaussian distributions $\mathcal{N}(\mu, \Sigma)$ stored with diagonal plus low-rank covariance factors ($r = 8$).
- **`LexicalStore`**: SQLite FTS5 virtual table (`passages_fts`) operating in WAL mode with BM25 candidate scoring.

### 2. Six-Stage Sub-100ms Retrieval Pipeline
1. **Clef Ambiguity Gate**: Non-autoregressive `noul` query gate running via Cloudflare Clef in <40ms.
2. **Taxonomy Branch Pruning**: Hierarchical document clustering pruning 60–80% of candidate branches before candidate generation.
3. **Hybrid Dual-Filter**: SQLite FTS5 BM25 search + μ-ANN generating ~30 high-signal candidates in <2ms.
4. **Clef 4-Question Passage Gate**: Parallel batched decision filtering for relevance, factual evidence, and contradictions.
5. **Woodbury ELK Reranking**: Analytical Expected Likelihood Kernel calculation accelerated from $O(d^3)$ to $O(d \cdot r^2)$ using low-rank Woodbury inversion and Sylvester determinant evaluation (<3ms).
6. **Provider Generation**: Uncertainty-grounded answer generation using Mistral, OpenAI, or Gemini.

### 3. Universal Corpus Generator & Text Normalizer
- **`TextNormalizer`**: Strips HTML tags/entities, fixes OCR line-break hyphenation (`word-\nbreak`), unwraps soft wraps, removes page numbers and header noise.
- **`CorpusGenerator`**: Adaptively detects document structure (numbered laws, Q&A pairs, transcripts, semantic sections) and exports to JSONL, Markdown, or text with content-addressed caching (`CorpusCache`).
- **`data/generate_corpus.py` CLI**: Unified terminal tool supporting piped stdin, file input, raw strings, and presets (`--preset 48-laws`).
