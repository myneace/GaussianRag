# Usage Guide — GaussianRAG

> How to install, add documents, run the pipeline, run queries (including polysemy-aware), and run the test suite.

---

## 1. Installation

```bash
# Clone and enter the repo
git clone <repo-url> && cd manifold

# Sync dependencies (uses uv — https://docs.astral.sh/uv/)
uv sync

# Install dev extras (required for running tests)
uv sync --extra dev
```

> **Note:** `sentence-transformers` is downloaded on first run. `all-MiniLM-L6-v2` (~90 MB) will be fetched from HuggingFace and cached automatically.

---

## 2. Configuration

### Environment file

```bash
cp .env.example .env
```

Edit `.env` with your values:

```bash
# Required only for `answer` and `polysemy-query` commands
MISTRAL_API_KEY="sk-..."
MISTRAL_BASE_URL="https://api.mistral.ai/v1/chat/completions"
MISTRAL_MODEL="mistral-small-latest"

# Optional overrides
GAUSSIAN_RAG_STORE_PATH=".gaussian_rag_store"   # default store location
GAUSSIAN_RAG_TOP_K=5                            # default retrieval top-k
```

### Config file

Pipeline settings live in `config/config.yaml`. Run from the **repo root** so the config is found correctly.

---

## 3. Adding Your Own Documents

Documents must be plain `.txt` files. Drop them into a directory:

```
my_docs/
├── document_a.txt
├── document_b.txt
└── subdirectory/
    └── document_c.txt
```

Or pass a single `.txt` file directly to `--input`.

**Ingest them into a local store:**

```bash
uv run python -m gaussian_rag.main ingest \
  --input my_docs \
  --output .gaussian_rag_store
```

This recursively finds all `.txt` files, chunks and embeds them as Gaussians, and writes a persistent JSON store to `.gaussian_rag_store/`.

Sample output:

```json
{
  "input": "my_docs",
  "output": ".gaussian_rag_store",
  "documents": 3,
  "chunks": 12
}
```

Prepare one or more UTF-8 `.txt` files in a local directory of your choice:

```bash
uv run python -m gaussian_rag.main ingest \
--input my_docs \
  --output .gaussian_rag_store
```

---

## 4. Running the Pipeline

All commands must be run **from the repo root**.

### 4.1 Standard Query (ELK retrieval)

Returns the top-k highest-overlap chunks from your ingested store.

```bash
uv run python -m gaussian_rag.main query \
  --store .gaussian_rag_store \
  --text "What are the treatment options for mitral valve regurgitation?" \
  --top-k 3
```

**Output** (JSON):
```json
[
  {
    "rank": 1,
    "id": "medical-chunk-0",
    "confidence": 0.91,
    "uncertainty_label": "low",
    "w2_distance": 0.124,
    "fr_distance": null,
    "elk_score": -0.018,
    "text": "Mitral valve regurgitation..."
  },
  ...
]
```

Switch to Fisher-Rao metric:

```bash
uv run python -m gaussian_rag.main query \
  --store .gaussian_rag_store \
  --text "ambiguous legal precedent" \
  --metric fisher_rao \
  --top-k 5
```

### 4.2 Polysemy-Aware Query (Disambiguation Pipeline)

For queries containing ambiguous terms (e.g. "Apple", "bank", "trial"), the 5-agent disambiguation pipeline fires first, then routes to single- or dual-sense retrieval based on entropy.

```bash
uv run python -m gaussian_rag.main polysemy-query \
  --store .gaussian_rag_store \
  --text "What are the outcomes of the Apple trial?" \
  --top-k 3
```

**Requires:** `MISTRAL_API_KEY` set in `.env` — Agent 1 (Sense Generator) calls the LLM.

**Output** (JSON):
```json
{
  "query": "What are the outcomes of the Apple trial?",
  "senses_detected": 2,
  "entropy": 1.82,
  "fallback_triggered": false,
  "selected_sense_id": null,
  "context_prompt": "[Sense A: technology/corporate]\n..."
}
```

| `entropy` | Routing |
|---|---|
| ≤ 1.5 bits | Single-sense — top match returned directly |
| 1.5 – 2.5 bits | Dual-sense — top-2 senses retrieved in parallel, results interleaved |
| > 2.5 bits | Fallback — broad retrieval, `[Sense A] / [Sense B]` tagged output |

### 4.3 Answer Generation (Retrieval + LLM)

Retrieves context from the store and generates a full answer via Mistral.

```bash
uv run python -m gaussian_rag.main answer \
  --store .gaussian_rag_store \
  --text "Summarize the ambiguity in legal precedent across jurisdictions" \
  --top-k 3
```

**Requires:** `MISTRAL_API_KEY` in `.env`, `generation.mode=provider` and `generation.provider=mistral` in `config/config.yaml`.

**Output** (JSON):
```json
{
  "query": "Summarize the ambiguity...",
  "retrieved": [...],
  "answer": "Legal precedent varies significantly across jurisdictions...",
  "prompt": "..."
}
```

### 4.4 Demo Commands (No Ingest Required)

These use a built-in 3-document corpus and don't need a persisted store.

```bash
# Demo retrieval — built-in medical/legal/scientific corpus
uv run python -m gaussian_rag.main demo-query \
  --text "ambiguous treatment guidance" \
  --top-k 3

# Demo evaluation — runs nDCG@k and Recall@k on built-in benchmark
uv run python -m gaussian_rag.main evaluate \
  --benchmark demo \
  --split test \
  --top-k 3

# Ablation run — compares retrieval metrics across configurations
uv run python -m gaussian_rag.main ablation
```

### 4.5 Field Visualization

Generate an interactive 3D topological map of your knowledge field. Concepts are clustered by semantic similarity (PCA-reduced means), while sphere sizes represent distributional entropy (uncertainty).

```bash
uv run python -m gaussian_rag.main visualize \
  --store .gaussian_rag_store \
  --output .visualizer
```

**Output:** Generates `.visualizer/index.html`.

To view it locally:
```bash
python3 -m http.server 8000 --directory .visualizer
```
Then open [http://localhost:8000](http://localhost:8000) in your browser.

---

## 5. CLI Reference

| Command | Required args | Optional args | Notes |
|---|---|---|---|
| `ingest` | `--input PATH` | `--output PATH` (default: `.gaussian_rag_store`) | Accepts a `.txt` file or directory of `.txt` files |
| `query` | `--text TEXT` | `--store PATH`, `--top-k N`, `--metric [elk\|wasserstein2\|fisher_rao]` | Requires prior `ingest` |
| `polysemy-query` | `--text TEXT` | `--store PATH`, `--top-k N`, `--metric` | Requires `MISTRAL_API_KEY` |
| `answer` | `--text TEXT` | `--store PATH`, `--top-k N`, `--metric` | Requires `MISTRAL_API_KEY` and `generation.mode=provider` |
| `demo-query` | `--text TEXT` | `--top-k N`, `--metric` | No store needed |
| `evaluate` | _(none)_ | `--benchmark demo`, `--split test`, `--top-k N` | Built-in benchmark only |
| `ablation` | _(none)_ | _(none)_ | Runs preset ablation matrix |
| `visualize` | _(none)_ | `--store PATH`, `--output PATH` | Generates 3D interactive HTML |

You can also invoke via the installed script entry point:

```bash
gaussian-rag query --store .gaussian_rag_store --text "my query"
```

---

## 6. Running the Tests

All tests live in `tests/test_pipeline.py`. Run them from the repo root.

### Run the full test suite

```bash
uv run pytest
```

### Run with verbose output

```bash
uv run pytest -v
```

### Run a specific test

```bash
uv run pytest -v -k "test_ingest_and_query_store_round_trip"
```

### Run only fast unit tests (skip CLI subprocess tests)

```bash
uv run pytest -v -k "not cli"
```

### Test categories

| Test | What it checks |
|---|---|
| `test_demo_pipeline_retrieves_results` | Demo corpus loads and store is non-empty |
| `test_phase1_metrics_are_bounded` | nDCG@3 and Recall@3 are in [0,1] and > 0 |
| `test_phase2_training_score_is_non_negative` | Augmentation consistency loss > 0 (views diverge) |
| `test_phase3_geometry_score_is_non_negative` | Coverage score > 0 (chunks differ in manifold) |
| `test_ablation_runner_returns_named_results` | Ablation matrix runs without error |
| `test_project_config_loads_expected_defaults` | `config.yaml` parses correctly |
| `test_provider_env_loads_mistral_keys_from_file` | `.env` file parsing works |
| `test_provider_client_parses_mistral_response` | Mistral HTTP response parsing (mocked) |
| `test_fisher_rao_metric_populates_distance_field` | FR distances are non-null after retrieval |
| `test_ingest_and_query_store_round_trip` | Ingest → persist → load → query all succeed |
| `test_query_requires_persisted_store` | Missing store raises a clear `ValueError` |
| `test_answer_uses_provider_backed_generator` | `answer` calls `ProviderBackedGenerator` (mocked) |
| `test_answer_runs_through_provider_client_contract` | `answer` sends HTTP request and parses response (mocked) |
| `test_answer_requires_provider_mode` | `generation.mode` validation fires correctly |
| `test_ablation_cli_runs` | CLI `ablation` subprocess produces expected JSON |
| `test_demo_query_cli_runs` | CLI `demo-query` subprocess returns chunk IDs |
| `test_ingest_and_query_cli_flow_runs` | Full CLI ingest + query subprocess flow |
| `test_evaluate_cli_runs` | CLI `evaluate` returns benchmark JSON |
| `test_answer_cli_uses_env_and_store` | CLI `answer` reads `.env` and store path |
| `test_sigma_head_predict_returns_valid_diag_and_low_rank` | SigmaHead shapes and positivity |
| `test_contrastive_w2_loss_pushes_positive_closer_than_margin` | Loss is 0 for distant negatives, > 0 otherwise |
| `test_covariance_regularization_loss_is_positive_and_scales_with_inputs` | Regularization loss scales monotonically |

---

## 7. Preparing Target Text

### Format

- Plain UTF-8 `.txt` files. No special markup required.
- One topic per file produces cleaner covariance estimates.
- Shorter, focused chunks (1–3 paragraphs) embed more precisely than long multi-topic files.

### What makes a good document for this system

| Good | Why |
|---|---|
| Specific, focused paragraphs | Produces tight $\Sigma$ → high-confidence retrieval |
| Contested / multi-interpretation content | Produces wide $\Sigma$ → system correctly signals uncertainty |
| Ambiguous terms (polysemous entities) | Exercises the `polysemy-query` disambiguation path |

### Writing queries

| Good query | Why |
|---|---|
| `"ambiguous treatment options for mitral valve regurgitation"` | Includes domain signal + ambiguity marker |
| `"conflicting evidence on drug interaction and toxicity"` | Triggers uncertainty-aware retrieval |
| `"What are the outcomes of the Apple trial?"` | Polysemous — will trigger disambiguation pipeline |

Avoid:
- Single-word queries (no distributional context for covariance)
- Exact verbatim sentences from documents (cosine degenerates to exact match; Wasserstein advantage disappears)

---

## 8. Programmatic Usage

```python
from gaussian_rag.core.gaussian_embedding import GaussianEmbedder
from gaussian_rag.creator.ingestion import IngestionPipeline
from gaussian_rag.core.store import KnowledgeStore
from gaussian_rag.rag.retrieval.retriever import GaussianRetriever

# Set up
embedder = GaussianEmbedder()
pipeline = IngestionPipeline(embedder)
store = KnowledgeStore()

# Ingest text
chunks = pipeline.ingest_document(
    "The mitral valve regulates blood flow. Treatment is context-dependent.",
    document_id="doc-1",
)
store.add(chunks)

retriever = GaussianRetriever(store)
retriever.refresh()

# Query
query = embedder.encode_query("What controls blood flow in the heart?", with_covariance=True)
results = retriever.retrieve(query, top_k=3, metric="elk")

for r in results:
    print(r.rank, r.uncertainty_label, r.elk_score, r.knowledge.text[:60])
```

### With disambiguation

```python
from gaussian_rag.rag.sense_cache import SenseCache
from gaussian_rag.rag.disambiguation.pipeline import disambiguate
from gaussian_rag.rag.disambiguation.router import route

cache = SenseCache()

# llm_caller must return raw JSON string matching the SenseGenerator schema
def my_llm(prompt: str) -> str:
    # wire to Mistral, OpenAI, or any LLM
    ...

packet = disambiguate("Apple trial outcomes", embedder, cache, llm_caller=my_llm)
print(f"Entropy: {packet.entropy:.2f} bits — Selected: {packet.selected_sense_id}")

sense_results, fallback = route(packet, embedder, retriever, top_k=3)
```
