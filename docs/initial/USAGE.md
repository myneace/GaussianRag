# Usage Guide — Gaussian-Clef RAG

> Comprehensive developer guide: installation, high-level Python API, flexible corpus generation CLI, hybrid dual-store search, telemetry dashboard, and evaluation suites.

---

## 1. Installation

Install the library in editable mode with standard tools or `uv`:

```bash
# Clone the repository
git clone <repo-url> && cd manifold/GaussianRag

# Install standalone library with pip
pip install -e .

# Alternatively, sync dependencies using uv
uv sync
```

> **Note:** On first execution, the default local embedding model `sentence-transformers/all-MiniLM-L6-v2` (~90 MB) will be downloaded and cached automatically. No external API keys are required for local operation.

---

## 2. Configuration & Environment Variables

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Set environment variables as needed:

```ini
# Optional Cloudflare Workers AI (Enables Clef decision routing & passage gating)
CLOUDFLARE_ACCOUNT_ID="your_cloudflare_account_id"
CLOUDFLARE_API_TOKEN="your_cloudflare_api_token"

# Optional Cloud LLM Generation (OpenAI / Gemini / Mistral)
GEMINI_API_KEY="your_gemini_key"
OPENAI_API_KEY="your_openai_key"
MISTRAL_API_KEY="your_mistral_key"

# Optional Store & Runtime Defaults
GAUSSIAN_RAG_STORE_PATH=".gaussian_rag_store"
GAUSSIAN_RAG_TOP_K=5
```

Pipeline hyper-parameters reside in `config/config.yaml`:
- `encoder.embedding_backend`: `"local"` or `"gemini_cloud"`
- `encoder.dimension`: `384` (for MiniLM) or `768` (for Gemini)
- `encoder.low_rank`: `8` (rank of covariance factor $L$)
- `retrieval.metric`: `"elk"` (Woodbury-accelerated Expected Likelihood Kernel)

---

## 3. High-Level Python API (`GaussianRAG`)

The `GaussianRAG` facade provides a clean, unified interface for ingestion, storage, search, and answer generation.

### 3.1 Basic Usage

```python
from gaussian_rag import GaussianRAG

# 1. Initialize local instance (zero API keys required)
rag = GaussianRAG.create()

# 2. Add texts
rag.add_texts([
    "Never outshine the master. Make those above you feel comfortably superior.",
    "Always say less than necessary. When you try to impress people with words, the more you say, the more common you appear.",
    "Conceal your intentions. Keep people off-balance and in the dark by never revealing the purpose behind your actions.",
])

# 3. Search using Woodbury-accelerated ELK
results = rag.search("How should I behave around superiors?", top_k=3)
for chunk in results:
    print(f"[{chunk.rank}] (Confidence: {chunk.confidence:.3f}) {chunk.knowledge.text}")

# 4. Generate grounded answer (uses configured provider or SimpleGenerator)
answer = rag.query("How do you handle insecure leaders?")
print(answer)
```

### 3.2 Ingesting Raw Messy Text with Normalization

```python
dirty_html = """
<p>Law 1: Never Outshine the Master<br/>
Always make those above you feel comfortably su-
perior. In your desire to please them, do not go too far.</p>

<p>Law 2: Never Put Too Much Trust in Friends<br/>
Former enemies are often more loyal than friends.</p>
"""

# Automatically cleans HTML, repairs hyphenated line wraps, and structures chunks:
rag.add_raw_text(dirty_html, title="48 Laws Excerpt")
```

### 3.3 Universal Format Ingestion

```python
# Markdown with heading hierarchy
rag.add_markdown("""# Company Handbook
## Vacation Policy
Employees receive 20 days paid leave per year.
## Remote Work
Hybrid schedules require two office days per week.
""", document_id="handbook")

# Structured JSON / JSONL records
rag.add_records([
    {"id": "faq_01", "text": "Returns accepted within 30 days.", "category": "Returns"},
    {"id": "faq_02", "text": "Warranty lasts 12 months.", "category": "Support"}
], text_field="text", id_field="id")

# Files and entire directory trees
rag.add_file("docs/whitepaper.md")
rag.add_directory("knowledge_base/")
```

### 3.4 Persistence (Save & Load)

```python
# Save knowledge store and SQLite FTS5 index to disk
rag.save(".my_knowledge_store")

# Load saved store in one line
loaded_rag = GaussianRAG.load(".my_knowledge_store")
results = loaded_rag.search("vacation allowance", top_k=3)
```

---

## 4. Flexible Corpus Generator CLI (`data/generate_corpus.py`)

The multi-source corpus generator transforms arbitrary unformatted text into structured corpora (JSONL, Markdown, plain text) with SHA256 content-addressed caching.

```bash
# 1. Preset 48 Laws of Power benchmark corpus
python data/generate_corpus.py --preset 48-laws -o data/48_laws_of_power.txt

# 2. Pipe unformatted text from stdin into JSONL
cat raw_notes.txt | python data/generate_corpus.py --stdin -o corpus.jsonl

# 3. Clean messy OCR scan text into structured Markdown
python data/generate_corpus.py -i dirty_scan.txt -o clean_corpus.md -f markdown

# 4. Supply raw text string directly
python data/generate_corpus.py -t "Law 1: Never outshine the master..." -o laws.jsonl

# 5. Enable OpenAI-compatible AI assistant for difficult OCR repairs
python data/generate_corpus.py -i ocr_scan.txt -o corpus.jsonl --ai
```

### Supported CLI Options

| Flag | Argument | Description |
|---|---|---|
| `--preset` | `48-laws` | Generate built-in benchmark dataset |
| `-i`, `--input` | `FILE` | Path to raw input file |
| `-t`, `--text` | `STRING` | Raw unformatted text string |
| `--stdin` | _(flag)_ | Read raw text piped from standard input |
| `-o`, `--output` | `FILE` | Target output path (defaults to stdout) |
| `-f`, `--format` | `jsonl \| markdown \| text` | Output formatting (inferred from `-o` extension if omitted) |
| `--ai` | _(flag)_ | Enable OpenAI-compatible LLM structuring assistant |
| `--no-cache` | _(flag)_ | Bypass content-addressed hash cache |
| `--chunk-size` | `INTEGER` | Chunk token length (default: 150) |

---

## 5. Command-Line RAG Operations (`main.py`)

### 5.1 Ingest Documents into Dual-Store

Ingests documents into both `KnowledgeStore` and SQLite FTS5 `LexicalStore`:

```bash
python -m gaussian_rag.main ingest \
  --input data/my_docs \
  --output .gaussian_rag_store
```

### 5.2 Query Knowledge Store (ELK Retrieval)

```bash
python -m gaussian_rag.main query \
  --store .gaussian_rag_store \
  --text "How should I handle insecure leaders?" \
  --top-k 5 \
  --metric elk
```

### 5.3 Answer Generation (Provider-Backed)

```bash
python -m gaussian_rag.main answer \
  --store .gaussian_rag_store \
  --text "Summarize strategies for managing powerful superiors" \
  --top-k 3
```

### 5.4 Polysemy-Aware Query Disambiguation

```bash
python -m gaussian_rag.main polysemy-query \
  --store .gaussian_rag_store \
  --text "What are the outcomes of the Apple trial?" \
  --top-k 3
```

### 5.5 Interactive 3D Plotly Visualization

```bash
python -m gaussian_rag.main visualize \
  --store .gaussian_rag_store \
  --output .visualizer
```

Open `.visualizer/index.html` in your browser to inspect the semantic manifold.

---

## 6. 3D Manifold Dashboard & Telemetry Server

Gaussian RAG includes a real-time 3D dashboard displaying live stage latencies streamed via Server-Sent Events (SSE):

```bash
# Terminal 1: Start FastAPI telemetry server (port 8001)
python dashboard_server.py

# Terminal 2: Start Vite dashboard frontend (port 5173)
cd dashboard
npm install
npm run dev
```

Visit `http://localhost:5173` to explore:
- Interactive 3D PCA Gaussian ellipsoids with uncertainty axes.
- Real-time SSE stage telemetry (Clef gating, taxonomy routing, FTS5 lookup, Woodbury ELK).
- Live Clef decision routing confidence traces.

---

## 7. Testing & Verification

### 7.1 End-to-End Evaluation on *The 48 Laws of Power*
Validates complete pipeline execution with exact-match Law 1 retrieval:

```bash
python test_48_laws_e2e.py
```

### 7.2 Latency Verification (<100ms)
```bash
pytest tests/test_benchmark_latency.py -v
```

### 7.3 Facade & Universal Ingestion Tests
```bash
pytest tests/test_library_facade.py -v
```

### 7.4 Corpus Generator & Normalizer Tests
```bash
pytest tests/test_corpus_generator.py -v
```

### 7.5 Dual-Store Synchronization Tests
```bash
pytest tests/test_dual_store_sync.py -v
```

### 7.6 Run Full Test Suite
```bash
pytest
```
