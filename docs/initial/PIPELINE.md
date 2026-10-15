# Pipeline — GaussianRAG End-to-End System

> This document describes the full system pipeline from document ingestion to LLM-augmented generation.

---

## Overview

This document mixes the long-term target architecture with the current executable prototype. When the two differ, the text below calls out the current prototype explicitly.

```
┌─────────────────────────────────────────────────────────────┐
│                        INGESTION                            │
│  Documents → Chunking → Encoding → Σ Estimation → Store    │
└─────────────────────────────┬───────────────────────────────┘
                              │
                    Knowledge Manifold
                   { N(μ_i, Σ_i) }
                              │
┌─────────────────────────────┴───────────────────────────────┐
│                        RETRIEVAL                            │
│  Query → Encode → ELK Overlap → Rank → Confidence Weight    │
└─────────────────────────────┬───────────────────────────────┘
                              │
                    Retrieved Gaussians
                  with uncertainty scores
                              │
┌─────────────────────────────┴───────────────────────────────┐
│                       GENERATION                            │
│  Build Uncertainty-Aware Context → Augment LLM → Generate  │
└─────────────────────────────────────────────────────────────┘
```

---

## Phase 0 — Disambiguation (Polysemy Resolution)

Before ingestion or retrieval, ambiguous queries are routed through the 5-agent disambiguation pipeline. This phase is optional for non-ambiguous input and runs automatically when the text contains polysemous terms.

```
Input Text
    ↓
[QueryDissolver] ────────── Agent 0 — ambiguity detection + interpretation draft
    ↓
[SenseGenerator] ────────── LLM-driven Agent 1
    ↓                         Produces SenseNode list for ambiguous terms
[DomainGrounder] ────────── Agent 2 — W2-proximity scoring + confidence
    ↓
[FieldMapper] ───────────── Agent 3 — Bhattacharyya edges, Shannon entropy
    ↓
[MemoryPackager] ────────── Agent 4 — SenseCache TTL, contrastive hints
    ↓
[Router] ────────────────── Entropy gate:
         entropy ≤ 1.5 bits  → single-sense retrieval
         entropy 1.5–2.5     → dual-sense parallel retrieval + interleave fusion
         entropy > 2.5       → clarification fallback or broad multi-sense output
```

See `docs/concepts/polysemy_rag_spec.md` for the full specification. The pipeline lives in `gaussian_rag/rag/disambiguation/`.

---

## Phase 1 — Ingestion

### 1.1 Document Chunking

Input documents are split into chunks. Strategy affects $\Sigma$ quality significantly.

```
Document
   ↓
Sentence-aware splitter (128 tokens, 16 overlap)
   ↓
Chunks: [c_1, c_2, ..., c_n]
```

**Chunking decisions that affect $\Sigma$:**
- Short, specific chunks → naturally small $\Sigma$
- Long, multi-faceted chunks → naturally large $\Sigma$
- Overlapping chunks → higher inter-chunk Gaussian overlap

### 1.2 Mean Encoding ($\mu$)

Each chunk is encoded by a frozen or fine-tuned sentence encoder:

```
chunk c_i  →  [Encoder]  →  μ_i ∈ ℝ^d
```

The active encoder is configuration-driven. The repository supports local `sentence-transformers/all-MiniLM-L6-v2` and Gemini Cloud embeddings; the live runtime follows `config/config.yaml`.

The encoder output is the mean $\mu_i$ of the Gaussian. This is the standard dense embedding — no change here.

### 1.3 Covariance Estimation ($\Sigma$)

This is the novel component. Two modes:

**Mode A — Heuristic (Phase 1 / Prototype)**

Estimate $\Sigma$ from chunk properties without training:

```python
def heuristic_sigma(chunk, mu, encoder):
    # Generate K augmented views
    augmented = augment(chunk, K=8)
    mus = [encoder(a) for a in augmented]
    
    # Empirical covariance of augmented encodings
    Sigma = empirical_covariance(mus)
    
    # Project to diagonal + low-rank
    return diag_plus_lowrank(Sigma, rank=8)
```

**Mode B — Learned (Phase 2 / Main Contribution)**

A small MLP head $f_\phi$ on top of the encoder outputs $\Sigma$:

```
chunk c_i  →  [Encoder]  →  h_i  →  [MLP head f_φ]  →  (v_i, L_i)
                                                              ↓
                                              Σ_i = diag(v_i) + L_i L_i^T
```

Trained with contrastive probabilistic loss (see THEORY.md §5.2).

### 1.4 Knowledge Store

Each chunk is stored as a `GaussianKnowledge` object:

```python
@dataclass
class GaussianKnowledge:
    id: str
    text: str
    mu: np.ndarray        # (d,)
    sigma_diag: np.ndarray  # (d,)  diagonal component
    sigma_L: np.ndarray   # (d, r) low-rank component
    confidence: float     # = 1 / (1 + tr(Σ))
    metadata: dict
```

Target storage is a hybrid index:
- $\mu$ vectors → **FAISS** (approximate nearest neighbor on mean)
- $\Sigma$ components → **flat store** (retrieved by ID for exact W2 computation)

The current executable prototype keeps both in local Python structures and JSON persistence via `gaussian_rag/core/store.py`. `gaussian_rag/creator/ingestion.py` drives the write path; `gaussian_rag/rag/retrieval/retriever.py` drives the read path.

---

## Phase 2 — Retrieval

Target architecture is described first in this section. Current prototype deviations are called out inline where they differ.

### 2.1 Query Encoding

**Current live query path (ELK / Fisher-Rao):**
```
query text  →  [Encoder + Σ estimator]  →  N(μ_q, Σ_q)
```

**Alternate point-query behavior (legacy / explicit metric choice):**
```
query text  →  [Encoder]  →  μ_q ∈ ℝ^d
```

Where $\Sigma_q$ is estimated by the active covariance mode (heuristic, learned, entity-aware, or dissolver fallback) whenever the retrieval metric requires a full query bubble.

### 2.2 Candidate Retrieval

Target first pass: approximate nearest neighbor on $\mu$ vectors via FAISS:

```
μ_q  →  FAISS ANN  →  top-K candidates by μ distance
```

The current prototype uses a NumPy candidate scan (`gaussian_rag/rag/retrieval/ann_index.py`) instead. It preserves the same two-stage retrieval shape but not the production ANN backend.

### 2.3 Exact Manifold Ranking

Second pass: compute exact Gaussian overlap between query and each candidate:

```
for each candidate N(μ_i, Σ_i):
    score_i = N(μ_q; μ_i, Σ_q + Σ_i)   # Expected Likelihood Kernel
```

Re-rank candidates by descending `score_i`. Wasserstein-2 and Fisher-Rao remain available as alternate geometric probes.

### 2.4 Diversity Filtering

Remove redundant retrievals via manifold overlap check:

```
selected = []
for candidate in ranked_candidates:
    if all W2(candidate, s) > diversity_threshold for s in selected:
        selected.append(candidate)
    if len(selected) == top_k:
        break
```

### 2.5 Output

```python
@dataclass
class RetrievedChunk:
    knowledge: GaussianKnowledge
    w2_distance: float
    fr_distance: float | None
    elk_score: float | None
    confidence: float        # 1 / (1 + tr(Σ))
    uncertainty_label: str   # "high" | "medium" | "low"
```

---

## Phase 3 — Generation

This section primarily describes the target system. In the current executable prototype, generation can call a live env-configured provider for the `answer` path, but retrieval remains the local prototype store path described elsewhere in this document.

### 3.1 Context Construction

Retrieved chunks are assembled into an uncertainty-aware context:

```
[CONTEXT START]

[Chunk 1] — Confidence: HIGH
"The mitral valve regulates blood flow between the left atrium and ventricle..."

[Chunk 2] — Confidence: MEDIUM
"Treatment options vary depending on severity and patient history..."

[Chunk 3] — Confidence: LOW (multiple interpretations exist)
"Recent studies suggest conflicting evidence regarding..."

[CONTEXT END]
```

### 3.2 LLM Prompt Template

```
You are given retrieved knowledge chunks with confidence levels.
HIGH confidence = well-established, specific knowledge.
MEDIUM confidence = generally accepted, some variance.
LOW confidence = contested, ambiguous, or uncertain knowledge.

Use confidence levels to calibrate your response — hedge appropriately
for LOW confidence information, and be definitive for HIGH confidence.

CONTEXT:
{uncertainty_weighted_context}

QUERY:
{query}

ANSWER:
```

### 3.3 Current Prototype Generation Mode

In the current executable repository state, `answer` uses `gaussian_rag/rag/generation/generator.py` → `ProviderBackedGenerator` backed by env-configured Mistral credentials and refuses to silently downgrade to template output. Demo-only commands stay outside that production-oriented answer path.

### 3.4 Generation Output

The LLM generates an answer that:
- Reflects certainty for high-confidence retrieved chunks
- Hedges or qualifies for low-confidence chunks
- Can abstain or flag when all retrieved chunks are uncertain

---

## Phase 4 — Feedback (Future)

### 4.1 LLM Confidence Signal

After generation, extract LLM confidence as weak signal:

```
logit_entropy = entropy(softmax(output_logits))
self_consistency = variance(sample(LLM, query, n=10))
```

### 4.2 Kalman-Style $\Sigma$ Update

Update $\Sigma_i$ for retrieved chunks based on LLM signal:

$$\Sigma_{i,t+1}^{-1} = \Sigma_{i,t}^{-1} + \lambda \cdot \text{precision}_{\text{LLM}}$$

Risk: confirmation bias. Use only with human verification or multi-agent cross-check.

---

## Full Data Flow

```
Input Document
      │
      ▼
Chunker (128 tok, 16 overlap)
      │
      ▼
  Encoder → μ_i ∈ ℝ^768
      │
      ├──→  Augmenter (K=8 views)
      │            │
      │            ▼
      │     Empirical Cov → Σ_i (diag + low-rank)
      │
      ▼
  GaussianKnowledge { id, text, μ, Σ, confidence }
      │
      ▼
  FAISS (μ) + Flat Store (Σ)
      │
      │  ← Query arrives
      │
      ▼
  ANN on μ → top-100 candidates
      │
      ▼
Exact ELK overlap ranking → top-5, with W2/FR retained as alternate geometry probes
      │
      ▼
  Diversity filter → final retrieved set
      │
      ▼
  Uncertainty-aware context builder
      │
      ▼
  LLM (Claude / GPT / Llama)
      │
      ▼
  Calibrated Answer
```

---

## Configuration

```yaml
# config.yaml
encoder:
  embedding_backend: "gemini_cloud"
  dimension: 384
  sigma_mode: "learned"
  low_rank: 8

chunking:
  max_tokens: 128
  overlap_tokens: 16

retrieval:
  ann_candidates: 10
  top_k: 5
  metric: "elk"  # or "wasserstein2" / "fisher_rao"
  diversity_threshold: 0.3

generation:
  mode: "provider"
  provider: "mistral"
  model: "mistral-small-latest"
  timeout_seconds: 30.0
```

The current answer path supports env-backed Mistral generation. FAISS-based ANN indexing and BEIR-scale evaluation remain planned future work.
