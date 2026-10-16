# Pipeline — Gaussian-Clef RAG End-to-End System

> Architectural specification of the 6-stage sub-100ms retrieval engine, dual-store ingestion pipeline, and flexible corpus generation system.

---

## High-Level Architecture Overview

Gaussian-Clef RAG operates on statistical manifolds where every chunk of knowledge is represented as a Gaussian distribution $\mathcal{N}(\mu, \Sigma)$. It accelerates probabilistic retrieval from multi-second prototype latency to **sub-100ms production response times** through non-autoregressive decision intelligence and algebraic kernel acceleration.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       DUAL-STORE INGESTION PIPELINE                         │
│                                                                             │
│  Raw Input (Strings, Markdown, Records, Files, OCR/HTML)                    │
│      │                                                                      │
│      ▼                                                                      │
│  [TextNormalizer] ── HTML strip, OCR line repair, soft-wrap unwrap          │
│      │                                                                      │
│      ▼                                                                      │
│  [CorpusGenerator] ── Adaptive structure detection (laws, Q&A, transcripts) │
│      │                                                                      │
│      ▼                                                                      │
│  [UniversalIngestor] ───────────────────────────────────────────────┐       │
│      │                                                              │       │
│      ▼ (SentenceTransformer)                                        ▼       │
│  N(μ, Σ) Embedder                                           SQLite FTS5     │
│      │                                                              │       │
│      ▼                                                              ▼       │
│  KnowledgeStore                                              LexicalStore   │
│  { N(μ_i, Σ_i) }                                            (passages_fts)  │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                 6-STAGE SUB-100MS RETRIEVAL PIPELINE                        │
│                                                                             │
│  User Query                                                                 │
│      │                                                                      │
│      ▼                                                                      │
│  [S1] Clef Ambiguity Gate (non-autoregressive noul/choice via Clef-Flash)   │
│      │ ── (Unambiguous: bypass multi-sense; Ambiguous: 5-agent resolution)  │
│      ▼                                                                      │
│  [S2] Taxonomy Routing (prunes 60-80% candidate branches in tree)           │
│      │                                                                      │
│      ▼                                                                      │
│  [S3] Hybrid Dual-Filter (SQLite FTS5 BM25 + μ-ANN in <2ms)                 │
│      │ ── (Yields ~30 candidate chunks with guaranteed lexical hits)        │
│      ▼                                                                      │
│  [S4] Clef 4-Question Passage Gate (relevance, contradiction, evidence)     │
│      │ ── (Prunes candidates down to top 5-8 high-signal survivors in <40ms)│
│      ▼                                                                      │
│  [S5] Woodbury ELK Reranking (O(d * r^2) exact Gaussian overlap in <3ms)    │
│      │ ── (Score(Q, D) = N(μ_q; μ_d, Σ_q + Σ_d))                            │
│      ▼                                                                      │
│  [S6] Provider Grounded Generation (Mistral, Gemini, or OpenAI)             │
│      │                                                                      │
│      ▼                                                                      │
│  Calibrated, Uncertainty-Aware Answer                                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Detailed Ingestion Pipeline

### 1. Text Normalization (`TextNormalizer`)
Raw, messy, unformatted text (web clippings, OCR/PDF scans, chat logs) is normalized deterministically:
- **Unicode normalization**: NFKC canonicalization, smart quotes (`“”` $\to$ `"`, `‘’` $\to$ `'`), zero-width space removal.
- **HTML stripping**: Strips tags and unescapes entities (`&nbsp;`, `&amp;`, `&#39;`).
- **OCR hyphenation repair**: Rejoins words broken across line wraps (`(\w+)-\s*\n\s*(\w+)` $\to$ `\1\2`).
- **Soft line-break unwrap**: Stitches hard wraps within sentences into clean prose while preserving genuine paragraph breaks (`\n\n`).
- **Page artifact removal**: Strips running headers, footers, page numbers (`Page \d+ of \d+`).

### 2. Adaptive Structure Detection (`CorpusGenerator`)
Instead of fixed token windows that slice through thoughts, `CorpusGenerator` detects natural document structure:
- **Numbered sections/laws**: Matches `Law \d+`, `Section \d+`, `Rule \d+`.
- **Q&A / FAQ pairs**: Matches `Q:` / `A:` or `Question:` / `Answer:` patterns.
- **Transcripts**: Matches speaker turns `Speaker: ...`.
- **Semantic paragraph chunks**: Falls back to token-bounded paragraph boundaries with sliding overlaps.
- **Content-Addressed Cache (`CorpusCache`)**: SHA-256 hash lookup bypasses duplicate document parsing.

### 3. Lockstep Dual-Store Indexing (`UniversalIngestor`)
Chunks are passed to `UniversalIngestor`, which updates two stores synchronously:
1. **`KnowledgeStore`**: Encodes chunk into Gaussian $\mathcal{N}(\mu, \Sigma)$ where $\Sigma = \text{diag}(v) + LL^T$ ($r = 8$).
2. **`LexicalStore`**: Inserts row into SQLite `passages_fts` virtual table (`chunk_id`, `title`, `text`) using SQLite FTS5 BM25.
3. **`TaxonomyRouter`**: Updates hierarchical agglomerative branch tree over chunk centroids.

---

## Detailed 6-Stage Retrieval Pipeline

### Stage 1: Clef Ambiguity Gate
- Query text is evaluated by Cloudflare Workers AI Clef-Flash via a typed `noul` (yes/no/unknown) question: *"Is this query polysemous or ambiguous?"*
- **Unambiguous query (<40ms)**: Bypasses polysemy generation completely, saving hundreds of tokens and latency.
- **Ambiguous query**: Routes to the 5-agent disambiguation pipeline to produce distinct `SenseNode` representations.

### Stage 2: Taxonomy Routing
- The query is matched against the root nodes of the hierarchical document taxonomy tree.
- Clef `choice` questions select active branches.
- Prunes 60–80%+ of irrelevant document branches, eliminating thousands of candidate matrix operations.

### Stage 3: Hybrid Dual-Filter (SQLite FTS5 + μ-ANN)
- Query keywords and entities are extracted and sanitized for SQLite FTS5 syntax.
- **Lexical path**: BM25 search over `passages_fts` retrieves top lexical matches in <2ms.
- **Dense semantic path**: μ-space ANN index retrieves top semantic matches.
- **Candidate fusion**: Merges unique IDs into a bounded candidate set of ~30 chunks. Ensures rare entities, names (e.g. "Nicolas Fouquet"), and specific terms are never bypassed.

### Stage 4: Clef 4-Question Passage Gate
The 30 candidate passages are evaluated in parallel batches via Clef-Flash on 4 orthogonal decision axes:
1. **Relevance**: Does the passage address the query intent?
2. **Factual Evidence**: Does the passage contain concrete factual grounding?
3. **Contradiction**: Does the passage contradict known context or established facts?
4. **Sufficiency**: Can this passage contribute to answering the query?

Filters out hallucinations and noise, pruning the candidate set from 30 down to **5–8 high-signal survivors in <40ms**.

### Stage 5: Woodbury-Accelerated ELK Reranking
Exact overlap between the query Gaussian $\mathcal{N}(\mu_q, \Sigma_q)$ and candidate Gaussian $\mathcal{N}(\mu_d, \Sigma_d)$ is calculated via the Expected Likelihood Kernel (ELK):

$$\text{Score}(Q, D) = \mathcal{N}(\mu_q; \mu_d, \Sigma_q + \Sigma_d) = \frac{1}{(2\pi)^{d/2} \det(\Sigma_q + \Sigma_d)^{1/2}} \exp\left(-\frac{1}{2} (\mu_q - \mu_d)^T (\Sigma_q + \Sigma_d)^{-1} (\mu_q - \mu_d)\right)$$

#### Inversion Acceleration (Woodbury Identity)
Let $A = \text{diag}(v_q + v_d) \in \mathbb{R}^{d \times d}$ and $U = [L_q, L_d] \in \mathbb{R}^{d \times 2r}$. Inversion runs in $O(d \cdot r^2)$ instead of $O(d^3)$:

$$(A + U U^T)^{-1} = A^{-1} - A^{-1} U (I_{2r} + U^T A^{-1} U)^{-1} U^T A^{-1}$$

#### Determinant Acceleration (Sylvester's Theorem)
$$\det(A + U U^T) = \det(A) \det(I_{2r} + U^T A^{-1} U)$$

Evaluating the $(2r \times 2r)$ core requires only a tiny $(16 \times 16)$ Cholesky factorization, computing exact scores for 30 candidates in **under 3ms**.

### Stage 6: Grounded Provider Generation
The ranked, surviving passages are formatted with confidence indicators (`[HIGH/MEDIUM/LOW]`) based on distributional covariance entropy. Answer generation is synthesized via Mistral, Gemini, or OpenAI clients, adhering to calibrated hedging guidelines.

---

## Telemetry and Stage Timing Budget

| Stage | Component | Latency Target |
|---|---|---|
| **S1** | Clef Ambiguity Gate | 30–40ms |
| **S2** | Taxonomy Routing | 5–10ms |
| **S3** | Hybrid Dual-Filter (SQLite FTS5 + ANN) | <2ms |
| **S4** | Clef 4-Question Passage Gate | 30–40ms |
| **S5** | Woodbury ELK Reranking | <3ms |
| **Total Retrieval Pipeline** | **End-to-End Search** | **<80–95ms** |
| **S6** | LLM Generation (Optional) | Provider-dependent |

Real-time telemetry events are broadcast over Server-Sent Events (SSE) via `dashboard_server.py` and visualized in the React/Vite 3D dashboard.
