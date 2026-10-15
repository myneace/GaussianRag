# Polysemy-Resilient Parallel Multi-Agent RAG — Production Specification

**Version:** 1.0  
**Status:** Draft  
**Relates to:** `gaussian_knowledge_graph.md`, `agents_implementation.md`, `gaussian_rag/core/`, `gaussian_rag/rag/`

---

## 1. Problem Statement

The current GaussianRAG pipeline maps each query to a single `QueryRepresentation` (one `μ`, one `Σ`). When the query or document contains **polysemous entities** (e.g., "Apple", "Law", "trial"), the retrieval collapses ambiguity prematurely — before the manifold ranker can exploit covariance-based discrimination. This causes:

- **Sense Conflation**: The embedding mean `μ` lands between senses, matching neither well.
- **False Geodesics**: Wasserstein-2 finds low-distance candidates across incompatible semantic domains.
- **Context Drift**: Multi-hop reasoning in the GSKG propagates the wrong distribution forward.

---

## 2. Architecture Overview

```
Input Text
    │
    ▼
┌─────────────────────────────────────────────────────┐
│          PARALLEL AGENT DISPATCH LAYER              │
│  ┌──────────────┐                                    │
│  │  Agent 0     │                                    │
│  │  Query       │                                    │
│  │  Dissolver   │                                    │
│  └──────┬───────┘                                    │
│         ▼                                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────┐  │
│  │  Agent 1     │  │  Agent 2     │  │ Agent 3  │  │
│  │ Sense        │  │ Domain       │  │ Field    │  │
│  │ Generator    │  │ Grounder     │  │ Mapper   │  │
│  └──────┬───────┘  └──────┬───────┘  └────┬─────┘  │
│         └────────────┬────┘               │        │
│                      ▼                    │        │
│              ┌───────────────┐            │        │
│              │  Agent 4      │◄───────────┘        │
│              │  Memory &     │                     │
│              │  RAG Packager │                     │
│              └───────┬───────┘                     │
└──────────────────────┼──────────────────────────── ┘
                       ▼
          ┌────────────────────────┐
          │  Disambiguation Router │
          │  (entropy gate)        │
          └────────────────────────┘
                  │           │
          confident      ambiguous
                  │           │
                  ▼           ▼
           Ranked         Clarification
           GSKG           Agent (fallback)
           retrieval
```

---

## 3. Data Schemas

### 3.1 `SenseNode` — per-interpretation unit

```python
@dataclass
class SenseNode:
    sense_id: str              # e.g. "tech_apple_01"
    term: str                  # original ambiguous term
    domain: str                # e.g. "technology/corporate"
    confidence: float          # 0.0–1.0, Bayesian prior × evidence
    context_hints: list[str]   # discriminative tokens
    supporting_evidence: list[str]
    mu_anchor: np.ndarray      # D-dim embedding mean for this sense
    sigma_anchor: np.ndarray   # D-dim diagonal covariance for this sense
    sigma_L_anchor: np.ndarray | None
    retrieval_query: str       # expanded query string for this sense
    ttl_hours: int = 24
    version: int = 1
```

### 3.2 `DisambiguationPacket` — master output schema

```python
@dataclass
class DisambiguationPacket:
    disambiguation_id: str     # UUID
    original_text: str
    senses: list[SenseNode]
    field_weights: dict[str, float]   # sense_id → normalized weight
    selected_sense_id: str | None     # None = multi-sense routing
    entropy: float                    # H(field_weights) — routing signal
    memory_ops: MemoryOps
    fallback_triggered: bool = False
```

### 3.3 `MemoryOps`

```python
@dataclass
class MemoryOps:
    upsert_nodes: list[SenseNode]
    prune_ids: list[str]        # sense_ids to expire
    context_window: list[str]   # active sense_ids from previous turns
```

---

## 4. Agent Prompts (Production-Ready)

### Agent 1 — Polysemy & Sense Generator

```
ROLE: Linguistic sense analyst operating within a Gaussian RAG pipeline.

TASK:
Given the input text, identify every term that could have ≥2 distinct meanings
in plausible downstream retrieval contexts. For each term, produce 2–4 *mutually
exclusive* senses. Each sense must be differentiable by domain and co-occurrence
vocabulary — do not produce senses that share the same discriminative keywords.

CONSTRAINTS:
- Named entities, common nouns, and nominalized verbs are all candidates.
- Do NOT merge senses. Polysemy is your input; disambiguation is downstream.
- Each sense must carry exactly: id (snake_case, term + index), domain
  (coarse/fine), and context_hints (3–6 discriminative tokens).
- Output ONLY valid JSON — no prose.

INPUT: {target_text}

OUTPUT SCHEMA:
[
  {
    "term": "<ambiguous token or phrase>",
    "senses": [
      {
        "id": "<term>_<domain_abbrev>_<idx>",
        "domain": "<coarse>/<fine>",
        "context_hints": ["<token1>", "<token2>", "..."]
      }
    ]
  }
]
```

---

### Agent 2 — Domain Context Grounder

```
ROLE: Domain evidence synthesizer. You receive a list of senses from the
Sense Generator and enrich each with grounded evidence and a confidence score.

TASK:
For each sense in the input, attach:
1. supporting_evidence: 2–4 concrete sources or evidence types
   (e.g., "SEC 10-K filing", "botanical taxonomy", "legal corpus BM25 hits").
2. confidence: float 0.0–1.0. Base this on:
   - Lexical prior (how frequently is this sense used globally?)
   - Context signal (how many context_hints appear near the term in input text?)
   - Domain specificity (narrow domains = higher base confidence when triggered)
3. mu_anchor: a *description* of where this sense should cluster in embedding
   space (use landmark tokens — the embedding model will project these).
4. retrieval_query: expanded query string optimized for this specific sense.

CONSTRAINTS:
- Confidence scores must sum to ≤ 1.0 *across senses of the same term*.
  Renormalize if needed.
- Do NOT invent facts. If evidence cannot be cited, mark as "simulated_prior".
- Output must be compatible with SenseNode schema.

INPUT: {sense_list_from_agent1}
CONTEXT: {surrounding_document_context}  # 512-token window
```

---

### Agent 3 — Cross-Context Validator & Field Mapper

```
ROLE: Information-geometry field mapper. You receive enriched SenseNodes and
must construct a coherent Contextual Semantic Field over them.

TASK:
1. COMPATIBILITY EDGES: For each pair of senses (across different terms),
   compute a compatibility score in [0, 1]:
   - 1.0 = senses co-occur naturally (e.g., "Apple_tech" + "iOS_mobile")
   - 0.0 = senses are mutually exclusive in this context
   Use context_hints overlap and domain taxonomy distance.

2. CONFLICT FLAGS: If two senses of the SAME term both have confidence > 0.45,
   flag them as CONFLICTED — signal to the router that clarification is needed.

3. FIELD WEIGHTS: Apply Gaussian kernel smoothing over the compatibility graph.
   Each sense's weight = its confidence * average compatibility with co-selected
   senses of other terms. Normalize to sum = 1.0.

4. ENTROPY: Compute Shannon entropy H = -Σ w_i * log(w_i) over field_weights.
   High entropy (> 1.5 bits) → ambiguous → trigger clarification fallback.

5. RECOMMENDATION: If entropy ≤ 1.5, recommend the highest-weight sense as
   selected_sense_id. Otherwise set selected_sense_id = null.

OUTPUT: field_weights dict, conflict_flags list, entropy float, selected_sense_id.
```

---

### Agent 4 — Memory & RAG Context Packager

```
ROLE: Persistent context manager and RAG query constructor.

TASK:
1. MEMORY CHECK: Load active sense_ids from context_window (previous turns).
   If any input SenseNode matches an active session context (same term + domain),
   boost its confidence by 0.15 (capped at 0.95). This is context continuity.

2. UPSERT: Add all SenseNodes with confidence > 0.3 to the memory store.
   Key: sense_id. Value: full SenseNode + timestamp.

3. PRUNE: Expire any stored node where:
   - TTL exceeded (now - created_at > ttl_hours)
   - Confidence < 0.15 after field smoothing
   - Superseded by a higher-confidence node for the same term+domain

4. QUERY CONSTRUCTION: For the selected sense (or top-2 if multi-sense routing):
   - Construct a structured retrieval query embedding this sense's mu_anchor tokens
   - Preserve the sense covariance anchor so ELK can score a full query bubble.

5. OUTPUT: Complete DisambiguationPacket as JSON, ready for the GSKG retrieval
   layer. Include memory_ops with upsert_nodes and prune_ids.

CONSTRAINTS:
- All node IDs must be stable across sessions (hash of term + domain + version).
- Contrastive exclusion is handled by ANN/search filtering, not by injecting literal
  `NOT (...)` text into the retrieval query string.
```

---

## 5. Memory & Context-Handling Strategy

### 5.1 Three-Tier Context Stack

| Tier | Scope | Storage | TTL |
|------|-------|---------|-----|
| **Turn cache** | Single query | In-process dict | 0 (ephemeral) |
| **Session store** | Conversation | Redis / in-memory JSON | 24 h |
| **Long-term KB** | Cross-session | GSKG (Weaviate + Neo4j) | Permanent |

### 5.2 Context Resolution Order

When a new query arrives:
1. Check **turn cache** — exact sense_id match → reuse immediately.
2. Check **session store** — same term in context_window → boost + reuse.
3. Check **GSKG nodes** — Wasserstein-2 distance from query μ to stored node μ < θ_memory (default 0.3) → inherit domain context.
4. If no match at any tier → full 5-agent disambiguation pipeline.

### 5.3 Covariance-Based Disambiguation in GSKG

Each stored `SenseNode` maps directly to a `GaussianKnowledge` node:

```python
def sense_to_gaussian(sense: SenseNode, embedder: GaussianEmbedder) -> GaussianKnowledge:
    mu, sigma_diag, sigma_L = embedder.embed(sense.retrieval_query)
    # Narrow Σ for high-confidence (specific) senses
    scale = 1.0 / (1.0 + sense.confidence)
    return GaussianKnowledge(
        id=sense.sense_id,
        text=sense.retrieval_query,
        mu=mu,
        sigma_diag=sigma_diag * scale,
        sigma_L=sigma_L * scale,
        metadata={"domain": sense.domain, "term": sense.term}
    )
```

High-confidence senses → small `Σ` → tight Wasserstein ball → precise retrieval.  
Low-confidence / broad senses → large `Σ` → diffuse field → wide retrieval net.

---

## 6. Disambiguation Workflow & Fallback Logic

```
Query arrives
    │
    ▼
[Tier 1–3 cache check] ──── HIT ──────────────────────► Inject cached context
    │ MISS                                                       │
    ▼                                                            │
[Agent 0 dissolves query, then Agents 1, 2, 3 refine in parallel] │
    │                                                            │
    ▼                                                            │
[Agent 4 merges + computes entropy]                             │
    │                                                            │
    ├── entropy ≤ 1.5 bits ──────────────────────────────────────┤
    │   → single-sense routing                                   │
    │   → inject selected SenseNode as QueryRepresentation       │
    │                                                            ▼
    ├── entropy 1.5–2.5 bits                              [manifold_ranker.py
    │   → dual-sense routing                               rerank_candidates()]
    │   → query GSKG with top-2 senses in parallel              │
    │   → merge results (interleave by rank, deduplicate)        │
    │                                                            ▼
    └── entropy > 2.5 bits                             [Generator with
        → CLARIFICATION FALLBACK                        injected context]
        → spawn ClarificationAgent (see §6.1)
        → if no resolution after 1 turn → broad retrieval
          with multi-sense output tags [Sense A] / [Sense B]
```

### 6.1 Clarification Agent Prompt

```
ROLE: Disambiguation clarifier.

The retrieval system detected high ambiguity (entropy={entropy:.2f} bits) 
for the query: "{original_text}"

Competing interpretations:
{for each top-3 sense: "- [{sense.domain}]: {sense.retrieval_query} (confidence={sense.confidence:.2f})"}

Ask the user ONE targeted question to resolve the ambiguity. The question must:
- Reference specific domain signals from the text
- Offer concrete choices, not open-ended responses
- Be phrased to take ≤10 seconds to answer

If this is an automated pipeline (no user present), select the highest-confidence 
sense and set fallback_triggered=true in the output packet.
```

### 6.2 Fallback Chain

```
entropy > 2.5 AND no user response available
    │
    ├─ Step 1: Force-select highest-confidence sense (fallback_triggered=True)
    ├─ Step 2: Expand retrieval_query with all context_hints from top-2 senses
    ├─ Step 3: Retrieve top-10 (not top-5) to increase recall
    └─ Step 4: Generator receives both sense contexts in prompt, produces
               dual-paragraph output tagged [Domain A] / [Domain B]
```

---

## 7. Integration with Existing Codebase

### 7.1 `manifold_ranker.py` — Multi-Sense Extension

Add a `rerank_multi_sense()` function:

```python
def rerank_multi_sense(
    senses: list[QueryRepresentation],
    candidates: list[GaussianKnowledge],
    *,
    metric: str = "elk",
    fusion: str = "interleave",  # or "weighted_sum"
) -> list[RetrievedChunk]:
    """
    Runs rerank_candidates() for each sense independently,
    then fuses results. 'interleave' alternates top results from each
    sense list; 'weighted_sum' scores = Σ(sense_weight * rank_score).
    """
    per_sense_results = [
        (sense, rerank_candidates(sense, candidates, metric=metric))
        for sense in senses
    ]
    return _fuse_results(per_sense_results, strategy=fusion)
```

### 7.2 Session Context Layer

Current implementation uses `gaussian_rag/rag/sense_cache.py` as the session context layer alongside `gaussian_rag/core/store.py` for persisted knowledge:

```python
class SenseCache:
    def get_by_term(self, term: str) -> list[SenseNode]: ...
    def upsert(self, nodes: list[SenseNode]) -> None: ...
    def prune_expired(self) -> int: ...
```

### 7.3 `core/types.py` — Add `SenseNode` & `DisambiguationPacket`

Add the dataclasses from §3 to `types.py`. Both are pure Python dataclasses with no new dependencies.

---

## 8. Evaluation Criteria

### 8.1 Disambiguation Accuracy

| Metric | Target | Measurement |
|--------|--------|-------------|
| Sense selection accuracy (polysemous queries) | ≥ 85% | SemEval-2013 WSD benchmark |
| False disambiguation rate (unambiguous queries wrongly split) | ≤ 5% | Held-out unambiguous set |
| Context continuity (correct sense reuse across turns) | ≥ 90% | Multi-turn session tests |

### 8.2 Retrieval Quality

| Metric | Target | Measurement |
|--------|--------|-------------|
| Precision@5 on ambiguous queries vs. point-embedding baseline | +15% relative | Custom polysemy retrieval benchmark |
| Wasserstein distance — correct-sense docs vs. incorrect-sense docs | > 0.5 separation | Per-sense retrieval set |
| Entropy calibration (predicted entropy vs. actual answer variance) | Spearman ρ ≥ 0.7 | Manual annotation of 200 queries |

### 8.3 Context Robustness

| Test | Pass Condition |
|------|----------------|
| Same query, opposite context prefix | Retrieves different top sense each time |
| Session context injection | Turn 3 inherits Turn 1 sense without re-running full pipeline |
| High-entropy fallback | Dual-paragraph output produced; no single wrong sense selected |
| Context TTL expiry | Expired nodes not reused; pipeline re-runs fresh disambiguation |

### 8.4 Latency Budget (production targets)

| Stage | P50 | P95 |
|-------|-----|-----|
| Agent 1–3 parallel dispatch | 800 ms | 1.8 s |
| Agent 4 memory + packaging | 200 ms | 500 ms |
| GSKG retrieval (single sense) | 50 ms | 150 ms |
| GSKG retrieval (multi-sense) | 100 ms | 300 ms |
| Total end-to-end | 1.2 s | 2.8 s |

---

## 9. Open Issues / Decisions Required

1. **Embedding model**: `all-MiniLM-L6-v2` (fast, 384-dim) vs. `text-embedding-3-small` (1536-dim, higher fidelity). The sense `μ_anchor` quality depends on this.
2. **Session store backend**: In-memory dict (current `store.py` pattern) is sufficient for MVP; Redis needed for multi-process / containerized deployment.
3. **Clarification agent UX**: Automated pipelines must set `fallback_triggered=True` — need to decide whether to surface this flag in generation prompt or silently absorb.
4. **GSKG edge threshold (θ_memory)**: W2 distance threshold for "same sense" cache hit. Needs calibration on real polysemous query set.
5. **Contrastive negative hints**: Not supported by current `ann_index.py` (no NOT filter). Would require upgrading to FAISS with metadata filtering or Weaviate.
