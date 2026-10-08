# Historical Gap Analysis — Polysemy RAG Spec vs. Early Existing Codebase

This document records the early implementation-gap snapshot that guided the first build-out of the polysemy pipeline. It is retained for provenance and migration history; for current runtime behavior, treat the executable code plus `docs/initial/*` and `docs/concepts/polysemy_rag_spec.md` as canonical.

---

## Legend

| Symbol | Meaning |
|--------|---------|
| ✅ | Already exists — use as-is |
| 🔧 | Exists but needs extension / modification |
| 🆕 | Does not exist — must be created |

---

## Component Map

### 1. Data Types (`gaussian_rag/core/types.py`)

| Spec Type | Status | Notes |
|-----------|--------|-------|
| `GaussianKnowledge` (μ, Σ, id, text, metadata) | ✅ | Exact match. `sigma_diag + sigma_L` low-rank form already matches `SenseNode.sigma_anchor` needs. |
| `QueryRepresentation` (μ, optional Σ) | ✅ | One `QueryRepresentation` per sense — already the right shape. |
| `RetrievedChunk` (w2_distance, fr_distance, confidence, rank) | ✅ | Present; runtime now also carries `elk_score` for ELK-first retrieval. |
| `SenseNode` (sense_id, domain, confidence, mu_anchor, retrieval_query, ttl) | ✅ | Implemented in `types.py`, now including covariance anchors. |
| `DisambiguationPacket` (senses, field_weights, entropy, selected_sense_id) | ✅ | Implemented in `types.py`. |
| `MemoryOps` (upsert_nodes, prune_ids, context_window) | ✅ | Implemented in `types.py`. |

**Action:** Keep the implemented dataclasses aligned with the evolving spec; current gaps are documentation-level rather than missing types.

---

### 2. Embedding (`gaussian_rag/core/gaussian_embedding.py`)

| Spec Need | Status | Notes |
|-----------|--------|-------|
| Embed a sense's `retrieval_query` string → `GaussianKnowledge` | ✅ | `GaussianEmbedder.encode(text, item_id=sense_id)` does this exactly. |
| Scale Σ by confidence (`scale = 1/(1+confidence)`) | 🔧 | `encode()` ignores confidence. Need a wrapper `encode_sense(sense: SenseNode)` that applies the scale to `sigma_diag` and `sigma_L` after encoding. |
| Encode per-sense query with covariance | ✅ | `encode_query(text, with_covariance=True)` already works. |
| `HeuristicSigmaEstimator` — broader Σ for vague text | ✅ | Works correctly — short/vague sense text naturally produces wider covariance. |

**Action:** Add `encode_sense(sense: SenseNode) -> GaussianKnowledge` to `GaussianEmbedder` — 10 lines, no breaking changes.

---

### 3. Information Geometry (`gaussian_rag/core/information_geometry.py`)

| Spec Need | Status | Notes |
|-----------|--------|-------|
| Wasserstein-2 between senses (disambiguation distance) | ✅ | `wasserstein2(g1, g2)` — exact match. Used for sense-pair compatibility edge computation. |
| Fisher-Rao (path uncertainty, multi-hop) | ✅ | `fisher_rao(g1, g2)` — exact match. Used to penalize high-variance traversal. |
| KL-divergence (confidence calibration) | ✅ | `kl_divergence()` — available. Agent 3 can use this for compatibility scoring. |
| Bhattacharyya (sense overlap detection) | ✅ | `bhattacharyya()` — available. Useful for CONFLICTED sense detection. |
| Shannon entropy over field weights | 🆕 | Pure Python/NumPy — `H = -sum(w * log(w))`. Add as `field_entropy(weights: dict) -> float` in `gaussian_rag/rag/disambiguation/field_mapper.py`. |

**Action:** `information_geometry.py` needs **no changes**. All four metrics are already implemented.

---

### 4. Knowledge Store (`gaussian_rag/core/store.py` + `gaussian_rag/rag/sense_cache.py`)

| Spec Need | Status | Notes |
|-----------|--------|-------|
| Store `GaussianKnowledge` items by ID | ✅ | `KnowledgeStore.add()` / `.get()` |
| Persist to disk (JSON) | ✅ | `save()` / `load()` |
| Session context: store `SenseNode`s with TTL | 🆕 | `KnowledgeStore` only handles `GaussianKnowledge`. Need a parallel `SenseCache` class. |
| Prune expired nodes by TTL | 🆕 | No expiry logic anywhere in the store. |
| Upsert by sense_id (merge, not duplicate) | 🔧 | `add()` already overwrites by ID — so upsert is implicit. Just need stable ID generation (`hash(term + domain + version)`). |
| Context window (active sense_ids from prev turns) | 🆕 | No session concept exists. New `SenseCache` handles this. |

**Action:** Maintain the existing split between `gaussian_rag/core/store.py` and `gaussian_rag/rag/sense_cache.py`.

```
SenseCache:
  - items: dict[str, SenseNode]          ← in-memory session store
  - upsert(nodes: list[SenseNode])       ← add/overwrite by sense_id
  - get_by_term(term: str) → list[SenseNode]
  - prune_expired() → int               ← removes nodes past TTL
  - context_window() → list[str]        ← returns all active sense_ids
  - save/load (JSON, same pattern as KnowledgeStore)
```

---

### 5. ANN Index (`gaussian_rag/rag/retrieval/ann_index.py`)

| Spec Need | Status | Notes |
|-----------|--------|-------|
| Search candidates by μ (cosine/L2) | ✅ | `MuAnnIndex.search(mu, top_k)` — NumPy linear scan. |
| Per-sense candidate search (run once per sense) | ✅ | Call `.search()` once per `QueryRepresentation.mu` — already works. |
| Contrastive NOT-filter (exclude tokens of competing senses) | 🆕 | `MuAnnIndex` has no concept of exclusion. **This requires ANN with metadata filtering** (FAISS IDSelector or Weaviate `where` filter). For MVP: implement soft exclusion — post-filter results whose text has high overlap with negative hint tokens. |

**Action (MVP):** Add `search_excluding(mu, top_k, exclude_ids: list[str])` to `MuAnnIndex` — filters out specific chunk IDs from results. Full NOT-filter (semantic) is a v2 concern.

---

### 6. Manifold Ranker (`gaussian_rag/rag/retrieval/manifold_ranker.py`)

| Spec Need | Status | Notes |
|-----------|--------|-------|
| Single-sense reranking | ✅ | `rerank_candidates(query, candidates, metric=)` |
| Multi-sense parallel reranking + fusion | ✅ | `rerank_multi_sense()` exists in `manifold_ranker.py`. |
| Interleave fusion (alternate top results per sense) | ✅ | Implemented and used by the live router. |
| Weighted-sum fusion | ✅ | Implemented as alternate fusion mode. |

**Action:** Maintain parity between the implemented fusion modes and the production spec.

---

### 7. Retriever (`gaussian_rag/rag/retrieval/retriever.py`)

| Spec Need | Status | Notes |
|-----------|--------|-------|
| Single-sense retrieval | ✅ | `GaussianRetriever.retrieve()` |
| Diversity filtering | ✅ | `retrieve_with_diversity()` — uses `wasserstein2` between chunks to filter. |
| Multi-sense retrieval (run per-sense, merge) | ✅ | Implemented in `GaussianRetriever`, including MMASA parallel dispatch. |

**Action:** Keep retrieval docs aligned with the current MMASA implementation and ELK-first defaults.

---

### 8. Context Builder (`gaussian_rag/rag/generation/context_builder.py`)

| Spec Need | Status | Notes |
|-----------|--------|-------|
| Build LLM context string from retrieved chunks | ✅ | `build_context(query, retrieved)` |
| Multi-sense output — `[Sense A] / [Sense B]` tags | 🆕 | Single context only. No sense-labeling. |
| Inject active sense domain into prompt | 🆕 | Chunks have no domain metadata surfaced in prompt. |

**Action:** Add `build_multi_sense_context(query, sense_results: dict[str, list[RetrievedChunk]]) -> str` — renders chunks grouped by sense with domain labels.

---

### 9. Disambiguation Layer — Module (`gaussian_rag/rag/disambiguation/`)

This module now exists and corresponds to the 5-agent pipeline. The agents are implemented as **plain Python functions** that call the existing components rather than external orchestration frameworks.

| Component | File | Description |
|-----------|------|-------------|
| Sense generator | `sense_generator.py` | Wraps `GaussianEmbedder.encode_query()` + heuristic token splitting to produce `SenseNode` candidates |
| Domain grounder | `domain_grounder.py` | Scores each `SenseNode` using `wasserstein2` / `bhattacharyya` against a seeded domain-anchor store |
| Field mapper | `field_mapper.py` | Computes compatibility matrix, Gaussian kernel smoothing, Shannon entropy |
| Memory packager | `memory_packager.py` | Wraps `SenseCache` — upsert, prune, TTL boost, contrastive query construction |
| Disambiguation router | `router.py` | Entropy gate: single-sense → `retrieve()`, dual-sense → `retrieve_multi_sense()`, high-entropy → fallback |
| Pipeline entry point | `pipeline.py` | `disambiguate(text, embedder, retriever, cache) -> DisambiguationPacket` |

---

## Complete File Inventory

```
gaussian_rag/
├── core/
│   ├── types.py                ← 🔧 Add SenseNode, DisambiguationPacket, MemoryOps
│   ├── gaussian_embedding.py   ← 🔧 Add encode_sense()
│   ├── information_geometry.py ← ✅ No changes
│   ├── covariance.py           ← ✅ No changes
│
├── knowledge/
│   ├── store.py                ← ✅ No changes
│   ├── sense_cache.py          ← 🆕 NEW: SenseCache (TTL, session context)
│   ├── ingestion.py            ← ✅ No changes
│   ├── chunker.py              ← ✅ No changes
│   ├── drift.py                ← ✅ No changes
│
├── retrieval/
│   ├── ann_index.py            ← 🔧 Add search_excluding()
│   ├── manifold_ranker.py      ← 🔧 Add rerank_multi_sense()
│   ├── retriever.py            ← 🔧 Add retrieve_multi_sense()
│   ├── diversity.py            ← ✅ No changes
│   ├── uncertainty_weighter.py ← ✅ No changes
│
├── generation/
│   ├── context_builder.py      ← 🔧 Add build_multi_sense_context()
│
├── disambiguation/             ← 🆕 NEW MODULE
│   ├── __init__.py
│   ├── sense_generator.py
│   ├── domain_grounder.py
│   ├── field_mapper.py
│   ├── memory_packager.py
│   ├── router.py
│   └── pipeline.py
```

---

## Work Sizing Summary

| Category | Files | Effort |
|----------|-------|--------|
| ✅ Use as-is | `information_geometry.py`, `covariance.py`, `store.py`, `ingestion.py`, `chunker.py`, `drift.py`, `diversity.py`, `uncertainty_weighter.py` | 0 |
| 🔧 Minor extensions (additive) | `types.py`, `gaussian_embedding.py`, `ann_index.py`, `manifold_ranker.py`, `retriever.py`, `context_builder.py` | ~150 lines total |
| 🆕 New code | `sense_cache.py` + all of `gaussian_rag/rag/disambiguation/` (6 files) | ~400 lines total |

**Core insight:** The spec is **very well aligned** with the codebase. The math (`wasserstein2`, `fisher_rao`, `GaussianKnowledge`, `HeuristicSigmaEstimator`) is already correct and production-ready. What's missing is entirely the **disambiguation orchestration layer** on top — the routing logic, sense-level cache, and multi-sense retrieval path.
