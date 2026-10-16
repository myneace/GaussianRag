# Concept Specifications — `docs/concepts/`

This directory contains architecture specifications, implementation blueprints, and design notes for Gaussian-Clef RAG sub-systems.

---

## Documents

| File | Status | Description |
|---|---|---|
| [polysemy_rag_spec.md](./polysemy_rag_spec.md) | ✅ Production spec | Full specification for the polysemy-resilient parallel multi-agent RAG pipeline. |
| [agent_dissolver.md](./agent_dissolver.md) | ✅ Implemented | High-resolution semantic ingestion engine. Handles adaptive expansion, ambiguity scoring, and manifold filament generation. |
| [multi_agent_disambiguation.md](./multi_agent_disambiguation.md) | ✅ Implemented | Details on the parallel semantic inference and routing pipeline. |
| [polysemy_rag_gap_analysis.md](./polysemy_rag_gap_analysis.md) | 📋 Historical snapshot | Early implementation gap-analysis retained for provenance; superseded by modern test suite. |
| [gaussian_knowledge_graph.md](./gaussian_knowledge_graph.md) | 📐 Concept note | Describes Gaussian Semantic Knowledge Graphs (GSKG). |
| [agents_implementation.md](./agents_implementation.md) | 📋 Brainstorm (superseded) | Original raw brainstorm for the multi-agent orchestration concept. Superseded by `polysemy_rag_spec.md`. Retained for historical context. |

---

## Cross-References

- **Documentation Hub:** See [docs/README.md](../README.md) for the unified documentation map.
- **Root README:** See [README.md](../../README.md) for the project quick start, library facade, and 3D dashboard.
- **Implementation:**
  - Fast Clef Disambiguation: `gaussian_rag/rag/clef_client.py`
  - 5-Agent Pipeline: `gaussian_rag/rag/disambiguation/`
  - Dual-Store & Taxonomy: `gaussian_rag/core/lexical_store.py` and `gaussian_rag/rag/retrieval/taxonomy_router.py`
  - Woodbury ELK Reranker: `gaussian_rag/rag/retrieval/manifold_ranker.py` and `gaussian_rag/core/covariance.py`
- **Theory:** [THEORY.md](../initial/THEORY.md) covers the information geometry foundations used by all agents.
- **Pipeline:** [PIPELINE.md](../initial/PIPELINE.md) describes where disambiguation fits in the 6-stage sub-100ms pipeline.
