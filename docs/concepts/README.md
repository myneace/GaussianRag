# Concept Specifications — `docs/concepts/`

This directory contains architecture specs, implementation blueprints, and design notes for GaussianRAG sub-systems.

---

## Documents

| File | Status | Description |
|---|---|---|
| [polysemy_rag_spec.md](./polysemy_rag_spec.md) | ✅ Production spec | Full specification for the polysemy-resilient parallel multi-agent RAG pipeline. |
| [agent_dissolver.md](./agent_dissolver.md) | ✅ Implemented | High-resolution semantic ingestion engine. Handles adaptive expansion, ambiguity scoring, and manifold filament generation. |
| [multi_agent_disambiguation.md](./multi_agent_disambiguation.md) | ✅ Implemented | Details on the parallel semantic inference and routing pipeline. |
| [polysemy_rag_gap_analysis.md](./polysemy_rag_gap_analysis.md) | 📋 Historical snapshot | Early implementation gap-analysis retained for provenance; not the canonical current-state summary. |
| [gaussian_knowledge_graph.md](./gaussian_knowledge_graph.md) | 📐 Concept note | Describes Gaussian Semantic Knowledge Graphs (GSKG). |
| [agents_implementation.md](./agents_implementation.md) | 📋 Brainstorm (superseded) | Original raw brainstorm for the multi-agent orchestration concept. Superseded by `polysemy_rag_spec.md`. Retained for historical context. |

---

## Cross-References

- **Root docs:** See [README.md](../../README.md) for the full documentation index.
- **Implementation:** `gaussian_rag/rag/disambiguation/` — the executed implementation of `polysemy_rag_spec.md`.
- **Theory:** [THEORY.md](../initial/THEORY.md) covers the information geometry foundations used by all agents.
- **Pipeline:** [PIPELINE.md](../initial/PIPELINE.md) §Phase 0 describes where disambiguation fits in the full pipeline.
