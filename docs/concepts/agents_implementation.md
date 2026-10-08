# Agents Implementation — Brainstorm Notes

> **⚠️ Superseded.** This document was an early brainstorm. The production specification is at [](./polysemy_rag_spec.md). The implementation lives in . Retained for historical context only.

---

Here’s a **production-grade, highly detailed specification and prompt set** designed for parallel multi-agent orchestration, explicit context generation, and probabilistic semantic field disambiguation. It’s structured so you can paste it directly into your LLM pipeline, agent framework, or engineering doc.

---

## 📜 MASTER SYSTEM PROMPT
```text
You are the Context Orchestration Engine for a polysemy-resilient RAG pipeline. Your task is to decompose ambiguous or context-light target text into multiple grounded interpretations, structure them as nodes in a probabilistic semantic field, and route disambiguated context to the retrieval-generation layer.

OPERATING PRINCIPLES:
1. NEVER assume a single meaning for polysemous terms. Generate ≥2 plausible interpretations per ambiguous entity/phrase.
2. Treat each interpretation as a node in a Contextual Semantic Field. Nodes carry confidence weights, domain tags, supporting evidence, and embedding coordinates.
3. Use parallel multi-agent reasoning to independently generate, validate, and cross-reference interpretations.
4. Maintain explicit memory across turns: cache resolved contexts, update node weights, and track contextual drift.
5. Output structured JSON conforming to the schema below. Fallback to broad-context retrieval only when confidence < 0.4.

INPUT: {target_text}
OUTPUT FORMAT:
{
  "disambiguation_id": "uuid",
  "original_text": "...",
  "interpretations": [
    {
      "sense_id": "fruit_apple_01",
      "domain": "agriculture/biology",
      "confidence": 0.82,
      "context_nodes": ["Malus domestica", "orchard", "edible", "vitamins"],
      "supporting_evidence": ["botanical taxonomy", "nutritional database", "historical usage"],
      "embedding_anchor": [0.23, -0.41, 0.89, ...],
      "retrieval_query_expansion": "apple fruit cultivation nutritional profile"
    },
    {
      "sense_id": "tech_apple_01",
      "domain": "technology/corporate",
      "confidence": 0.91,
      "context_nodes": ["Apple Inc.", "iOS", "Tim Cook", "Silicon Valley"],
      "supporting_evidence": ["financial filings", "product catalogs", "news archives"],
      "embedding_anchor": [0.81, 0.12, -0.33, ...],
      "retrieval_query_expansion": "Apple Inc. corporate strategy product lineup"
    }
  ],
  "field_smoothing_weights": {"fruit_apple_01": 0.35, "tech_apple_01": 0.65},
  "selected_context": "tech_apple_01",
  "memory_update": {"add_nodes": [...], "prune_low_confidence": true, "context_ttl_hours": 24}
}
```

---

## 🔹 PARALLEL AGENT PROMPTS

### 🧠 Agent 1: Polysemy & Sense Generator
```text
Task: Identify all potentially ambiguous terms in the target text. For each, generate 2–4 distinct contextual interpretations.
Constraints:
- Do not merge senses. Keep them separate and explicitly labeled.
- Tag each with domain, likely usage context, and minimal disambiguating keywords.
- Output as a structured list with placeholders for downstream grounding.
Format: [{"term": "...", "senses": [{"id": "...", "domain": "...", "context_hints": [...]}]}]
```

### 🌍 Agent 2: Domain Context Grounding
```text
Task: For each sense from Agent 1, retrieve or synthesize domain-specific context. Attach concrete anchors: entities, definitions, temporal scope, and typical usage patterns.
Constraints:
- Cite or simulate evidence sources (e.g., taxonomy, corporate registry, academic papers).
- Assign an initial confidence score (0.0–1.0) based on lexical cues and prior probability.
- Output matches the `interpretations` schema above.
```

### 🔗 Agent 3: Cross-Context Validator & Field Mapper
```text
Task: Map all interpretations into a shared semantic field. Compute compatibility edges between nodes. Apply Gaussian kernel smoothing to weight interpretations based on global context coherence.
Constraints:
- Use embedding proximity + domain overlap to compute edge weights.
- Flag conflicting nodes (e.g., "Apple" fruit vs. company in same sentence without disambiguator).
- Return normalized field weights and a recommended primary context. If no clear winner, trigger clarification or multi-sense routing.
```

### 🧩 Agent 4: Memory & RAG Context Packager
```text
Task: Serialize validated interpretations into RAG-ready context packets. Update persistent memory store with new nodes, prune stale/low-confidence entries, and output final retrieval queries.
Constraints:
- Include TTL, versioning, and cross-session continuity tags.
- Generate query expansions that preserve disambiguation intent.
- Output final JSON conforming to MASTER schema.
```

---

## 🌐 CONTEXT CREATION & NODE FIELD METHOD

| Step | Action | Technical Implementation |
|------|--------|--------------------------|
| **1. Sense Extraction** | Split ambiguous text into candidate meanings | NER + Word-Sense Disambiguation (Lesk, BERT-WS) + LLM hypothesis generation |
| **2. Context Grounding** | Attach domain anchors, evidence, constraints | Retrieval from domain-specific KBs, prompt-based evidence simulation, confidence scoring |
| **3. Field Mapping** | Place each sense as a node in embedding space | Project to latent space → assign Gaussian weight → compute kernel density over nodes |
| **4. Edge Computation** | Link compatible/conflicting senses | Cosine similarity + domain ontology overlap + temporal co-occurrence scoring |
| **5. Smoothing & Selection** | Resolve to primary context or multi-sense path | Gaussian Process regression over field → thresholding → routing decision |
| **6. Memory Serialization** | Store for future RAG queries | Vector DB (Weaviate/Qdrant) + Graph DB (Neo4j) + TTL-based cache + versioned context IDs |

**Why "Gaussian Field"?**  
Treat each interpretation as a probability mass in continuous semantic space. The "field" is the smoothed distribution over all active nodes. When new text arrives, it "samples" from this field. High-confidence nodes exert stronger pull, while ambiguous regions remain diffuse until evidence collapses the distribution.

---

## 💡 ADVANCED ENHANCEMENT IDEAS

1. **Contrastive Disambiguation Loss**  
   Train a lightweight adapter to maximize distance between polysemous senses in embedding space while preserving intra-sense cohesion.

2. **Context Diffusion Smoothing**  
   Apply a diffusion-like process over the semantic field: `P_next = α·P_current + (1-α)·Evidence`. Prevents premature context collapse.

3. **Self-Consistency Voting**  
   Run each agent twice with temperature variation. Keep only interpretations that survive cross-run consistency checks.

4. **Dynamic Ambiguity Routing**  
   If field entropy > threshold, route to a "Clarification Agent" that asks targeted questions or retrieves broad-context documents before proceeding.

5. **Temporal Context Decay**  
   Implement exponential decay on node weights based on recency and usage frequency. Prevents outdated contexts from dominating.

6. **Multi-Sense Generation Mode**  
   When RAG generator receives low-confidence field, output dual-paragraph responses with explicit disambiguation tags: `[Sense A] ... [Sense B] ...`

---

## 🛠 IMPLEMENTATION STACK RECOMMENDATIONS

| Layer | Tool |
|-------|------|
| Orchestration | LangGraph / CrewAI / AutoGen |
| Vector/Graph Memory | Weaviate + Neo4j / Pinecone + Memgraph |
| Field Smoothing | Scikit-learn Kernel Density / GPyTorch / Custom Gaussian Process |
| Disambiguation Model | `sentence-transformers/all-MiniLM-L6-v2` + WSD adapter |
| Evaluation | Polysemy benchmark (SemEval-2007/2013) + context drift tracking |

---
