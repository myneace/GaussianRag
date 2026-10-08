# Agent Dissolver: High-Resolution Semantic Ingestion

The **Agent Dissolver** is the core ingestion engine of the Manifold RAG system. It evolves standard document chunking from a "point-based" approach into a "field-based" approach by decomposing text into an exhaustive set of domain-specific interpretations.

## 1. The Core Philosophy: "Expansion, Not Compression"

Standard RAG systems attempt to compress a text chunk into a single embedding vector (a point). This loses semantic nuance, especially for ambiguous or technical text. 

The Agent Dissolver does the opposite: it **dissolves** the chunk into its constituent semantic dimensions.

- **Anchor Node**: Represents the "center of mass" of the chunk.
- **Interpretation Nodes**: High-precision points representing specific domain-specific views (legal, clinical, molecular, etc.).
- **Manifold Filaments**: The logical links that bind interpretations to their parent anchors.

## 2. Adaptive Semantic Ingestion

Manifold does not use fixed-count expansion. It uses an **LLM-driven Ambiguity Score** (1-10) to determine how many interpretations are required.

| Ambiguity Score | Logic | Resolution |
| :--- | :--- | :--- |
| **1-2 (Literal)** | Simple facts, direct statements. | Low (1-2 nodes) |
| **3-6 (Nuanced)** | Context-dependent technical info. | Medium (3-6 nodes) |
| **7-10 (Polysemous)** | High metaphor, legal ambiguity, medical complexity. | High (8-12+ nodes) |

## 3. Mathematical Foundation: Dissolver Sigma Estimation

The "Uncertainty" ($\Sigma$) of an anchor node is not guessed; it is **measured**.

$$ \Sigma_{diag} = \text{Var}(\text{Embeddings}(\text{Interpretations})) $$

The system calculates the sample variance across all generated interpretations. This means:
- If interpretations are scattered (highly ambiguous), the Anchor's Gaussian field is **large and fuzzy**.
- If interpretations are tight (low ambiguity), the Anchor's Gaussian field is **small and sharp**.

## 4. Manifold Filaments & Topology

In the Manifold Explorer, you can visualize the filaments connecting interpretations to their hubs. This topology allows the retriever to:
1.  **Pinpoint**: Find a specific interpretation that matches the query's domain.
2.  **Generalize**: Use the Anchor's broad field if the query itself is ambiguous.

## 5. Validation Metrics

We validate the effectiveness of the Dissolver using three primary metrics:

1.  **Semantic Distinctness**: Average W2 distance between interpretations. We target $> 0.5$ to ensure no redundancy.
2.  **Sharpening Gain**: The distance ratio between the best interpretation and the anchor. A high gain proves the manifold is removing "semantic noise."
3.  **W2-Sensitivity**: Measuring how much the Covariance ($\Sigma$) contributes to retrieval ranking.

## 6. Usage

To run the high-resolution ingestion:

```bash
python3 gaussian_rag/main.py ingest --input ./docs --output ./knowledge_store
```

To validate the manifold density:

```bash
python3 scripts/validate_manifold.py --store ./knowledge_store
```
