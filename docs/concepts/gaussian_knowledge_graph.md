# Gaussian Semantic Knowledge Graphs (GSKG)

## The Core Concept
Standard Graph RAG relies on nodes and edges represented as single-point embeddings in a vector space. A **Gaussian Semantic Knowledge Graph (GSKG)** elevates this by treating every node not as a point, but as a **probability distribution on an information manifold**.

By integrating our existing Gaussian RAG (`GaussianEmbedder`, Wasserstein-2/Fisher-Rao metrics) into a graph structure, we can map both the semantic *meaning* and the semantic *uncertainty* of entities and relationships.

## Key Architectural Advantages

### 1. Nodes as Distributions (Semantic Breadth)
In a standard KG, the node for "Law" and the node for "Rule 11 Sanctions" are just points. In a GSKG:
*   **"Rule 11 Sanctions"** has a tightly clustered covariance matrix ($\Sigma$) because it is highly specific.
*   **"Law"** has a massive covariance matrix ($\Sigma$) because it is a broad, ambiguous concept that encompasses many sub-topics.
*   **Benefit**: The graph naturally understands hierarchy and specificity without needing explicit "is-a" ontology rules. Broad nodes envelop specific nodes in the manifold.

### 2. Edges as Geodesics (Optimal Transport)
Instead of static scalar weights for edges, the "distance" between nodes is defined by information geometry (e.g., Wasserstein-2).
*   **Semantic Collisions Avoided**: Two nodes might have similar text (e.g., "Apple" the fruit vs. "Apple" the company), putting their means ($\mu$) close together. But their variances ($\Sigma$) will differ drastically based on context. The Wasserstein distance prevents false edge traversal by penalizing variance mismatch.

### 3. Multi-Hop Uncertainty Propagation
A major flaw in standard Graph RAG is that multi-hop reasoning (A -> B -> C) often leads to confident hallucinations because the system loses context along the path.
*   In a GSKG, as you traverse from node A to node B, you are navigating a statistical manifold. If node B is highly ambiguous (large $\Sigma$), traversing *through* it naturally expands the uncertainty of the subsequent hop.
*   **Benefit**: The final retrieved answer has a mathematically rigorous **Confidence Score** based on the compounded uncertainty of the path taken.

### 4. Dynamic Ambiguity Resolution
When a user asks an ambiguous query (e.g., "What are the outcomes of the trial?"), the query itself is mapped as a high-variance Gaussian. 
*   The graph traversal engine can specifically look for sub-graphs (clusters of nodes) whose collective distribution matches the query's distribution.
*   It inherently prefers to retrieve a diverse set of specific nodes that collectively "fill" the query's broad covariance, rather than just fetching the single closest point.

## Implementation Path
To build this on our current stack:
1.  **Node Extraction**: Standard NER to extract entities.
2.  **Node Embedding**: Use `GaussianEmbedder` to generate $\mu$ and $\Sigma$ for each entity based on its surrounding document context.
3.  **Edge Construction**: Compute `wasserstein2()` between all entity pairs in a document. Edges are formed if the manifold distance is below a threshold.
4.  **Graph Retrieval**: Modify `manifold_ranker.py` to support multi-hop retrieval, penalizing paths where the Fisher-Rao distance is too high.
