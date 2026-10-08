# About the Learned Covariance Model (PyTorch SigmaHead)

This document captures common questions and answers about the PyTorch `SigmaHead` model integrated into GaussianRAG as learned covariance estimation.

---

## Q: What happens after training finishes?

When training completes, the `SigmaHead` neural network has "learned" how to predict covariance matrices — the uncertainty shapes — that support the Gaussian query/document overlap geometry used by the live system, with Wasserstein-2 retained as an alternate geometric probe.

Three things need to happen to put that knowledge to work:

1. **Save the weights** — The trained PyTorch weights are serialized to `.gaussian_rag_weights_{dimension}.pt` so they match the active embedding backend dimension.
2. **Re-ingest the knowledge base** — Old chunks stored in `knowledge.json` used a heuristic covariance. After training, upload your documents again so each chunk is re-embedded using the newly trained `SigmaHead`.
3. **Switch the config** — `config/config.yaml` must have `encoder.sigma_mode: "learned"` so that every query and ingestion request goes through the PyTorch model.

---

## Q: Are the weights used at both ingestion and retrieval?

**Yes.** This is critical for correctness. The PyTorch weights are used in both places to keep the geometry aligned:

- **Ingestion:** Every extracted proposition is passed through the `SigmaHead` to compute its covariance bubble. That shape is permanently saved to `knowledge.json`.
- **Retrieval (Chat):** When you type a query, the exact same `SigmaHead` computes an uncertainty bubble for that query in real time.

Because both the stored facts *and* your live query are shaped by identical weights, the live retrieval metric can compare the same family of Gaussian bubbles consistently. If we only used the weights at one stage, the geometric comparison would be meaningless.

---

## Q: So are we basically training an embedding model?

Yes and no. We are **not** training the entire embedding model from scratch — that would require massive compute (thousands of GPUs, petabytes of data).

Instead, we:
- **Freeze** the base `all-MiniLM-L6-v2` transformer, which computes the *meaning* (the center point / Mean `mu`).
- **Train** only the custom `SigmaHead` network, which sits on top and learns to predict *uncertainty* (the Covariance `sigma`).

This upgrades a standard flat vector embedding model into a **Gaussian distribution** — a point with a meaningful uncertainty bubble — without modifying the underlying transformer at all.

---

## Q: How did we train it without any labelled data?

We used **Self-Supervised Contrastive Learning**. No human ever manually labels "this sentence has uncertainty 0.5" — that would be impossible to do at scale.

Instead, the training loop generates its own labels automatically:

1. **Augmentation (Creating Positive Pairs):** Each sentence is algorithmically shuffled or reworded to create a "noisy view" of the same concept. For example:
   - Original: *"Mitral valve regurgitation describes backward blood flow."*
   - Augmented: *"blood flow backward describes regurgitation Mitral valve."*

2. **Contrastive Loss:** The model is given two objectives:
   - **Pull together** the original and its augmented view — they represent the same concept, so the `SigmaHead` learns to produce a large uncertainty bubble to cover both.
   - **Push apart** completely unrelated sentences (e.g., a medical fact vs. a legal fact) using Wasserstein-2 geometry.

3. **Result:** Over 10 epochs, the network naturally learns how ambiguity behaves in language — no human labels required!

---

## Q: Where is the training data?

The training corpus is stored in [`data/training/robust_corpus.jsonl`](../data/training/robust_corpus.jsonl).

### Robust Data Structure (JSONL)
To make the model robust, we use a **JSON Lines (JSONL)** format. Each line is a self-contained JSON object:

```json
{"text": "Specific fact here", "domain": "medical"}
{"text": "Ambiguous query here", "domain": "general"}
```

This structure is superior to a simple list because:
1. **Scalability:** You can stream thousands of examples from disk without loading them all into memory.
2. **Domain Awareness:** The model can use the `domain` field to ensure it is contrasting truly different concepts.
3. **Metadata Support:** You can add fields like `entropy` or `source` to further guide the learning process.

To train on your own data, simply add your own JSON objects to `data/training/robust_corpus.jsonl` and re-run:

```bash
uv run python run_training.py
```

The system strictly uses the JSONL format for robustness. Legacy JSON lists are no longer supported.

---

## Q: Does the model get smarter over time?

**Yes!** We've implemented a **Continual Learning** pipeline based on "Query Radiation".

Every time you interact with the system, it naturally produces new pairs of `{text, domain}`:
- **Your Queries:** `{"text": "What is X?", "domain": "query"}`
- **Dissolver Senses:** `{"text": "Domain-specific interpretation", "domain": "tech"}`
- **Ingested Nodes:** `{"text": "Factual knowledge", "domain": "anchor"}`

The `OnlineRadiationCollector` now records a structured radiation event and appends derived training samples to `robust_corpus.jsonl`.

Once enough new samples accumulate (default: 5), a **background daemon thread** wakes up, fine-tunes the `SigmaHead` on the full expanded corpus for 2 epochs, and automatically swaps the weights on disk. Your very next query will instantly use the smarter, refined model without any server restarts.

You can monitor the absorption statistics and training runs in the dashboard logs, or via the `/api/radiation/stats` API endpoint.
---

## Architecture Diagram

```
Your Text
    │
    ▼
┌──────────────────────────────────┐
│  all-MiniLM-L6-v2 (FROZEN)      │  ← Standard pre-trained transformer
│  Computes: mu (Mean / Center)    │
└──────────────────────┬───────────┘
                       │  hidden_state (384-dim vector)
                       ▼
┌──────────────────────────────────┐
│  SigmaHead (TRAINED via PyTorch) │  ← Our custom neural network
│  Computes: sigma (Covariance)    │
│  Diagonal + Low-Rank components  │
└──────────────────────┬───────────┘
                       │
                       ▼
            Gaussian Distribution
            N(mu, Sigma)  — a point
            with an uncertainty bubble
            ready for Wasserstein-2 geometry
```

---

## Key Files

| File | Purpose |
|---|---|
| `gaussian_rag/training/sigma_head.py` | PyTorch `nn.Module` that predicts covariance |
| `gaussian_rag/training/losses.py` | Differentiable Wasserstein-2 + contrastive loss |
| `gaussian_rag/training/trainer.py` | Self-supervised training loop with augmentation |
| `gaussian_rag/training/online_trainer.py` | Continual learning daemon (radiation collector) |
| `gaussian_rag/training/augmentation.py` | Text shuffling / reordering for positive pairs |
| `run_training.py` | Training entrypoint — run this to train the model manually |
| `data/training/robust_corpus.jsonl` | The robust training corpus in JSONL format |
| `data/training/radiation_stats.json` | Persistent tracking of absorbed radiation and training runs |
| `.gaussian_rag_weights_{dimension}.pt` | Saved PyTorch weights for the active embedding dimension |
| `config/config.yaml` | Set `encoder.sigma_mode: "learned"` to activate |

---

## Q: How does information geometry relate to the Gaussian embeddings?

The space of all possible Gaussian distributions `𝒩(μ, Σ)` is not a flat Euclidean space — it is a curved **statistical manifold**. Information geometry is the study of this manifold using tools from differential geometry.

### The Core Idea: Embeddings Live on a Manifold

Standard embeddings treat semantic space as flat **Euclidean space** — every point is just a vector, and distance is distance. Information geometry says this is wrong: the space of probability distributions has **intrinsic curvature**, and the "right" metric to use is the **Fisher information metric**.

### What Your Gaussian Embeddings Represent

In this system, each embedding isn't a point — it's a **Gaussian distribution** `(μ, Σ)`:

| Component | Geometric Meaning |
|---|---|
| `μ` (mean) | Location on the probability manifold |
| `Σ` (covariance / SigmaHead output) | The **local curvature** — how "spread out" the belief is |
| Low σ | Narrow, confident region on the manifold |
| High σ | Broad, uncertain region — the point is ambiguous |

### The Fisher Information Metric

The natural distance between two Gaussians isn't Euclidean — it's the **KL divergence**, which is the geodesic distance under the Fisher metric:

```
D_KL(P || Q) ≈ (1/2) (μ_P - μ_Q)ᵀ Σ⁻¹ (μ_P - μ_Q)   [Mahalanobis form]
```

This is what the `SigmaHead` is learning to calibrate: it is shaping the **local metric tensor** of the manifold. A well-trained SigmaHead means the geodesic distances between embeddings correctly reflect semantic dissimilarity — not just directional similarity.

### Why This Matters for RAG Retrieval

```
Query (μ_q, Σ_q) ──geodesic──▶ Document (μ_d, Σ_d)
```

- **High-uncertainty query** (broad Σ_q): The "radiation" from the query spreads over a larger manifold region → retrieves more candidates → resolves ambiguity by context voting
- **Low-uncertainty document** (narrow Σ_d): A sharp, well-defined concept → strong attractor on the manifold → confident match
- **Σ mismatch** (wide query, narrow doc): The Mahalanobis distance naturally penalizes this → the system prefers docs whose uncertainty profile is compatible with the query's

### The "Shape" Intuition

Think of the manifold as a **landscape**:

```
Flat region  → Low σ, unambiguous concept ("photosynthesis")
Valley/basin → Multiple related senses pooling (high σ, "bank")
Ridge        → Narrow path between two meanings — SigmaHead learns the dividing geometry
```

The `OnlineRadiationCollector` accumulates **evidence of geodesics traversed** — each query-retrieval pair is a sample of the manifold's geometry, and the continual SigmaHead training is literally **refining the local metric tensor** from empirical data.

> **TL;DR:** The Gaussian embeddings are coordinates on an information manifold. The `SigmaHead` learns the **Fisher metric** at each point. Retrieval is **geodesic nearest-neighbour search** under that metric, not flat cosine similarity. More radiation = better metric estimates = tighter geodesics = more precise retrieval.

---

## Q: How do uncertainty bubbles work during retrieval?

### Visualising the Bubbles

Imagine the semantic space projected to 3D (as the dashboard does). Each document and query doesn't appear as a single dot — it appears as a **glowing, translucent bubble**:

- The **centre** of the bubble is `μ` — the core semantic meaning.
- The **size and shape** of the bubble is `Σ` — the uncertainty. A tight bubble means high confidence; a large, diffuse bubble means high ambiguity.

### Retrieval as Bubble Intersection

In standard RAG, retrieval finds the nearest vector (Cosine Similarity).

In GaussianRAG, retrieval calculates the **overlap of the probability mass** between the query bubble and the document bubbles — using the Expected Likelihood Kernel or KL Divergence:

```
Score(Q, D) = 𝒩(μ_q ; μ_d, Σ_q + Σ_d)   [closed-form overlap integral]
```

This fundamentally changes how search works:

- **Broad query (ambiguous):** A single word like *"Apple"* → high uncertainty, massive bubble spanning both the tech and food clusters → retrieves documents from both → system honestly acknowledges it doesn't know which meaning you want.
- **Specific query (confident):** *"Apple M4 Max chip benchmark"* → low uncertainty, tiny bubble → misses fruit documents entirely, overlaps only with specific hardware documents.

### The Asymmetry of Information (The Funnel)

Because the `SigmaHead` learns a proper metric tensor, retrieval becomes **asymmetric**. Distance from A to B is not the same as B to A.

- **Query (broad) → Document (specific):** A large query bubble completely engulfs a tiny, precise document bubble → system says *"this detailed document is a valid subset of your broad question."* (Strong match.)
- **Query (specific) → Document (broad):** A tiny query bubble hits a massive, vague document bubble → *"your query is highly specific, but this document is too generic to be a reliable answer."* (Weak match.)

Standard cosine similarity cannot do this — it treats a vague document and a specific document in the same general direction as identical.

### How It Appears in the Dashboard

1. When a query is fired, it creates a **radiation zone** — the query bubble appears on the manifold.
2. The lines connecting the query to retrieved document nodes represent the **integral of the overlapping volumes** between bubbles.
3. As the user clicks nodes or provides feedback, the query bubble physically **shrinks and shifts** on the manifold, dropping connections to irrelevant nodes as their bubbles no longer overlap.

By using uncertainty bubbles, GaussianRAG doesn't just ask *"What is closest?"* — it asks *"Which documents are most likely to exist within the radius of what the user meant?"*
