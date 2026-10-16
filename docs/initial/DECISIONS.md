# Design Decisions — Gaussian-Clef RAG

> Architectural Decision Records (ADRs): rationale, alternatives considered, and tradeoffs for core design decisions. Update this document as decisions evolve.

---

## DD-01: Covariance Structure — Diagonal + Low-Rank

**Decision:** Use $\Sigma = \text{diag}(v) + LL^\top$ with a low-rank factor configured per runtime (default $r = 8$).

**Alternatives considered:**
- Full $\Sigma$ — $O(d^2)$ parameters, intractable at $d = 768$, overfits without strong priors
- Diagonal only — $O(d)$, misses cross-dimensional correlations
- Spherical $\sigma^2 I$ — loses all directional information, defeats the purpose

**Rationale:** Diagonal + low-rank balances expressiveness (captures dominant cross-axis correlations via $LL^\top$) with tractability. Standard in probabilistic ML (PPCA, VAEs with structured covariance).

---

## DD-02: Primary Retrieval Metric — Expected Likelihood Kernel (ELK)

**Decision:** Use ELK for live retrieval. Retain W2 and FR as alternate geometric probes and analysis metrics.

**Alternatives considered:**
- Wasserstein-2 — useful geometric probe, but not the canonical query/document overlap rule from the paper
- Fisher-Rao — reparameterization-invariant but $O(d^3)$, no closed form for general $\Sigma$
- KL divergence — asymmetric, undefined for non-overlapping support, poor for retrieval
- Cosine similarity — point-only, ignores $\Sigma$ entirely

**Rationale:** ELK directly matches the canonical overlap rule `Score(Q, D) = 𝒩(μ_q; μ_d, Σ_q + Σ_d)` and treats both query and document as Gaussian distributions.

---

## DD-03: Two-Stage Retrieval — ANN Candidate Filtering then Exact Reranking

**Decision:** Filter the candidate space first with fast Approximate Nearest Neighbor (ANN) and lexical pre-filtering, then compute exact Expected Likelihood Kernel overlap on the top candidate set (~30 candidates).

**Rationale:** Exact manifold evaluation across millions of vectors is intractable. Pre-filtering bounds the candidate set so exact geometric overlap scoring runs in under 3ms.

---

## DD-04: Query Representation — Gaussian Query in Live Retrieval

**Decision:** Live ELK and FR retrieval paths build Gaussian queries $\mathcal{N}(\mu_q, \Sigma_q)$; point-query behavior is retained only for backward compatibility.

**Rationale:** The ELK retrieval rule depends on $\Sigma_q + \Sigma_d$. Query covariance captures query ambiguity and specificity, enabling calibrated overlap estimation.

---

## DD-05: $\Sigma$ Estimation — Augmentation Consistency & SigmaHead

**Decision:** Use augmentation-based empirical covariance for heuristic $\Sigma$, and a PyTorch `SigmaHead` MLP trained with contrastive W2 loss for learned $\Sigma$.

---

## DD-06: Diversity Filtering — W2 Threshold

**Decision:** Filter retrieved chunks where pairwise W2 distance < diversity threshold (0.3).

---

## DD-07: Uncertainty Communication to LLM — Calibrated Prompt Labels

**Decision:** Pass structured prompt labels `[HIGH/MEDIUM/LOW]` to the generation LLM derived from covariance trace $\text{tr}(\Sigma)$.

---

## DD-08: SPD Regularization — Eigenvalue Flooring

**Decision:** Enforce $\Sigma \succeq \epsilon I$ with $\epsilon = 10^{-4}$ via eigenvalue clipping to prevent numerical degeneracy.

---

## DD-09: Encoder — Frozen Pretrained with Plug-and-Play Backends

**Decision:** Support both local `sentence-transformers/all-MiniLM-L6-v2` (384-dim, 100% offline) and cloud Gemini embeddings (768-dim) via configuration.

---

## DD-10: Dual-Store Ingestion Contract (`KnowledgeStore` + `LexicalStore`)

**Decision:** Rather than relying solely on dense vector search, maintain lockstep synchronization between `KnowledgeStore` (dense $\mathcal{N}(\mu, \Sigma)$) and `LexicalStore` (embedded SQLite FTS5).

**Rationale:** Rare entities, unique codes, and exact names (e.g. "Nicolas Fouquet") are vulnerable to ANN semantic drift. Hybrid indexing guarantees 100% lexical recall while preserving manifold geometry.

---

## DD-11: Woodbury Matrix Identity and Sylvester Determinant Acceleration

**Decision:** Accelerate ELK kernel calculation from $O(d^3)$ to $O(d \cdot r^2)$ by applying the Woodbury matrix inversion lemma and Sylvester determinant theorem to low-rank covariance sums.

**Alternatives considered:**
- Direct `np.linalg.inv` or `np.linalg.eigh`: $O(d^3)$ complexity, ~150ms per query across 30 candidates.
- Pure diagonal approximation: Drops $LL^T$ entirely, losing directional correlation.

**Rationale:** The sum of two low-rank Gaussians $\Sigma_q + \Sigma_d = \text{diag}(v_q + v_d) + [L_q, L_d] [L_q, L_d]^T$ has rank $2r \le 16$. Inverting a $16 \times 16$ core matrix takes microseconds, reducing batch reranking latency from 150ms to <3ms.

---

## DD-12: Cloudflare Clef-Flash for Non-Autoregressive Ambiguity & Decision Gating

**Decision:** Use Cloudflare Workers AI Clef-Flash (`@cf/cloudflare/clef-flash`) for ambiguity detection and passage gating instead of autoregressive conversational LLMs.

**Alternatives considered:**
- Autoregressive LLM prompts: Generates token sequences, prone to hallucination and JSON schema syntax errors, 1,200–2,000ms latency.
- Static regex or dictionary heuristics: Too rigid, fails on subtle semantic polysemy.

**Rationale:** Clef-Flash returns direct probability calibrations over typed `choice`, `noul`, and `score` questions in 30–40ms, cutting decision latency by 95% without JSON parse failures.

---

## DD-13: Parallel 4-Question Passage Gating

**Decision:** Filter the candidate pool of 30 candidates through a 4-question Clef passage gate (relevance, factual evidence, contradiction, and sufficiency) in parallel batches prior to generation.

**Rationale:** Pruning noisy and contradictory chunks before generation protects the synthesis LLM from hallucinations and reduces generation context bloat.

---

## DD-14: Standalone Python Package & High-Level `GaussianRAG` Facade

**Decision:** Structure Gaussian-Clef RAG as an installable standalone Python library (`pip install -e .`) with an intuitive facade (`GaussianRAG.create()`, `add_texts()`, `add_raw_text()`, `add_markdown()`, `search()`, `query()`, `save()`, `load()`).

**Rationale:** Decouples the core engine from ad-hoc script runners and CLI scaffolds, enabling seamless integration into external Python applications.

---

## DD-15: Flexible Corpus Generation & Text Normalization with Content Caching

**Decision:** Implement a multi-stage text normalization (`TextNormalizer`) and structure-detecting corpus generation engine (`CorpusGenerator`) with SHA256 content-addressed caching (`CorpusCache`).

**Rationale:** Real-world knowledge inputs are messy: OCR hyphenation wraps, web scraping HTML tags, chat transcripts. Built-in normalization and caching ensure effortless onboarding of dirty text without external ETL dependencies.
