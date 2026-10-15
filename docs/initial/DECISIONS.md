# Design Decisions — GaussianRAG

> This document records every major design decision, the alternatives considered, and the rationale for the choice made. Update this as decisions evolve.

---

## DD-01: Covariance Structure — Diagonal + Low-Rank

**Decision:** Use $\Sigma = \text{diag}(v) + LL^\top$ with a low-rank factor configured per runtime (default `r = 8`).

**Alternatives considered:**
- Full $\Sigma$ — $O(d^2)$ parameters, intractable at $d = 768$, overfits without strong priors
- Diagonal only — $O(d)$, misses cross-dimensional correlations
- Spherical $\sigma^2 I$ — loses all directional information, defeats the purpose

**Rationale:** Diagonal + low-rank balances expressiveness (captures dominant cross-axis correlations via $LL^\top$) with tractability. Standard in probabilistic ML (PPCA, VAEs with structured covariance). The active prototype exposes `low_rank` through config so the factor can evolve without changing the covariance contract.

**Risk:** Cross-dimensional correlations not captured by rank-16 approximation are lost. Acceptable tradeoff.

---

## DD-02: Primary Retrieval Metric — Expected Likelihood Kernel

**Decision:** Use ELK for live retrieval. Retain W2 and FR as alternate geometric probes and analysis metrics.

**Alternatives considered:**
- Wasserstein-2 — useful geometric probe, but not the canonical query/document overlap rule from the paper
- Fisher-Rao — reparameterization-invariant but $O(d^3)$, no closed form for general $\Sigma$
- KL divergence — asymmetric, undefined for non-overlapping support, poor for retrieval
- Bhattacharyya — symmetric, tractable, but less theoretically motivated for manifold geometry
- Cosine similarity — point-only, ignores $\Sigma$ entirely

**Rationale:** ELK directly matches the paper's `Score(Q, D) = 𝒩(μ_q; μ_d, Σ_q + Σ_d)` retrieval rule and treats both query and document as Gaussian bubbles. W2 and FR remain valuable for manifold auditing, diversity checks, and comparative experiments.

---

## DD-03: Two-Stage Retrieval — ANN then Exact

**Decision:** The target design is FAISS ANN on $\mu$ vectors (top-100) followed by exact ELK ranking (top-5). The current repository prototype uses a NumPy linear scan while preserving the same two-stage interface.

**Alternatives considered:**
- Exact ELK for all chunks — $O(n)$ per query, prohibitive at scale
- ANN directly on manifold — no mature library, requires tangent-space projection
- Only ANN on $\mu$, no ELK re-ranking — ignores $\Sigma$ entirely

**Rationale:** ANN on $\mu$ is fast and filters the space. Exact ELK on the candidate set restores the paper-aligned Gaussian overlap score without paying $O(n)$ per query across the full corpus.

---

## DD-04: Query Representation — Gaussian Query in Live Retrieval

**Decision:** Live ELK and FR retrieval paths build Gaussian queries $\mathcal{N}(\mu_q, \Sigma_q)$; purely point-query behavior is retained only for older or alternate metric paths when explicitly requested.

**Alternatives considered:**
- Point query only — simpler, but incompatible with the paper's ELK retrieval rule
- Fixed isotropic query $\mathcal{N}(\mu_q, \alpha I)$ — crude but better than point

**Rationale:** Once ELK became the live default, a Gaussian query was no longer optional: the retrieval rule itself depends on `Σ_q + Σ_d`. Query covariance is therefore part of the active prototype rather than a deferred extension.

---

## DD-05: $\Sigma$ Estimation — Augmentation Consistency

**Decision:** Use augmentation-based empirical covariance for heuristic $\Sigma$, contrastive W2 loss for learned $\Sigma$.

**Alternatives considered:**
- Chunk-length heuristic — too crude, ignores semantic structure
- Lexical entropy — fast but ignores embedding space geometry
- Normal-Inverse-Wishart prior — principled Bayesian approach but complex to implement stably
- Laplace approximation over encoder — theoretically clean but computationally expensive

**Rationale:** Augmentation consistency has a clean geometric interpretation: $\Sigma$ should reflect the spread of a chunk's meaning under semantic-preserving perturbations. Easy to compute without training. Contrastive loss then refines this with supervision from retrieval performance.

---

## DD-06: Diversity Filtering — W2 Threshold

**Decision:** Filter retrieved chunks where pairwise W2 distance < diversity threshold (0.3).

**Alternatives considered:**
- MMR (Maximal Marginal Relevance) — standard approach, but uses cosine similarity
- Submodular selection — theoretically optimal but $O(k^2 n)$ per query
- No diversity filtering — risk of retrieving near-duplicate chunks

**Rationale:** W2-based threshold is consistent with the rest of the pipeline (same metric) and runs in $O(k^2)$ after retrieval. Submodular selection is a future upgrade.

---

## DD-07: Uncertainty Communication to LLM — Prompt Labels

**Decision:** Phase 1 uses structured prompt labels `[HIGH/MEDIUM/LOW]`. Soft token weighting and fine-tuning are Phase 3.

**Alternatives considered:**
- No uncertainty signal — wastes the $\Sigma$ information at generation time
- Raw confidence scores in prompt — less interpretable to LLM without training
- Attention modulation — requires architecture changes, not usable with API-based LLMs
- Fine-tuning from the start — requires labeled uncertainty data, defer until we have a working system

**Rationale:** Prompt labels are immediately usable with any LLM via API, require no architecture changes, and are interpretable. Soft token weighting and fine-tuning are planned for later phases once the retrieval improvement is validated.

---

## DD-08: SPD Regularization — Eigenvalue Flooring

**Decision:** Enforce $\Sigma \succeq \epsilon I$ with $\epsilon = 10^{-4}$ via eigenvalue clipping.

**Alternatives considered:**
- Log-barrier penalty in loss — soft constraint, doesn't guarantee SPD
- Cholesky parameterization — guarantees SPD by construction, preferred for learned $\Sigma$
- No regularization — risks singular $\Sigma$, manifold degeneracy

**Rationale:** Cholesky parameterization for learned $\Sigma$ (output $L$ where $\Sigma = LL^\top + \epsilon I$). Eigenvalue flooring as post-hoc safety for heuristic $\Sigma$.

---

## DD-09: Encoder — Frozen Pretrained

**Decision:** Use a config-selected frozen encoder for $\mu$ with a heuristic or fitted scaling $\Sigma$ head. The current repo supports both local `all-MiniLM-L6-v2` and Gemini Cloud embeddings, with the active runtime determined by `config/config.yaml`.

**Alternatives considered:**
- End-to-end fine-tuning — expensive, risks catastrophic forgetting, unnecessary for Phase 1
- Train encoder jointly with $\Sigma$ head — complex, large computational budget

**Rationale:** The encoder already provides high-quality $\mu$. The research contribution is in $\Sigma$ — isolate that. Joint fine-tuning is a potential future direction if $\Sigma$ quality is bottlenecked by $\mu$ quality.

---

## DD-10: Knowledge Base Format — Flat + FAISS Hybrid

**Decision:** The target design is FAISS for $\mu$ plus flat numpy-side $\Sigma$ storage indexed by chunk ID. The current prototype keeps both in local Python structures with the same retrieval contract.

**Alternatives considered:**
- Qdrant with custom distance — no native W2 support
- Full manifold database — doesn't exist yet for SPD manifolds
- Everything in FAISS — FAISS doesn't support custom metrics natively

**Rationale:** FAISS is the fastest ANN library for $\mu$ search. $\Sigma$ retrieval by ID is $O(1)$ after FAISS returns candidate IDs. Pragmatic hybrid that works today.

---

## Open Decisions (To Be Made)

| Decision | Options | Blocking |
|---|---|---|
| Low-rank $r$ value | 4, 8, 16, 32 | EXP-03 |
| Contrastive loss margin | 0.1 – 1.0 | EXP-05 |
| Diversity threshold | 0.2 – 0.5 | EXP-01 |
| ANN candidate pool size | 50, 100, 200 | EXP-08 |
| Query $\Sigma$ estimation | heuristic, learned, entity-aware, dissolver fallback | Ongoing calibration |
