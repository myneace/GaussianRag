# Research Questions — GaussianRAG

> These 20 questions form the theoretical and empirical backbone of the research. Every design decision in the system should be traceable to at least one of these questions.

---

## 🔬 Foundational / Theoretical

**Q1. Gaussians vs. Point Embeddings**
Does representing knowledge as Gaussian distributions $\mathcal{N}(\mu, \Sigma)$ preserve more semantic information than point embeddings, and under what conditions?

- *Current understanding:* Point embeddings collapse semantic variance into a single vector, discarding ambiguity and evidence strength. Gaussians preserve second-order structure.
- *Conditions for gain:* Noisy, multi-faceted, or heterogeneous sources. Gains diminish for highly curated, atomic facts.
- *Caveat:* Gaussians assume unimodality — polysemous concepts may need mixtures.

---

**Q2. Fisher-Rao vs. Wasserstein-2 vs. KL Divergence**
What is the theoretical justification for each metric, and when does each break down?

- *Fisher-Rao:* Reparameterization-invariant, geodesically complete. Best for theory. Breaks near degenerate $\Sigma$, $O(d^3)$ cost.
- *Wasserstein-2:* Symmetric, robust to covariance misspecification. Best for retrieval. Breaks under high anisotropy without regularization.
- *KL:* Tractable but asymmetric. Use only for training objectives.
- *Decision (historical):* Early experiments prioritized W2 for retrieval and FR for theoretical analysis, but the current paper-aligned runtime is ELK-first with W2/FR retained as alternate probes.

---

**Q3. Is the Manifold Well-Behaved?**
Is the space of all knowledge Gaussians a legitimate Riemannian manifold, or does it have degenerate regions?

- *Answer:* $\mathcal{M} = \mathbb{R}^d \times \text{SPD}(d)$ with FR metric is smooth and geodesically complete.
- *Degenerate regions:* $\det(\Sigma) \to 0$ causes singularities. Enforce $\Sigma \succeq \epsilon I$.
- *Indexing challenge:* ANN on $\mathcal{M}$ lacks mature libraries — requires tangent-space projection.

---

**Q4. Same $\mu$, Different $\Sigma$ — What Does It Mean?**
Are two chunks with identical $\mu$ but different $\Sigma$ "the same knowledge with different confidence"?

- *Answer:* No. $\Sigma$ is a *directional* semantic covariance, not a scalar confidence.
- *Example:* $\mu$ = "cardiovascular drug", $\Sigma$ large along "dosage" dimension but tight along "mechanism". Same concept, different specificity.

---

**Q5. Fisher-Rao Geodesic as Concept Interpolation**
Can the geodesic between two knowledge Gaussians be interpreted as meaningful semantic interpolation?

- *Answer:* Yes in theory. Smoothly interpolates both location and spread.
- *Limitation:* LLM embedding spaces are entangled — geodesic midpoints may not align with human-interpretable concepts without disentanglement.

---

## 📐 Modeling / Design

**Q6. What Covariance Structure?**
Should $\Sigma$ be full, diagonal, or spherical?

- *Full:* $O(d^2)$ params, intractable for $d \geq 384$.
- *Diagonal:* $O(d)$, usually sufficient after PCA/whitening.
- *Spherical:* Too restrictive, loses directional semantics.
- *Decision:* **Diagonal + low-rank correction** $\Sigma = \text{diag}(v) + LL^\top$, $r \ll d$.

---

**Q7. How Do We Learn $\Sigma$?**
What is the training signal for covariance estimation?

- *Approaches:*
  - Contrastive probabilistic loss (Bhattacharyya or W2 margin)
  - Augmentation consistency — high agreement across perturbations → small $\Sigma$
  - Evidential/Bayesian head over encoder
- *Regularization:* $\log\det(\Sigma)$ penalties or KL to isotropic prior.

---

**Q8. Should the Query Be a Gaussian Too?**
Point query vs. Gaussian query — which is better?

- *Gaussian query:* Best for ambiguous, multi-intent queries.
- *Point query:* Simpler, faster, sufficient for precise queries.
- *Decision (historical):* The initial prototype started with point queries, but the current ELK-first runtime now builds Gaussian queries on the live ELK and Fisher-Rao paths.

---

**Q9. How to Handle the GMM over the Knowledge Base?**
Do we retrieve individual Gaussians or integrate over mixture components?

- *Decision:* Index components individually, retrieve top-$k$ via manifold ANN, optionally re-rank using mixture-level similarity.
- Full integration is intractable at scale.

---

**Q10. Handling Heavy Overlap Between Retrieved Gaussians**
When chunks overlap significantly, how do we avoid redundant retrieval?

- *Approaches:*
  - Penalize pairwise FR/W2 distance below threshold
  - Submodular selection (facility location with Gaussian kernels)
  - Iterative conditional retrieval — mask covered manifold regions
- Overlap can also serve as a *consensus signal* for LLM calibration.

---

## ⚙️ Empirical / Experimental

**Q11. Does It Actually Outperform Cosine Similarity?**
On standard benchmarks (HotpotQA, TriviaQA, BEIR) — by how much?

- *Expected:* Neutral to +2–5% nDCG@10. Gains concentrate on ambiguous, multi-hop queries.
- *Baselines required:* Contriever, E5, BGE (strong dense retrievers).

---

**Q12. Query-Type Interaction Effect**
Does benefit increase for ambiguous vs. precise queries?

- *Hypothesis:* Yes. Strong interaction expected — Gaussian retrieval outperforms on high-entropy queries.
- *Evaluation:* Stratified analysis + interaction ANOVA.

---

**Q13. Scaling Behavior**
Does the manifold approach degrade gracefully at 1M, 10M chunks?

- *Risk:* Covariance noise compounds without careful indexing.
- *Mitigation:* Tangent-space HNSW, SPD product quantization, hierarchical clustering on $\mathcal{M}$.

---

**Q14. Sensitivity to $\Sigma$ Quality**
How badly does a miscalibrated $\Sigma$ hurt?

- *Answer:* Significantly. Poorly calibrated $\Sigma$ hurts more than just using point embeddings.
- *Mitigation:* Eigenvalue flooring, uncertainty-aware fallback, post-hoc isotonic calibration.

---

**Q15. Which Domain Benefits Most?**
Where are the gains largest?

- *Best domains:* Medical, legal, scientific, financial/policy, cross-lingual.
- *Common thread:* High inherent uncertainty or conflicting evidence.

---

## 🧠 LLM Integration

**Q16. How Do We Communicate Uncertainty to the LLM?**
Prompting vs. attention modulation vs. soft token weighting?

- *Most robust now:* Structured metadata in prompt (`[confidence: high/medium/low]`).
- *Most promising future:* Soft token weighting by $\exp(-\frac{1}{2}D_{FR}^2)$.

---

**Q17. Does the LLM Actually Become Better Calibrated?**
Does uncertainty-weighted context improve generation?

- *Zero-shot:* Marginal. LLMs ignore uncertainty without training.
- *Fine-tuned:* Significant gains in ECE, Brier score, factual consistency.
- *Evaluate via:* Abstention curves, selective risk, counterfactual stress tests.

---

**Q18. Can LLM Confidence Feed Back to Update $\Sigma$?**
A self-improving knowledge system?

- *Possible but risky.* Confirmation bias and error amplification are real dangers.
- *Safer:* Kalman-style updates on manifold with strong priors, human verification as signal.

---

## 🚀 Novel Contributions

**Q19. Knowledge Drift Detection**
Can we detect when stored knowledge becomes outdated by tracking Gaussian movement on the manifold?

- Model $(\mu(t), \Sigma(t))$ as trajectory. Drift velocity = $\|\dot{\gamma}(t)\|_{FR}$.
- Detect drift when geodesic displacement exceeds threshold, or $\Sigma$ shrinks/expands sharply.

---

**Q20. Knowledge Coverage Metric**
Can we measure how much of the manifold our knowledge base spans — to find gaps?

- Coverage = manifold volume covered by $\bigcup_i \{x : D_{FR}(x, \mathcal{N}_i) \leq \tau\}$.
- Practical proxy: average nearest-neighbor FR distance over query distribution.
- Use to guide active knowledge acquisition.

---

## Priority Order

Questions to answer first (spine of the paper):

1. **Q1** — Does the representation gain matter?
2. **Q2** — Which metric to use?
3. **Q7** — Can we learn $\Sigma$ well?
4. **Q11** — Does it outperform baselines?
5. **Q20** — What is the coverage of our knowledge base?
