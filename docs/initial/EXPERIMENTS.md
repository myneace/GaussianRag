# Experiments — What We Are Testing

> This document defines every experiment, its hypothesis, evaluation protocol, expected results, and what a positive/negative result means for the research.

---

## Experiment Philosophy

Every experiment is structured as:

1. **Hypothesis** — what we expect to be true
2. **Setup** — what we build and measure
3. **Baseline** — what we compare against
4. **Metric** — what a "win" looks like
5. **Interpretation** — what positive and negative results mean

---

## Phase 1 Experiments — Validate the Representation

### EXP-01: Gaussian vs. Point Embedding Retrieval

**Hypothesis:** Gaussian representations with heuristic $\Sigma$ outperform point embeddings on ambiguous queries, with neutral or slight degradation on precise queries.

**Setup:**
- Encode BEIR dataset chunks as both points (cosine) and Gaussians (W2, heuristic $\Sigma$)
- Run retrieval for all queries
- Stratify results by query ambiguity (lexical entropy)

**Baselines:**
- `BM25` — sparse retrieval
- `all-mpnet-base-v2` + cosine similarity — strong dense baseline
- `Contriever` — state-of-the-art dense retriever
- `E5-large`, `BGE-large` — top performing dense encoders

**Metrics:** nDCG@10, Recall@100

**Expected result:**
- +1–3% nDCG@10 on high-ambiguity query stratum
- Neutral (±0.5%) on low-ambiguity stratum
- Slight drop possible on precise factoid queries

**Positive result means:** Gaussian representation adds signal beyond the mean.

**Negative result means:** Heuristic $\Sigma$ is too noisy — move directly to learned $\Sigma$.

---

### EXP-02: Distance Metric Ablation

**Hypothesis:** W2 outperforms FR (diagonal approx) for retrieval; KL performs worst due to asymmetry.

**Setup:**
- Same knowledge base and queries as EXP-01
- Swap only the distance metric: cosine / W2 / FR (diagonal) / KL / Bhattacharyya

**Metric:** nDCG@10, query latency (ms)

**Expected result:**

| Metric | W2 | FR (diag) | KL | Bhattacharyya | Cosine |
|---|---|---|---|---|---|
| nDCG@10 | Best | 2nd | Worst | 3rd | Baseline |
| Latency | Medium | Medium | Fast | Fast | Fastest |

---

### EXP-03: Covariance Structure Ablation

**Hypothesis:** Diagonal + low-rank $\Sigma$ outperforms spherical and matches near-full covariance at a fraction of the cost.

**Setup:**
- Fix W2 metric and heuristic $\Sigma$ estimation
- Vary covariance structure: spherical / diagonal / diagonal+low-rank (r=4,8,16,32) / full
- Measure retrieval quality and memory footprint

**Metric:** nDCG@10 vs. memory (GB) Pareto frontier

**Expected result:** Diagonal+low-rank (r=16) sits on the Pareto frontier.

---

## Phase 2 Experiments — Validate Learned $\Sigma$

### EXP-04: Heuristic vs. Learned $\Sigma$

**Hypothesis:** Learned $\Sigma$ (contrastive probabilistic loss) outperforms heuristic $\Sigma$ by a significant margin.

**Setup:**
- Train $\Sigma$ head on MS-MARCO with contrastive W2 loss
- Evaluate on BEIR zero-shot
- Compare to heuristic $\Sigma$ from EXP-01

**Metric:** nDCG@10, Recall@100

**Critical check:** Learned $\Sigma$ must outperform heuristic $\Sigma$ by more than the compute cost justifies.

---

### EXP-05: Training Loss Ablation

**Hypothesis:** Contrastive W2 loss + augmentation consistency outperforms either alone.

**Setup:**
Train $\Sigma$ head with:
- W2 contrastive loss only
- Augmentation consistency loss only
- Both combined
- Bayesian evidential head (alternative approach)

**Metric:** nDCG@10 on BEIR, calibration (ECE) on held-out queries

---

### EXP-06: $\Sigma$ Sensitivity Analysis

**Hypothesis:** Retrieval quality degrades gracefully when $\Sigma$ is perturbed, but catastrophically when $\Sigma$ collapses to zero or explodes.

**Setup:**
- Take trained model
- Perturb $\Sigma$ by adding noise at varying scales: $\Sigma' = \Sigma + \epsilon \cdot \text{noise}$
- Measure retrieval quality at each perturbation level
- Also test: clamp $\Sigma \to 0$ (point embedding limit) and $\Sigma \to \infty$ (uniform distribution)

**Metric:** nDCG@10 vs. perturbation magnitude

**This answers Q14 directly.**

---

## Phase 3 Experiments — Domain and Scale

### EXP-07: Domain-Specific Evaluation

**Hypothesis:** Gaussian RAG provides the largest gains in high-uncertainty domains (medical, legal, scientific).

**Setup:**
Run full pipeline on domain-specific datasets:
- **Medical:** MedQA, PubMedQA, BioASQ
- **Legal:** LegalBench, CaseHOLD
- **Scientific:** SciFact, QASPER
- **Factoid (control):** Natural Questions, TriviaQA

**Metric:** nDCG@10 per domain, gain vs. cosine baseline

**Expected result:**

| Domain | Expected Gain |
|---|---|
| Medical | +3–6% |
| Legal | +3–5% |
| Scientific | +2–4% |
| Factoid | −1 to +1% |

---

### EXP-08: Scaling Behavior

**Hypothesis:** Manifold retrieval degrades gracefully with tangent-space HNSW indexing.

**Setup:**
- Scale knowledge base: 10K → 100K → 1M → 10M chunks
- Measure nDCG@10 and query latency at each scale
- Compare: exact W2 vs. HNSW-approximate W2 vs. cosine

**Metric:** nDCG@10 and latency (ms/query) at each scale

---

## Phase 4 Experiments — LLM Integration

### EXP-09: Uncertainty Communication Methods

**Hypothesis:** Prompt-based uncertainty passing improves LLM calibration. Fine-tuning improves it further.

**Setup:**
Run end-to-end QA with:
- A: No uncertainty signal (retrieved chunks only)
- B: Confidence labels in prompt (`[HIGH/MEDIUM/LOW]`)
- C: Soft token weighting by $\exp(-\frac{1}{2}W_2^2)$
- D: Fine-tuned LLM with uncertainty-aware objective

**Metric:** ECE (Expected Calibration Error), Brier Score, Selective Risk, AUROC on abstention

---

### EXP-10: Abstention Quality

**Hypothesis:** When all retrieved chunks have high $\Sigma$, the system should abstain or hedge — and this is measurably better than confidently answering with uncertain knowledge.

**Setup:**
- Construct "uncertain query" test set — questions where ground truth is contested or unknown
- Compare abstention rates and accuracy across: baseline RAG, GaussianRAG, GaussianRAG + fine-tuned LLM

**Metric:** Selective accuracy (accuracy when not abstaining), abstention recall

---

## Novel Contribution Experiments

### EXP-11: Knowledge Coverage Metric

**Hypothesis:** The coverage metric identifies genuine knowledge gaps — queries in low-coverage regions have lower retrieval quality.

**Setup:**
- Compute coverage over query distribution
- Identify low-coverage query clusters
- Verify: retrieval nDCG@10 correlates negatively with coverage (Pearson r)

**This directly validates Q20.**

---

### EXP-12: Knowledge Drift Detection

**Hypothesis:** Gaussian drift velocity $\|\dot{\gamma}\|_{FR}$ detects when knowledge becomes outdated.

**Setup:**
- Use timestamped Wikipedia revisions as dynamic knowledge base
- Track $(\mu(t), \Sigma(t))$ trajectories for evolving articles
- Measure: can drift signal predict when a stored fact has been contradicted?

**Metric:** AUC for "outdated fact detection" task

**This directly validates Q19.**

---

## Evaluation Protocol Summary

### Datasets

| Dataset | Task | Domain | Queries |
|---|---|---|---|
| BEIR (15 subsets) | Retrieval | Mixed | ~100K |
| HotpotQA | Multi-hop QA | Wikipedia | ~90K |
| TriviaQA | Factoid QA | Mixed | ~95K |
| MedQA | QA | Medical | ~12K |
| SciFact | Fact verification | Scientific | ~1K |
| LegalBench | Classification + QA | Legal | ~5K |

### Metrics

| Metric | What It Measures | Used In |
|---|---|---|
| nDCG@10 | Ranked retrieval quality | All retrieval exps |
| Recall@100 | Coverage of relevant docs | All retrieval exps |
| ECE | Calibration of confidence | LLM integration |
| Brier Score | Probabilistic accuracy | LLM integration |
| Selective Risk | Quality when not abstaining | Abstention |
| Latency (ms) | Query speed | Scaling |

### Statistical Rigor

- All experiments: 3 random seeds, report mean ± std
- Query-type stratification: split by lexical entropy quartile
- Statistical tests: paired t-test or Wilcoxon signed-rank vs. baselines
- Report effect size (Cohen's d) alongside p-values

---

## Ablation Summary Table

| Variable | Values Tested | Primary Metric |
|---|---|---|
| Distance metric | cosine, W2, FR, KL, Bhatt | nDCG@10 |
| Σ structure | spherical, diag, diag+LR, full | nDCG@10, memory |
| Σ estimation | heuristic, learned | nDCG@10 |
| Training loss | W2, consistency, both | nDCG@10, ECE |
| Query representation | point, Gaussian | nDCG@10 |
| Uncertainty signal to LLM | none, prompt, soft, fine-tuned | ECE, Brier |
| Knowledge base scale | 10K–10M | nDCG@10, latency |
