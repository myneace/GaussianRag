# Theory — Mathematical Foundations of GaussianRAG

> This document covers the information geometry, probabilistic representation, and retrieval mathematics that underpin the system.

---

## 1. The Core Representation

Every knowledge chunk $c$ is encoded as a multivariate Gaussian:

$$\mathcal{N}(\mu_c, \Sigma_c), \quad \mu_c \in \mathbb{R}^d, \quad \Sigma_c \in \text{SPD}(d)$$

where $\text{SPD}(d)$ is the space of symmetric positive definite matrices of dimension $d$.

- $\mu_c$ — the **central semantic meaning** of the chunk (learned from encoder)
- $\Sigma_c$ — the **directional uncertainty** of the chunk (learned or heuristic)

**Important:** $\Sigma_c$ is NOT a scalar confidence score. It is a full directional covariance — some semantic dimensions may be tight (certain) while others are wide (uncertain).

---

## 2. The Statistical Manifold

The space of all knowledge Gaussians lives on:

$$\mathcal{M} = \mathbb{R}^d \times \text{SPD}(d)$$

Equipped with the **Fisher-Rao (FR) metric**, this is a smooth, geodesically complete Riemannian manifold. This is the core of Amari's Information Geometry.

### Fisher Information Matrix

For a Gaussian $\mathcal{N}(\mu, \Sigma)$, the Fisher information metric at a point $\theta = (\mu, \Sigma)$ is:

$$g_{ij}(\theta) = \mathbb{E}\left[\frac{\partial \log p(x;\theta)}{\partial \theta_i} \frac{\partial \log p(x;\theta)}{\partial \theta_j}\right]$$

This induces a natural Riemannian geometry on the space of distributions.

### Degenerate Regions

When $\det(\Sigma) \to 0$, the manifold becomes singular. In practice, enforce:

$$\Sigma \succeq \epsilon I, \quad \epsilon = 10^{-4}$$

---

## 3. Distance Metrics

### 3.1 Wasserstein-2 Distance (Alternate Geometric Probe)

For two Gaussians $\mathcal{N}(\mu_1, \Sigma_1)$ and $\mathcal{N}(\mu_2, \Sigma_2)$, W2 has a closed form:

$$W_2^2(\mathcal{N}_1, \mathcal{N}_2) = \|\mu_1 - \mu_2\|^2 + \mathcal{B}^2(\Sigma_1, \Sigma_2)$$

where the Bures metric is:

$$\mathcal{B}^2(\Sigma_1, \Sigma_2) = \text{tr}(\Sigma_1) + \text{tr}(\Sigma_2) - 2\,\text{tr}\!\left(\left(\Sigma_1^{1/2} \Sigma_2 \Sigma_1^{1/2}\right)^{1/2}\right)$$

**Properties:** Symmetric, handles support mismatch, robust to covariance misspecification. In the current paper-aligned runtime it is retained as an alternate geometric probe rather than the canonical ELK retrieval rule.

### 3.2 Fisher-Rao Distance (Theoretical Analysis)

No closed form for general Gaussians. For diagonal $\Sigma = \text{diag}(\sigma_1^2, ..., \sigma_d^2)$:

$$d_{FR}^2(\mathcal{N}_1, \mathcal{N}_2) = \sum_i \left(\frac{(\mu_{1i} - \mu_{2i})^2}{2(\sigma_{1i}^2 + \sigma_{2i}^2)} + \frac{1}{2}\log\frac{\sigma_{1i}^2 + \sigma_{2i}^2}{2\sigma_{1i}\sigma_{2i}}\right)$$

**Properties:** Reparameterization-invariant, geodesically principled. Use for theoretical analysis and interpolation.

### 3.3 KL Divergence (Training Only)

$$D_{KL}(\mathcal{N}_1 \| \mathcal{N}_2) = \frac{1}{2}\left[\log\frac{|\Sigma_2|}{|\Sigma_1|} - d + \text{tr}(\Sigma_2^{-1}\Sigma_1) + (\mu_2 - \mu_1)^\top \Sigma_2^{-1} (\mu_2 - \mu_1)\right]$$

**Properties:** Asymmetric. Use only for training objectives (variational bounds, contrastive loss). Never use for retrieval.

### 3.4 Bhattacharyya Distance (Alternative Training Signal)

$$D_B(\mathcal{N}_1, \mathcal{N}_2) = \frac{1}{8}(\mu_1-\mu_2)^\top \Sigma^{-1}(\mu_1-\mu_2) + \frac{1}{2}\log\frac{|\Sigma|}{\sqrt{|\Sigma_1||\Sigma_2|}}$$

where $\Sigma = \frac{\Sigma_1 + \Sigma_2}{2}$.

---

## 4. Covariance Structure

### 4.1 Options

| Structure | Params | Expressiveness | Tractable at $d=768$? |
|---|---|---|---|
| Full $\Sigma$ | $O(d^2)$ | Maximum | ❌ |
| Diagonal $\Sigma$ | $O(d)$ | Axis-aligned only | ✅ |
| Spherical $\sigma^2 I$ | $O(1)$ | None | ✅ |
| Diagonal + Low-rank | $O(d + dr)$ | Near-full | ✅ |

### 4.2 Chosen Structure

$$\Sigma = \text{diag}(v) + LL^\top, \quad v \in \mathbb{R}^d_+, \quad L \in \mathbb{R}^{d \times r}, \quad r \ll d$$

This is the **diagonal + low-rank correction**. It captures dominant cross-dimensional correlations via $LL^\top$ while remaining computationally tractable.

---

## 5. Learning $\Sigma$

### 5.1 Augmentation Consistency

For a chunk $c$, generate $K$ augmented views $\{c_1, ..., c_K\}$ (paraphrase, truncation, noise). Encode each as $\mu_k$. Set:

$$\hat{\Sigma}_c = \frac{1}{K}\sum_k (\mu_k - \bar{\mu})(\mu_k - \bar{\mu})^\top$$

High agreement across augmentations → small $\Sigma$. Disagreement → large $\Sigma$.

### 5.2 Contrastive Probabilistic Loss

For positive pair $(c, c^+)$ and negative $c^-$:

$$\mathcal{L} = \max(0, D_B(\mathcal{N}_c, \mathcal{N}_{c^+}) - D_B(\mathcal{N}_c, \mathcal{N}_{c^-}) + \text{margin})$$

### 5.3 Regularization

Prevent $\Sigma$ collapse or explosion:

$$\mathcal{L}_{\text{reg}} = \lambda_1 \|\log\det(\Sigma)\|^2 + \lambda_2 D_{KL}(\mathcal{N}(\mu, \Sigma) \| \mathcal{N}(0, I))$$

---

## 6. Retrieval as Probabilistic Inference

Given a query $q$ encoded as $\mathcal{N}(\mu_q, \Sigma_q)$ and knowledge base $\{\mathcal{N}_i\}$, the paper-aligned live retrieval rule is the Expected Likelihood Kernel:

$$\text{score}(q, c_i) = \mathcal{N}(\mu_q; \mu_i, \Sigma_q + \Sigma_i)$$

Higher overlap means the query bubble places more probability mass on the candidate document bubble. Wasserstein-2 remains useful as an alternate geometric probe, but it is not the canonical live retrieval rule.

---

## 7. Geodesic Interpolation

The FR geodesic between $\mathcal{N}_1$ and $\mathcal{N}_2$ at time $t \in [0,1]$:

$$\gamma(t) = \mathcal{N}\!\left((1-t)\mu_1 + t\mu_2,\ \left[(1-t)\Sigma_1^{1/2} + t\Sigma_2^{1/2}\right]^2\right)$$

This can be used to:
- Interpolate between related knowledge chunks
- Detect knowledge drift as velocity $\|\dot{\gamma}\|_{FR}$
- Define knowledge coverage as geodesic ball radius

---

## 8. Knowledge Coverage

Define $\tau$-coverage of a knowledge base $\mathcal{K} = \{\mathcal{N}_i\}$ over a query distribution $\mathcal{Q}$:

$$\text{Coverage}_\tau(\mathcal{K}, \mathcal{Q}) = \mathbb{E}_{q \sim \mathcal{Q}}\left[\mathbf{1}\left[\min_i D_{FR}(q, \mathcal{N}_i) \leq \tau\right]\right]$$

Low coverage regions → gaps in knowledge → targets for active acquisition.

---

## 9. Key Pitfalls

- Treating $\Sigma$ as scalar confidence instead of directional covariance
- Using full covariance without low-rank structure at $d \geq 384$
- Using KL for retrieval (asymmetric, undefined for non-overlapping support)
- Evaluating only on factoid benchmarks where point embeddings already saturate

---

## 10. The Geometry of Density

In Information Geometry, the Fisher-Rao metric defines distances such that contours of equal distance correspond to contours of equal probability under the appropriate probability distribution. However, the relationship between manifold density and retrieval performance is subtle. Here is the precise breakdown:

In Gaussian RAG, a more densely sampled manifold (more interpretations) creates a more precise probability field, as it captures the "true" semantic shape of the knowledge and query distribution.  

*   If density increases, the probability field becomes more detailed. This allows the system to distinguish subtle semantic variations more accurately, leading to higher recall for queries that fall near the boundaries of different conceptual regions.
