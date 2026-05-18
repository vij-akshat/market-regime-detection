# Mathematical Derivations — Market Regime Detection

---

## 1. SVD Decomposition

### Setup

Given a centered returns matrix $\tilde{R} = R - \bar{R} \in \mathbb{R}^{n \times m}$ (days × assets), the SVD is:

$$\tilde{R} = U \Sigma V^T$$

where:
- $U \in \mathbb{R}^{n \times n}$ is orthogonal ($U^T U = I$) — temporal modes
- $\Sigma = \text{diag}(\sigma_1, \ldots, \sigma_r)$ with $\sigma_1 \geq \sigma_2 \geq \cdots \geq 0$
- $V \in \mathbb{R}^{m \times m}$ is orthogonal ($V^T V = I$) — asset loadings

The columns of $V$ are the eigenvectors of the sample covariance matrix $\hat{\Sigma} = \frac{1}{n-1}\tilde{R}^T \tilde{R}$, and:

$$\sigma_k^2 = (n-1) \cdot \lambda_k$$

where $\lambda_k$ are the eigenvalues of $\hat{\Sigma}$.

---

## 2. Explained Variance Ratio

The variance explained by component $k$ is:

$$\text{EVR}_k = \frac{\sigma_k^2}{\sum_{j=1}^r \sigma_j^2}$$

This equals the proportion of total variance in the centered returns matrix captured by the $k$-th principal direction. For sector ETFs, PC1 (market factor) typically explains 40–60%.

Cumulative variance through $K$ components:

$$\text{CVR}(K) = \sum_{k=1}^K \text{EVR}_k$$

---

## 3. Projection onto Principal Components

The low-dimensional representation of a return vector $r_t \in \mathbb{R}^m$ on day $t$ is:

$$\tilde{r}_t = (r_t - \bar{r}) V_K$$

where $V_K \in \mathbb{R}^{m \times K}$ contains the first $K$ right singular vectors. The scalar coordinates $(\tilde{r}_{t,1}, \tilde{r}_{t,2})$ define a point in PC space used for regime classification.

---

## 4. Regime Classification

Classification uses the sample medians of PC1 and PC2 as thresholds. For day $t$ with coordinates $(p_1, p_2)$:

| Condition | Regime |
|-----------|--------|
| $p_1 \geq \tilde{p}_1$, $p_2 \geq \tilde{p}_2$ | Risk-On Rally |
| $p_1 \geq \tilde{p}_1$, $p_2 < \tilde{p}_2$ | Defensive Rally |
| $p_1 < \tilde{p}_1$, $p_2 \geq \tilde{p}_2$ | Risk-Off Stress |
| $p_1 < \tilde{p}_1$, $p_2 < \tilde{p}_2$ | Broad Decline |

**Why medians?** The median is robust to outlier days and guarantees an equal 25% prior probability for each quadrant, avoiding degenerate regimes when returns are skewed.

---

## 5. Regime Intensity

The distance from the origin in PC space:

$$d_t = \sqrt{p_{t,1}^2 + p_{t,2}^2}$$

measures how extreme a given day's market move is relative to the average. Days in the 95th percentile of $d_t$ correspond to high-stress or high-momentum episodes.

---

## 6. Transition Matrix

Let $R_t \in \{1, 2, 3, 4\}$ be the regime on day $t$. The empirical transition matrix is:

$$\hat{P}_{ij} = \frac{\#\{t : R_t = i,\; R_{t+1} = j\}}{\#\{t : R_t = i\}}$$

This is a row-stochastic matrix ($\sum_j \hat{P}_{ij} = 1$). The diagonal entries $\hat{P}_{ii}$ measure regime **persistence** — how likely the market is to stay in the same regime tomorrow.

**Stationary distribution**: $\pi^T = \pi^T \hat{P}$, solved as the left eigenvector of $\hat{P}$ for eigenvalue 1. This gives the long-run fraction of time spent in each regime.

---

## 7. Effective Dimensionality

The **participation ratio** (inverse Simpson index) of the variance spectrum:

$$d_{\text{eff}} = \frac{1}{\sum_k \text{EVR}_k^2}$$

**Interpretation:**
- $d_{\text{eff}} = 1$: All variance in one factor (pure systemic risk)
- $d_{\text{eff}} = m$: Variance spread equally across all $m$ components (maximally diversified)

During market stress, $d_{\text{eff}}$ falls toward 1–2 as correlations spike and the market factor dominates. In calm regimes, $d_{\text{eff}}$ rises to 4–6 for a typical 11-sector universe.

---

## 8. Rolling SVD

For a rolling window of size $W$, at each time $t$ we compute:

$$\tilde{R}_{t-W:t} = U_t \Sigma_t V_t^T$$

The rolling $\text{EVR}_{k,t}$ and $d_{\text{eff},t}$ track how market structure evolves over time. A sharp rise in PC1 dominance signals increasing cross-asset correlation — a leading indicator of regime transitions toward stress.
