# Parameter Calibration Guide — Market Regime Detection

---

## Asset Universe

**Default**: 11 S&P 500 sector ETFs (XLK, XLF, XLV, XLY, XLP, XLE, XLI, XLB, XLU, XLRE, XLC)

**Why sector ETFs?**
- Broad market coverage with meaningful cross-correlations
- Liquid — minimal microstructure noise in daily returns
- Regime shifts manifest clearly across sectors (e.g. energy vs tech divergence)

**Alternatives**: Country ETFs (EWJ, EWZ, FXI…) for global macro regimes; factor ETFs (MTUM, QUAL, VLUE…) for style rotation regimes.

---

## Date Range

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| Lookback | 3–5 years | Captures at least 2 full market cycles |
| Min history | 252 days | Needed for stable covariance estimation |

**Avoid**: Using all available history (15+ years) if you want the PCA to reflect current correlations. Market structure changes over time — older data can distort PC loadings.

---

## Number of Components (`n_components`)

Default: **3**

- PC1: Market factor (explains ~40–60% of variance) — all sectors move together
- PC2: Rotation factor (~10–20%) — growth vs. defensive divergence
- PC3: Idiosyncratic (~5–10%) — energy/real estate vs. rest

**How to choose**: Look at the scree plot. Choose the elbow — where the marginal variance drops sharply. For 11 sector ETFs, 3 components typically explain 70–80%.

---

## Rolling Window (`window`)

Default: **63 trading days (~3 months)**

| Window | Pros | Cons |
|--------|------|------|
| 21 days | Very reactive to recent regimes | Noisy, many false transitions |
| 63 days | Balances recency and stability | ← Recommended |
| 126 days | Stable estimates | Slow to detect genuine transitions |

**Practical rule**: Use a window equal to the half-life of the regime you're trying to detect.

---

## Regime Classification Threshold

Default: **Median split** on PC1 and PC2

This gives a 25% prior on each regime. Alternative thresholds:

- **Percentile split**: Use 33rd/67th percentiles for a 3-regime model (bull/sideways/bear)
- **K-means**: Cluster the (PC1, PC2) scatter instead of using quadrants
- **Fixed thresholds**: Normalize PC scores to z-scores; use ±0.5σ as boundaries

**When to use fixed thresholds**: If you want regime labels to be consistent across different time periods for backtesting. Median-split regimes shift as market conditions change.

---

## Regime Intensity Threshold

Default: **95th percentile** of $d_t = \sqrt{p_1^2 + p_2^2}$

This identifies the top 5% most extreme market days. Raise to 99th percentile to flag only true outlier events (e.g. COVID crash, Lehman).

---

## Interpreting PC Loadings

PC loadings (right singular vectors $V$) tell you which sectors drive each mode:

| PC | Typical pattern | Interpretation |
|----|----------------|----------------|
| PC1 | All positive, roughly equal | Systemic/market risk |
| PC2 | Tech/Disc positive, Staples/Utes negative | Growth vs. defensive rotation |
| PC3 | Energy/Materials positive, Tech negative | Commodity vs. tech divergence |

**Caution**: PC sign is arbitrary. If PC1 loadings are all negative, flip the sign — it's the same factor. What matters is the relative loading pattern, not the sign.

---

## Common Issues

**Issue**: Regime distribution is highly unequal (e.g. 60% days in one regime)
**Fix**: Switch from median split to k-means or percentile split

**Issue**: PC loadings change dramatically month to month
**Fix**: Increase the rolling window or use full-period SVD for loadings, only roll the projections

**Issue**: Too many single-day regime transitions
**Fix**: Apply a smoothing filter — e.g. only declare a regime change if the new regime persists for 3+ consecutive days
