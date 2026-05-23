# Market Regime Detection via SVD

**Identifying distinct market states from sector ETF returns using Singular Value Decomposition**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-red)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Overview

Market regimes are persistent behavioral states where asset correlations, volatility, and return patterns exhibit consistent structure. This project uses SVD to decompose a returns matrix into principal components, project daily market data into that space, and classify each trading day into one of four regimes.

**Four detected regimes:**

| Regime | PC1 | PC2 | Interpretation |
|--------|-----|-----|----------------|
| Risk-On Rally | High | High | Broad market advance, growth leads |
| Defensive Rally | High | Low | Narrow advance, defensives lead |
| Risk-Off Stress | Low | High | Elevated volatility, rotation |
| Broad Decline | Low | Low | Correlated selloff across sectors |

---

## Mathematical Foundation -- [Full Derivations](docs/math_derivations.md) -- [Full Derivations](docs/math_derivations.md)

Given a returns matrix $R \in \mathbb{R}^{n \times m}$ (days × assets), SVD decomposes it as:

$$R = U \Sigma V^T$$

- $U$: Left singular vectors — temporal patterns
- $\Sigma$: Singular values — strength of each mode
- $V^T$: Right singular vectors — asset loadings per mode

**Explained variance** of component $k$:
$$\text{EVR}_k = \frac{\sigma_k^2}{\sum_j \sigma_j^2}$$

**Effective dimensionality** (participation ratio):
$$d_{\text{eff}} = \frac{1}{\sum_k \text{EVR}_k^2}$$

High $d_{\text{eff}}$ → diversified, sector-specific moves. Low $d_{\text{eff}}$ → market dominated by a single factor (stress regime).

---

## Project Structure

```
market-regime-detection/
│
├── app.py                  # Streamlit interactive app
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
│
└── docs/
    ├── math_derivations.md   # SVD theory and regime classification proofs
    └── parameter_guide.md    # How to calibrate window sizes and thresholds
```

---

## Quick Start

```bash
git clone https://github.com/<your-username>/market-regime-detection.git
cd market-regime-detection
pip install -r requirements.txt
streamlit run app.py
```

---

## App Features

- **Live data fetch** via `yfinance` — S&P 500 sector ETFs, configurable date range
- **Animated SVD build-up** — variance explained chart assembles step by step
- **PC space scatter** — daily returns projected onto PC1/PC2, colored by regime
- **Regime timeline** — scrollable history of regime transitions
- **Transition matrix** — heatmap of regime persistence and switching probabilities
- **Rolling analysis** — 63-day rolling PC1 dominance and effective dimensionality
- **Regime statistics** — Sharpe ratio, volatility, avg correlation per regime

---

## Key Results -- [Parameter Calibration Guide](docs/parameter_guide.md) -- [Parameter Calibration Guide](docs/parameter_guide.md)

- PC1 (market factor) typically explains **40–60%** of cross-sector variance
- Top 3 PCs capture **75–90%** of total variance
- Effective dimensionality collapses to **~2** during stress periods (2020, 2022)
- Regime persistence: diagonal of transition matrix typically **50–70%** per day

---

## Documentation

| Document | Contents |
|----------|----------|
| [Math Derivations](docs/math_derivations.md) | SVD proofs, regime classification, effective dimensionality, transition matrix, rolling SVD |
| [Parameter Guide](docs/parameter_guide.md) | Asset universe, date range, rolling window, classification thresholds |

---

## Documentation

| Document | Contents |
|----------|----------|
| [Math Derivations](docs/math_derivations.md) | SVD proofs, regime classification, effective dimensionality, transition matrix, rolling SVD |
| [Parameter Guide](docs/parameter_guide.md) | Asset universe, date range, rolling window, classification thresholds |

---

## Extensions

- **HMM**: Replace quadrant classification with Hidden Markov Model for probabilistic regime probabilities
- **Online SVD**: Streaming rank-1 updates for real-time regime detection
- **Regime-conditional strategies**: Use detected regime to weight mean-reversion vs momentum signals

---

## License

MIT License — see [LICENSE](LICENSE).
