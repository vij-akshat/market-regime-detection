"""
Market Regime Detection — Interactive Streamlit App
SVD-based regime classification of S&P 500 sector ETFs
"""

import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Market Regime Detection",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# THEME
# ─────────────────────────────────────────────
C = {
    "primary":   "#2ecc71",
    "secondary": "#3498db",
    "accent":    "#e74c3c",
    "warn":      "#f39c12",
    "purple":    "#9b59b6",
    "neutral":   "#95a5a6",
    "bg":        "#0e1117",
    "surface":   "#1c1c2e",
}

REGIME_COLORS = {
    "Risk-On Rally":   "#2ecc71",
    "Defensive Rally": "#3498db",
    "Risk-Off Stress": "#e74c3c",
    "Broad Decline":   "#9b59b6",
}

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
.metric-box {
    background: #1c1c2e; border-left: 4px solid #2ecc71;
    border-radius: 8px; padding: 14px 18px; margin: 6px 0;
}
.metric-label { color: #95a5a6; font-size: 0.78rem; letter-spacing: 0.05em; text-transform: uppercase; }
.metric-value { color: #ffffff; font-size: 1.45rem; font-weight: 700; }
.metric-sub   { color: #2ecc71; font-size: 0.82rem; margin-top: 2px; }
.insight-box {
    background: #12211a; border-left: 3px solid #2ecc71;
    padding: 10px 16px; border-radius: 0 8px 8px 0;
    color: #b8f0cc; font-size: 0.9rem; margin: 12px 0;
}
.section-header {
    font-size: 1.1rem; font-weight: 600; color: #3498db;
    border-bottom: 1px solid #1c2d40; padding-bottom: 4px; margin: 18px 0 10px 0;
}
</style>
""", unsafe_allow_html=True)

def metric_card(label, value, sub="", color="#2ecc71"):
    st.markdown(
        f'<div class="metric-box" style="border-left-color:{color}">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        f'<div class="metric-sub" style="color:{color}">{sub}</div>'
        f'</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SECTOR ETFs
# ─────────────────────────────────────────────
SECTOR_ETFS = {
    "XLK": "Technology", "XLF": "Financials", "XLV": "Healthcare",
    "XLY": "Consumer Disc", "XLP": "Consumer Staples", "XLE": "Energy",
    "XLI": "Industrials", "XLB": "Materials", "XLU": "Utilities",
    "XLRE": "Real Estate", "XLC": "Comm Services",
}

# ─────────────────────────────────────────────
# DATA + SVD (cached)
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data(years_back: int):
    try:
        import yfinance as yf
        end   = datetime.now()
        start = end - timedelta(days=years_back * 365)
        tickers = list(SECTOR_ETFS.keys())
        raw = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True)
        if isinstance(raw.columns, pd.MultiIndex):
            prices = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw["Adj Close"]
        else:
            prices = raw
        prices.columns = [SECTOR_ETFS.get(c, c) for c in prices.columns]
        prices = prices.dropna()
        returns = np.log(prices / prices.shift(1)).dropna()
        return prices, returns, None
    except Exception as e:
        return None, None, str(e)


@st.cache_data(show_spinner=False)
def run_svd(returns_json: str, n_components: int):
    returns = pd.read_json(returns_json)
    R = returns.values
    mean = R.mean(axis=0)
    centered = R - mean
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    var     = S ** 2
    evr     = var / var.sum()
    projs   = centered @ Vt[:n_components].T
    proj_df = pd.DataFrame(projs, index=returns.index,
                           columns=[f"PC{i+1}" for i in range(n_components)])
    loadings = pd.DataFrame(Vt[:n_components].T, index=returns.columns,
                            columns=[f"PC{i+1}" for i in range(n_components)])
    return evr, proj_df, loadings, S


@st.cache_data(show_spinner=False)
def rolling_svd(returns_json: str, window: int):
    returns = pd.read_json(returns_json)
    results = []
    for i in range(window, len(returns)):
        w = returns.iloc[i-window:i].values
        _, S, _ = np.linalg.svd(w - w.mean(axis=0), full_matrices=False)
        v = (S**2) / (S**2).sum()
        results.append({
            "date": returns.index[i],
            "pc1_var": v[0], "pc2_var": v[1],
            "top3_var": v[:3].sum(),
            "eff_dim": 1 / (v**2).sum(),
        })
    return pd.DataFrame(results).set_index("date")


def classify_regimes(proj_df):
    med1 = proj_df["PC1"].median()
    med2 = proj_df["PC2"].median()
    def label(row):
        if row["PC1"] >= med1 and row["PC2"] >= med2: return "Risk-On Rally"
        if row["PC1"] >= med1 and row["PC2"] <  med2: return "Defensive Rally"
        if row["PC1"] <  med1 and row["PC2"] >= med2: return "Risk-Off Stress"
        return "Broad Decline"
    return proj_df.apply(label, axis=1)


# ═══════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📡 Market Regime Detection")
    st.markdown("*SVD-based market state identification*")
    st.divider()

    page = st.radio("**Navigate**", [
        "🏠 Overview", "📊 SVD Analysis", "🗺️ Regime Map",
        "📉 Regime Stats", "🔄 Transitions", "📈 Rolling Analysis"
    ])

    st.divider()
    st.markdown("### ⚙️ Settings")
    years_back  = st.slider("Years of history", 1, 5, 3)
    n_components = st.slider("PC components", 2, 5, 3)
    roll_window  = st.slider("Rolling window (days)", 21, 126, 63)
    n_steps      = st.slider("Animation steps", 15, 60, 35)

    st.divider()
    st.markdown("### 🔗 Resources")
    st.markdown("[Math Derivations](docs/math_derivations.md)")
    st.markdown("[Parameter Guide](docs/parameter_guide.md)")


# ═══════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════
with st.spinner("Fetching sector ETF data…"):
    prices, returns, err = load_data(years_back)

if err or returns is None:
    st.error(f"Data fetch failed: {err}. Check your internet connection.")
    st.stop()

ret_json = returns.to_json()
evr, proj_df, loadings, S_vals = run_svd(ret_json, n_components)
regimes = classify_regimes(proj_df)
var_pct  = evr * 100
cum_var  = np.cumsum(var_pct)

# Rolling (only when needed — compute upfront for responsiveness)
@st.cache_data(show_spinner=False)
def get_rolling(ret_json, window):
    return rolling_svd(ret_json, window)

roll_stats = get_rolling(ret_json, roll_window)


# ═══════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ═══════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown("# Market Regime Detection via SVD")
    st.markdown("**Identifying distinct market states from S&P 500 sector ETF returns**")

    m1, m2, m3, m4 = st.columns(4)
    with m1: metric_card("Trading Days", f"{len(returns):,}", f"{returns.index[0].date()} → {returns.index[-1].date()}")
    with m2: metric_card("Sectors", str(len(returns.columns)), "S&P 500 GICS sectors")
    with m3: metric_card("PC1 Explains", f"{var_pct[0]:.1f}%", "of cross-sector variance")
    with m4: metric_card("Top 3 PCs", f"{cum_var[2]:.1f}%", "cumulative variance")

    st.divider()
    st.markdown("### Mathematical Foundation")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(r"""
        **SVD Decomposition**:
        $$R = U \Sigma V^T$$

        - $U$: temporal patterns (days × days)
        - $\Sigma$: singular values (strength of each mode)
        - $V^T$: asset loadings (modes × sectors)

        Each row of $V^T$ defines a direction in sector space.
        Projecting daily returns onto these directions gives
        a low-dimensional coordinate for each trading day.
        """)
    with col2:
        st.markdown(r"""
        **Regime Classification**:

        Days are split into four quadrants of (PC1, PC2) space
        using median thresholds:

        | Regime | PC1 | PC2 |
        |--------|-----|-----|
        | Risk-On Rally | ≥ med | ≥ med |
        | Defensive Rally | ≥ med | < med |
        | Risk-Off Stress | < med | ≥ med |
        | Broad Decline | < med | < med |

        PC1 ≈ market factor · PC2 ≈ growth vs. defensive rotation
        """)

    st.info("Use the sidebar to explore each analysis stage. Charts animate step-by-step to show how structure emerges.")


# ═══════════════════════════════════════════════════════
# PAGE: SVD ANALYSIS
# ═══════════════════════════════════════════════════════
elif page == "📊 SVD Analysis":
    st.markdown("## SVD Analysis — Variance & Loadings")

    tab1, tab2, tab3 = st.tabs(["📊 Variance Explained", "🧬 Asset Loadings", "🔬 Eigenvalue Spectrum"])

    with tab1:
        st.markdown("### How much variance does each PC capture?")
        n_show = min(10, len(S_vals))
        x_vals = list(range(1, n_show + 1))
        vp     = var_pct[:n_show]
        cv     = cum_var[:n_show]

        col1, col2 = st.columns(2)
        with col1:
            # Animated bar chart
            placeholder = st.empty()
            colors = [C["primary"] if i < n_components else C["neutral"] for i in range(n_show)]
            for step in range(1, n_steps + 1):
                frac = step / n_steps
                shown = max(1, int(frac * n_show))
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=x_vals[:shown], y=vp[:shown],
                    marker_color=colors[:shown], marker_line_color="white",
                    marker_line_width=1, name="Variance %",
                    text=[f"{v:.1f}%" for v in vp[:shown]],
                    textposition="outside",
                ))
                fig.update_layout(
                    template="plotly_dark", paper_bgcolor=C["bg"], plot_bgcolor=C["bg"],
                    height=350, title="Variance Explained per PC",
                    xaxis=dict(title="Principal Component", gridcolor="#1e2a38", dtick=1),
                    yaxis=dict(title="Variance (%)", gridcolor="#1e2a38"),
                    margin=dict(l=40, r=20, t=50, b=40), showlegend=False,
                )
                placeholder.plotly_chart(fig, use_container_width=True)
                time.sleep(0.03)

        with col2:
            # Animated cumulative line
            placeholder2 = st.empty()
            for step in range(1, n_steps + 1):
                frac = step / n_steps
                shown = max(1, int(frac * n_show))
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=x_vals[:shown], y=cv[:shown],
                    mode="lines+markers", line=dict(color=C["secondary"], width=2.5),
                    marker=dict(size=8, color=C["secondary"], line=dict(color="white", width=1.5)),
                    fill="tozeroy", fillcolor="rgba(52,152,219,0.15)",
                ))
                fig2.add_hline(y=90, line=dict(color=C["accent"], dash="dash", width=1.5),
                               annotation_text="90%", annotation_position="right")
                fig2.update_layout(
                    template="plotly_dark", paper_bgcolor=C["bg"], plot_bgcolor=C["bg"],
                    height=350, title="Cumulative Variance Explained",
                    xaxis=dict(title="Components", gridcolor="#1e2a38", dtick=1),
                    yaxis=dict(title="Cumulative %", gridcolor="#1e2a38", range=[0, 105]),
                    margin=dict(l=40, r=20, t=50, b=40),
                )
                placeholder2.plotly_chart(fig2, use_container_width=True)
                time.sleep(0.03)

        st.markdown(
            f'<div class="insight-box">PC1 (market factor) explains {var_pct[0]:.1f}% of all cross-sector variance. '
            f'The top {n_components} PCs together capture {cum_var[n_components-1]:.1f}%.</div>',
            unsafe_allow_html=True)

    with tab2:
        st.markdown("### How do sectors load onto each PC?")
        fig3 = make_subplots(rows=1, cols=n_components,
                             subplot_titles=[f"PC{i+1} ({var_pct[i]:.1f}%)" for i in range(n_components)])
        for i in range(n_components):
            col_name = f"PC{i+1}"
            vals   = loadings[col_name]
            colors = [C["accent"] if v < 0 else C["primary"] for v in vals]
            fig3.add_trace(
                go.Bar(x=vals, y=loadings.index, orientation="h",
                       marker_color=colors, marker_line_color="white",
                       marker_line_width=0.5, name=col_name,
                       showlegend=False),
                row=1, col=i+1,
            )
            fig3.add_vline(x=0, line=dict(color=C["neutral"], width=0.8), row=1, col=i+1)
        fig3.update_layout(
            template="plotly_dark", paper_bgcolor=C["bg"], plot_bgcolor=C["bg"],
            height=420, title="Asset Loadings on Principal Components",
            margin=dict(l=80, r=20, t=70, b=40),
        )
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown(
            '<div class="insight-box">PC1: all sectors load with the same sign → market factor. '
            'PC2: growth sectors (Tech, Disc) vs defensives (Staples, Utes) → rotation factor.</div>',
            unsafe_allow_html=True)

    with tab3:
        st.markdown("### Singular value spectrum")
        n_show2 = min(len(S_vals), 11)
        sing_pct = (S_vals[:n_show2]**2) / (S_vals**2).sum() * 100
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(
            x=list(range(1, n_show2+1)), y=sing_pct,
            mode="lines+markers",
            line=dict(color=C["warn"], width=2.5),
            marker=dict(size=9, color=C["warn"], line=dict(color="white", width=1.5)),
            fill="tozeroy", fillcolor="rgba(243,156,18,0.15)",
        ))
        fig4.update_layout(
            template="plotly_dark", paper_bgcolor=C["bg"], plot_bgcolor=C["bg"],
            height=380, title="Singular Value Spectrum (Scree Plot)",
            xaxis=dict(title="Component", gridcolor="#1e2a38", dtick=1),
            yaxis=dict(title="Variance %", gridcolor="#1e2a38"),
            margin=dict(l=50, r=20, t=50, b=40),
        )
        st.plotly_chart(fig4, use_container_width=True)
        eff_dim = 1 / (evr**2).sum()
        st.markdown(
            f'<div class="insight-box">Effective dimensionality (participation ratio): '
            f'<strong>{eff_dim:.2f}</strong> out of {len(S_vals)} possible dimensions. '
            f'Low values indicate a market dominated by a few factors.</div>',
            unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
# PAGE: REGIME MAP
# ═══════════════════════════════════════════════════════
elif page == "🗺️ Regime Map":
    st.markdown("## Regime Map — Days in PC Space")

    tab1, tab2, tab3 = st.tabs(["🎨 Regime Scatter", "📅 Regime Timeline", "📏 Regime Intensity"])

    with tab1:
        st.markdown("Each dot is one trading day, colored by its classified regime.")
        fig = go.Figure()
        for regime, color in REGIME_COLORS.items():
            mask = regimes == regime
            fig.add_trace(go.Scatter(
                x=proj_df.loc[mask, "PC1"], y=proj_df.loc[mask, "PC2"],
                mode="markers", name=regime,
                marker=dict(color=color, size=5, opacity=0.65,
                            line=dict(color="white", width=0.3)),
            ))
        fig.add_hline(y=proj_df["PC2"].median(), line=dict(color=C["neutral"], dash="dash", width=1))
        fig.add_vline(x=proj_df["PC1"].median(), line=dict(color=C["neutral"], dash="dash", width=1))
        fig.update_layout(
            template="plotly_dark", paper_bgcolor=C["bg"], plot_bgcolor=C["bg"],
            height=500, title="Market Days in Principal Component Space",
            xaxis=dict(title="PC1 (Market Factor)", gridcolor="#1e2a38"),
            yaxis=dict(title="PC2 (Rotation Factor)", gridcolor="#1e2a38"),
            margin=dict(l=50, r=20, t=50, b=50),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("Regime label for each trading day over time.")
        regime_num = regimes.map({k: i for i, k in enumerate(REGIME_COLORS)})
        fig2 = go.Figure()
        for i, (regime, color) in enumerate(REGIME_COLORS.items()):
            mask = regimes == regime
            fig2.add_trace(go.Scatter(
                x=regimes.index[mask], y=regime_num[mask],
                mode="markers", name=regime,
                marker=dict(color=color, size=4, opacity=0.7),
            ))
        fig2.update_layout(
            template="plotly_dark", paper_bgcolor=C["bg"], plot_bgcolor=C["bg"],
            height=380, title="Regime Timeline",
            xaxis=dict(title="Date", gridcolor="#1e2a38"),
            yaxis=dict(tickvals=list(range(4)), ticktext=list(REGIME_COLORS.keys()),
                       gridcolor="#1e2a38"),
            margin=dict(l=130, r=20, t=50, b=50),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        st.markdown("Distance from origin in PC space — higher = more extreme market move.")
        dist = np.sqrt(proj_df["PC1"]**2 + proj_df["PC2"]**2)
        thresh = dist.quantile(0.95)

        # Animated build-up
        placeholder = st.empty()
        x_full = dist.index
        y_full = dist.values
        step = max(1, len(x_full) // n_steps)

        for i in range(step, len(x_full)+1, step):
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(
                x=x_full[:i], y=y_full[:i], mode="lines",
                line=dict(color=C["secondary"], width=1.2),
                fill="tozeroy", fillcolor="rgba(52,152,219,0.15)",
                name="Regime Intensity",
            ))
            stress_mask = y_full[:i] > thresh
            fig3.add_trace(go.Scatter(
                x=x_full[:i][stress_mask], y=y_full[:i][stress_mask],
                mode="markers", name="High Stress (>95th pctl)",
                marker=dict(color=C["accent"], size=5),
            ))
            fig3.add_hline(y=thresh, line=dict(color=C["accent"], dash="dash", width=1.2))
            fig3.update_layout(
                template="plotly_dark", paper_bgcolor=C["bg"], plot_bgcolor=C["bg"],
                height=380, title="Regime Intensity Over Time",
                xaxis=dict(title="Date", gridcolor="#1e2a38"),
                yaxis=dict(title="Distance from Origin", gridcolor="#1e2a38"),
                margin=dict(l=50, r=20, t=50, b=50),
                legend=dict(bgcolor="rgba(0,0,0,0)"),
            )
            placeholder.plotly_chart(fig3, use_container_width=True)
            time.sleep(0.02)


# ═══════════════════════════════════════════════════════
# PAGE: REGIME STATS
# ═══════════════════════════════════════════════════════
elif page == "📉 Regime Stats":
    st.markdown("## Regime Characteristics")

    # Compute metrics
    rows = []
    for regime in REGIME_COLORS:
        mask = regimes == regime
        rr   = returns[mask]
        port = rr.mean(axis=1)
        rows.append({
            "Regime":             regime,
            "Days":               int(mask.sum()),
            "Frequency (%)":      round(mask.sum() / len(regimes) * 100, 1),
            "Avg Daily Ret (bps)": round(port.mean() * 10000, 1),
            "Ann. Vol (%)":        round(port.std() * np.sqrt(252) * 100, 2),
            "Sharpe":              round(port.mean() / port.std() * np.sqrt(252), 2) if port.std() > 0 else 0,
            "Avg Corr":            round(rr.corr().values[np.triu_indices_from(rr.corr().values, 1)].mean(), 3),
            "Worst Day (%)":       round(port.min() * 100, 2),
            "Best Day (%)":        round(port.max() * 100, 2),
        })
    metrics_df = pd.DataFrame(rows).set_index("Regime")

    # Summary metrics
    st.markdown('<div class="section-header">Summary</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    for i, (regime, color) in enumerate(REGIME_COLORS.items()):
        with cols[i]:
            row = metrics_df.loc[regime]
            metric_card(regime, f"{row['Frequency (%)']:.0f}% of days",
                        f"Sharpe: {row['Sharpe']:.2f}", color)

    st.markdown('<div class="section-header">Full Statistics</div>', unsafe_allow_html=True)
    st.dataframe(metrics_df.style.background_gradient(subset=["Sharpe"], cmap="RdYlGn"),
                 use_container_width=True)

    # Sector performance per regime
    st.markdown('<div class="section-header">Sector Performance by Regime</div>', unsafe_allow_html=True)
    fig = make_subplots(rows=2, cols=2, subplot_titles=list(REGIME_COLORS.keys()))
    positions = [(1,1),(1,2),(2,1),(2,2)]
    for (regime, color), (row, col) in zip(REGIME_COLORS.items(), positions):
        mask = regimes == regime
        ann_ret = returns[mask].mean() * 252 * 100
        bar_colors = [C["primary"] if v > 0 else C["accent"] for v in ann_ret]
        fig.add_trace(go.Bar(
            x=ann_ret.values, y=ann_ret.index, orientation="h",
            marker_color=bar_colors, marker_line_color="white",
            marker_line_width=0.5, name=regime, showlegend=False,
        ), row=row, col=col)
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=C["bg"], plot_bgcolor=C["bg"],
        height=580, title="Annualized Sector Returns by Regime",
        margin=dict(l=100, r=20, t=70, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════
# PAGE: TRANSITIONS
# ═══════════════════════════════════════════════════════
elif page == "🔄 Transitions":
    st.markdown("## Regime Transition Analysis")

    tab1, tab2 = st.tabs(["🔢 Transition Matrix", "⏱️ Duration Statistics"])

    with tab1:
        trans = pd.crosstab(regimes.shift(1).dropna(), regimes.iloc[1:], normalize="index") * 100
        fig = go.Figure(go.Heatmap(
            z=trans.values, x=trans.columns, y=trans.index,
            colorscale="Blues", zmin=0, zmax=100,
            text=trans.values.round(1),
            texttemplate="%{text}%",
            textfont=dict(size=13),
            colorbar=dict(title="Probability (%)"),
        ))
        fig.update_layout(
            template="plotly_dark", paper_bgcolor=C["bg"], plot_bgcolor=C["bg"],
            height=460, title="Regime Transition Probabilities (%)",
            xaxis=dict(title="To Regime"), yaxis=dict(title="From Regime"),
            margin=dict(l=130, r=20, t=60, b=80),
        )
        st.plotly_chart(fig, use_container_width=True)
        persistence = np.diag(trans.values).mean()
        st.markdown(
            f'<div class="insight-box">Average regime persistence: <strong>{persistence:.1f}%</strong> per day. '
            f'The market tends to stay in the same regime rather than switch.</div>',
            unsafe_allow_html=True)

    with tab2:
        # Duration stats
        dur_dict = {r: [] for r in REGIME_COLORS}
        curr = regimes.iloc[0]; cnt = 1
        for r in regimes.iloc[1:]:
            if r == curr: cnt += 1
            else:
                dur_dict[curr].append(cnt); curr = r; cnt = 1
        dur_dict[curr].append(cnt)

        rows = []
        for regime, durs in dur_dict.items():
            if durs:
                rows.append({
                    "Regime": regime, "Occurrences": len(durs),
                    "Avg Duration (days)": round(np.mean(durs), 1),
                    "Median Duration": round(np.median(durs), 1),
                    "Max Duration (days)": int(np.max(durs)),
                })
        dur_df = pd.DataFrame(rows).set_index("Regime")
        st.dataframe(dur_df, use_container_width=True)

        # Duration distribution
        fig2 = go.Figure()
        for regime, color in REGIME_COLORS.items():
            if dur_dict[regime]:
                fig2.add_trace(go.Box(
                    y=dur_dict[regime], name=regime,
                    marker_color=color, line_color=color,
                    boxmean=True,
                ))
        fig2.update_layout(
            template="plotly_dark", paper_bgcolor=C["bg"], plot_bgcolor=C["bg"],
            height=400, title="Regime Duration Distribution",
            yaxis=dict(title="Duration (days)", gridcolor="#1e2a38"),
            margin=dict(l=50, r=20, t=50, b=80),
        )
        st.plotly_chart(fig2, use_container_width=True)


# ═══════════════════════════════════════════════════════
# PAGE: ROLLING ANALYSIS
# ═══════════════════════════════════════════════════════
elif page == "📈 Rolling Analysis":
    st.markdown(f"## Rolling SVD Analysis ({roll_window}-day window)")
    st.markdown("How does market structure evolve over time?")

    tab1, tab2, tab3 = st.tabs(["📡 PC1 Dominance", "📐 Effective Dimension", "🔝 Top-3 Variance"])

    def animated_roll(series, title, ylabel, color, fill_color, hline=None, hline_label=""):
        placeholder = st.empty()
        x_full = series.index
        y_full = series.values
        step   = max(1, len(x_full) // n_steps)
        for i in range(step, len(x_full)+1, step):
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=x_full[:i], y=y_full[:i], mode="lines",
                line=dict(color=color, width=1.5),
                fill="tozeroy", fillcolor=fill_color,
            ))
            if hline:
                fig.add_hline(y=hline, line=dict(color=C["accent"], dash="dash", width=1.2),
                              annotation_text=hline_label, annotation_position="right")
            mean_val = y_full[:i].mean()
            fig.add_hline(y=mean_val, line=dict(color=C["neutral"], dash="dot", width=1),
                          annotation_text=f"Mean: {mean_val:.2f}", annotation_position="left")
            fig.update_layout(
                template="plotly_dark", paper_bgcolor=C["bg"], plot_bgcolor=C["bg"],
                height=360, title=title,
                xaxis=dict(title="Date", gridcolor="#1e2a38"),
                yaxis=dict(title=ylabel, gridcolor="#1e2a38"),
                margin=dict(l=50, r=80, t=50, b=50),
            )
            placeholder.plotly_chart(fig, use_container_width=True)
            time.sleep(0.02)

    with tab1:
        animated_roll(
            roll_stats["pc1_var"] * 100,
            f"Rolling PC1 Variance ({roll_window}-day window)",
            "PC1 Variance (%)", C["secondary"], "rgba(52,152,219,0.15)",
        )
        st.markdown(
            '<div class="insight-box">Spikes indicate market stress — when all sectors '
            'move together, one factor dominates. This is a real-time risk signal.</div>',
            unsafe_allow_html=True)

    with tab2:
        animated_roll(
            roll_stats["eff_dim"],
            f"Rolling Effective Dimensionality ({roll_window}-day window)",
            "Effective Dim", C["purple"], "rgba(155,89,182,0.15)",
        )
        st.markdown(
            '<div class="insight-box">Low effective dimension (near 1–2) signals a '
            'correlated, systemic market. High values signal healthy diversification.</div>',
            unsafe_allow_html=True)

    with tab3:
        animated_roll(
            roll_stats["top3_var"] * 100,
            f"Rolling Top-3 PC Variance ({roll_window}-day window)",
            "Top-3 Variance (%)", C["primary"], "rgba(46,204,113,0.15)",
            hline=90, hline_label="90%",
        )
        st.markdown(
            '<div class="insight-box">When top-3 PCs explain >90% of variance, market '
            'dynamics are low-dimensional — a few macro themes dominate all sector moves.</div>',
            unsafe_allow_html=True)
