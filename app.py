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

st.markdown("""
<style>
.metric-box {
    background:#1c1c2e;border-left:4px solid #2ecc71;
    border-radius:8px;padding:14px 18px;margin:6px 0;
}
.metric-label{color:#95a5a6;font-size:.78rem;letter-spacing:.05em;text-transform:uppercase;}
.metric-value{color:#fff;font-size:1.45rem;font-weight:700;}
.metric-sub{color:#2ecc71;font-size:.82rem;margin-top:2px;}
.insight-box{
    background:#12211a;border-left:3px solid #2ecc71;
    padding:10px 16px;border-radius:0 8px 8px 0;
    color:#b8f0cc;font-size:.9rem;margin:12px 0;
}
.section-header{
    font-size:1.1rem;font-weight:600;color:#3498db;
    border-bottom:1px solid #1c2d40;padding-bottom:4px;margin:18px 0 10px 0;
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


SECTOR_ETFS = {
    "XLK": "Technology", "XLF": "Financials", "XLV": "Healthcare",
    "XLY": "Consumer Disc", "XLP": "Consumer Staples", "XLE": "Energy",
    "XLI": "Industrials", "XLB": "Materials", "XLU": "Utilities",
    "XLRE": "Real Estate", "XLC": "Comm Services",
}


# ─────────────────────────────────────────────
# CACHED COMPUTATIONS
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data(years_back: int):
    import yfinance as yf
    import time as _time
    end   = datetime.now()
    start = end - timedelta(days=years_back * 365)
    tickers = list(SECTOR_ETFS.keys())
    for attempt in range(3):
        try:
            raw = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True)
            if isinstance(raw.columns, pd.MultiIndex):
                prices = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw["Adj Close"]
            else:
                prices = raw
            prices.columns = [SECTOR_ETFS.get(c, c) for c in prices.columns]
            prices = prices.dropna()
            if len(prices) < 30:
                raise ValueError("Too few rows — possible rate limit")
            returns = np.log(prices / prices.shift(1)).dropna()
            return prices, returns, None
        except Exception as e:
            if attempt < 2:
                _time.sleep(3 * (attempt + 1))
            else:
                return None, None, str(e)

@st.cache_data(show_spinner=False)
def run_svd(_returns: pd.DataFrame, n_components: int):
    R = _returns.values
    mean = R.mean(axis=0)
    centered = R - mean
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    var = S ** 2
    evr = var / var.sum()
    projs = centered @ Vt[:n_components].T
    proj_df = pd.DataFrame(projs, index=_returns.index,
                           columns=[f"PC{i+1}" for i in range(n_components)])
    loadings = pd.DataFrame(Vt[:n_components].T, index=_returns.columns,
                            columns=[f"PC{i+1}" for i in range(n_components)])
    return evr, proj_df, loadings, S


@st.cache_data(show_spinner=False)
def compute_rolling(_returns: pd.DataFrame, window: int):
    results = []
    for i in range(window, len(_returns)):
        w = _returns.iloc[i - window:i].values
        _, S, _ = np.linalg.svd(w - w.mean(axis=0), full_matrices=False)
        v = (S ** 2) / (S ** 2).sum()
        results.append({
            "date": _returns.index[i],
            "pc1_var": v[0], "pc2_var": v[1],
            "top3_var": v[:3].sum(),
            "eff_dim": 1 / (v ** 2).sum(),
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


def animated_series(series, title, ylabel, color, fill_color, n_steps,
                    hline=None, hline_label="", chart_id="roll"):
    """Animate a time series building up left to right inside an st.empty() slot."""
    slot = st.empty()
    x_full = series.index
    y_full = series.values
    step = max(1, len(x_full) // n_steps)
    for i in range(step, len(x_full) + 1, step):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_full[:i], y=y_full[:i], mode="lines",
            line=dict(color=color, width=1.5),
            fill="tozeroy", fillcolor=fill_color,
        ))
        if hline is not None:
            fig.add_hline(y=hline,
                          line=dict(color=C["accent"], dash="dash", width=1.2),
                          annotation_text=hline_label, annotation_position="right")
        mean_val = float(y_full[:i].mean())
        fig.add_hline(y=mean_val,
                      line=dict(color=C["neutral"], dash="dot", width=1),
                      annotation_text=f"Mean: {mean_val:.2f}", annotation_position="left")
        fig.update_layout(
            template="plotly_dark", paper_bgcolor=C["bg"], plot_bgcolor=C["bg"],
            height=360, title=title,
            xaxis=dict(title="Date", gridcolor="#1e2a38"),
            yaxis=dict(title=ylabel, gridcolor="#1e2a38"),
            margin=dict(l=50, r=80, t=50, b=50),
        )
        fig.add_annotation(text=f"uid_{chart_id}_{i}", x=0, y=0, opacity=0, showarrow=False, xref="paper", yref="paper")
        slot.plotly_chart(fig, width='stretch', key=f"anim_{chart_id}_{i}")
        time.sleep(0.02)


# ═══════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════
n_steps = 60  # max animation fps

with st.sidebar:
    st.markdown("## Market Regime Detection")
    st.markdown("*SVD-based market state identification*")
    st.divider()
    page = st.radio("**Navigate**", [
        "Overview", "SVD Analysis", "Regime Map",
        "Regime Stats", "Transitions", "Rolling Analysis"
    ])
    st.divider()
    st.markdown("### Settings")
    years_back   = st.slider("Years of history", 1, 5, 3)
    n_components = st.slider("PC components", 2, 5, 3)
    roll_window  = st.slider("Rolling window (days)", 21, 126, 63)



# ═══════════════════════════════════════════════════════
# LOAD + COMPUTE (runs once, cached)
# ═══════════════════════════════════════════════════════
with st.spinner("Fetching sector ETF data…"):
    prices, returns, err = load_data(years_back)

if err or returns is None or len(returns) == 0:
    st.error(
        "Could not load market data. Yahoo Finance rate-limits cloud servers on first load. "
        "Wait 30 seconds and refresh the page."
    )
    st.stop()

evr, proj_df, loadings, S_vals = run_svd(returns, n_components)
regimes = classify_regimes(proj_df)
var_pct = evr * 100
cum_var = np.cumsum(var_pct)
roll_stats = compute_rolling(returns, roll_window)


# ═══════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ═══════════════════════════════════════════════════════
if page == "Overview":
    st.markdown("# Market Regime Detection via SVD")
    st.markdown("**Identifying distinct market states from S&P 500 sector ETF returns**")

    m1, m2, m3, m4 = st.columns(4)
    with m1: metric_card("Trading Days", f"{len(returns):,}",
                         f"{returns.index[0].date()} → {returns.index[-1].date()}")
    with m2: metric_card("Sectors", str(len(returns.columns)), "S&P 500 GICS sectors")
    with m3: metric_card("PC1 Explains", f"{var_pct[0]:.1f}%", "of cross-sector variance")
    with m4: metric_card("Top 3 PCs", f"{cum_var[2]:.1f}%", "cumulative variance")

    st.divider()
    st.markdown("### Mathematical Foundation")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(r"""
**SVD Decomposition**: $R = U \Sigma V^T$

- $U$: temporal patterns (days × days)
- $\Sigma$: singular values — strength of each mode
- $V^T$: asset loadings (modes × sectors)

Each row of $V^T$ defines a direction in sector space.
Projecting daily returns onto these directions gives a
low-dimensional coordinate for each trading day.
""")
    with col2:
        st.markdown(r"""
**Regime Classification** — quadrants of (PC1, PC2):

| Regime | PC1 | PC2 |
|--------|-----|-----|
| Risk-On Rally | ≥ med | ≥ med |
| Defensive Rally | ≥ med | < med |
| Risk-Off Stress | < med | ≥ med |
| Broad Decline | < med | < med |

PC1 ≈ market factor · PC2 ≈ growth vs. defensive rotation
""")
    st.info("Use the sidebar to explore each stage. Charts animate step-by-step.")


# ═══════════════════════════════════════════════════════
# PAGE: SVD ANALYSIS
# ═══════════════════════════════════════════════════════
elif page == "SVD Analysis":
    st.markdown("## SVD Analysis — Variance & Loadings")

    tab_var, tab_load, tab_scree = st.tabs(
        ["Variance Explained", "Asset Loadings", "Eigenvalue Spectrum"])

    with tab_var:
        st.markdown("### How much variance does each PC capture?")
        n_show = min(10, len(S_vals))
        x_vals = list(range(1, n_show + 1))
        vp = var_pct[:n_show]
        cv = cum_var[:n_show]
        col1, col2 = st.columns(2)

        with col1:
            slot_bar = st.empty()
            bar_colors = [C["primary"] if i < n_components else C["neutral"]
                          for i in range(n_show)]
            for step in range(1, n_steps + 1):
                shown = max(1, int(step / n_steps * n_show))
                fig = go.Figure(go.Bar(
                    x=x_vals[:shown], y=vp[:shown],
                    marker_color=bar_colors[:shown],
                    marker_line_color="white", marker_line_width=1,
                    text=[f"{v:.1f}%" for v in vp[:shown]],
                    textposition="outside",
                ))
                fig.update_layout(
                    template="plotly_dark", paper_bgcolor=C["bg"], plot_bgcolor=C["bg"],
                    height=350, title="Variance Explained per PC", showlegend=False,
                    xaxis=dict(title="Principal Component", gridcolor="#1e2a38", dtick=1),
                    yaxis=dict(title="Variance (%)", gridcolor="#1e2a38"),
                    margin=dict(l=40, r=20, t=50, b=40),
                )
                fig.add_annotation(text=f"uid_bar_{step}", x=0, y=0, opacity=0, showarrow=False, xref="paper", yref="paper")
                slot_bar.plotly_chart(fig, width='stretch', key=f"anim_bar_{step}")
                time.sleep(0.03)

        with col2:
            slot_cum = st.empty()
            for step in range(1, n_steps + 1):
                shown = max(1, int(step / n_steps * n_show))
                fig2 = go.Figure(go.Scatter(
                    x=x_vals[:shown], y=cv[:shown],
                    mode="lines+markers",
                    line=dict(color=C["secondary"], width=2.5),
                    marker=dict(size=8, color=C["secondary"],
                                line=dict(color="white", width=1.5)),
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
                fig2.add_annotation(text=f"uid_3_{step}", x=0, y=0, opacity=0, showarrow=False, xref="paper", yref="paper")
                slot_cum.plotly_chart(fig2, width='stretch', key=f"anim_cum_{step}")
                time.sleep(0.03)

        st.markdown(
            f'<div class="insight-box">PC1 explains {var_pct[0]:.1f}% of cross-sector variance. '
            f'Top {n_components} PCs capture {cum_var[n_components-1]:.1f}%.</div>',
            unsafe_allow_html=True)

    with tab_load:
        st.markdown("### How do sectors load onto each PC?")
        fig3 = make_subplots(rows=1, cols=n_components,
                             subplot_titles=[f"PC{i+1} ({var_pct[i]:.1f}%)"
                                             for i in range(n_components)])
        for i in range(n_components):
            col_name = f"PC{i+1}"
            vals = loadings[col_name]
            bar_c = [C["accent"] if v < 0 else C["primary"] for v in vals]
            fig3.add_trace(
                go.Bar(x=vals, y=loadings.index, orientation="h",
                       marker_color=bar_c, marker_line_color="white",
                       marker_line_width=0.5, showlegend=False),
                row=1, col=i + 1)
            fig3.add_vline(x=0, line=dict(color=C["neutral"], width=0.8), row=1, col=i + 1)
        fig3.update_layout(
            template="plotly_dark", paper_bgcolor=C["bg"], plot_bgcolor=C["bg"],
            height=420, title="Asset Loadings on Principal Components",
            margin=dict(l=80, r=20, t=70, b=40),
        )
        fig3.add_annotation(text="uid_4", x=0, y=0, opacity=0, showarrow=False, xref="paper", yref="paper")
        st.plotly_chart(fig3, width='stretch', key="loadings_chart")
        st.markdown(
            '<div class="insight-box">PC1: all sectors load same sign → market factor. '
            'PC2: Tech/Disc vs Staples/Utes → rotation factor.</div>',
            unsafe_allow_html=True)

    with tab_scree:
        st.markdown("### Singular value spectrum")
        n_show2 = min(len(S_vals), 11)
        sing_pct = (S_vals[:n_show2] ** 2) / (S_vals ** 2).sum() * 100
        fig4 = go.Figure(go.Scatter(
            x=list(range(1, n_show2 + 1)), y=sing_pct,
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
        fig4.add_annotation(text="uid_5", x=0, y=0, opacity=0, showarrow=False, xref="paper", yref="paper")
        st.plotly_chart(fig4, width='stretch', key="scree_chart")
        eff_dim = 1 / (evr ** 2).sum()
        st.markdown(
            f'<div class="insight-box">Effective dimensionality: <strong>{eff_dim:.2f}</strong> '
            f'of {len(S_vals)} possible. Low = market dominated by few factors.</div>',
            unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
# PAGE: REGIME MAP
# ═══════════════════════════════════════════════════════
elif page == "Regime Map":
    st.markdown("## Regime Map — Days in PC Space")

    tab_scatter, tab_timeline, tab_intensity = st.tabs(
        ["Regime Scatter", "Regime Timeline", "Regime Intensity"])

    with tab_scatter:
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
        fig.add_hline(y=proj_df["PC2"].median(),
                      line=dict(color=C["neutral"], dash="dash", width=1))
        fig.add_vline(x=proj_df["PC1"].median(),
                      line=dict(color=C["neutral"], dash="dash", width=1))
        fig.update_layout(
            template="plotly_dark", paper_bgcolor=C["bg"], plot_bgcolor=C["bg"],
            height=500, title="Market Days in Principal Component Space",
            xaxis=dict(title="PC1 (Market Factor)", gridcolor="#1e2a38"),
            yaxis=dict(title="PC2 (Rotation Factor)", gridcolor="#1e2a38"),
            margin=dict(l=50, r=20, t=50, b=50),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
        )
        fig.add_annotation(text="uid_6", x=0, y=0, opacity=0, showarrow=False, xref="paper", yref="paper")
        st.plotly_chart(fig, width='stretch', key="regime_scatter")

    with tab_timeline:
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
        fig2.add_annotation(text="uid_7", x=0, y=0, opacity=0, showarrow=False, xref="paper", yref="paper")
        st.plotly_chart(fig2, width='stretch', key="regime_timeline")

    with tab_intensity:
        st.markdown("Distance from origin in PC space — higher = more extreme market move.")
        dist = np.sqrt(proj_df["PC1"] ** 2 + proj_df["PC2"] ** 2)
        thresh = dist.quantile(0.95)
        x_full = dist.index
        y_full = dist.values
        step_size = max(1, len(x_full) // n_steps)
        slot = st.empty()
        for i in range(step_size, len(x_full) + 1, step_size):
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
            fig3.add_annotation(text="uid_8", x=0, y=0, opacity=0, showarrow=False, xref="paper", yref="paper")
            slot.plotly_chart(fig3, width='stretch', key=f"anim_473_{i}")
            time.sleep(0.02)


# ═══════════════════════════════════════════════════════
# PAGE: REGIME STATS
# ═══════════════════════════════════════════════════════
elif page == "Regime Stats":
    st.markdown("## Regime Characteristics")

    rows = []
    for regime in REGIME_COLORS:
        mask = regimes == regime
        rr = returns[mask]
        port = rr.mean(axis=1)
        corr_vals = rr.corr().values
        rows.append({
            "Regime":              regime,
            "Days":                int(mask.sum()),
            "Frequency (%)":       round(mask.sum() / len(regimes) * 100, 1),
            "Avg Daily Ret (bps)": round(port.mean() * 10000, 1),
            "Ann. Vol (%)":        round(port.std() * np.sqrt(252) * 100, 2),
            "Sharpe":              round(port.mean() / port.std() * np.sqrt(252), 2)
                                   if port.std() > 0 else 0,
            "Avg Corr":            round(corr_vals[np.triu_indices_from(corr_vals, 1)].mean(), 3),
            "Worst Day (%)":       round(port.min() * 100, 2),
            "Best Day (%)":        round(port.max() * 100, 2),
        })
    metrics_df = pd.DataFrame(rows).set_index("Regime")

    st.markdown('<div class="section-header">Summary</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    for i, (regime, color) in enumerate(REGIME_COLORS.items()):
        with cols[i]:
            row = metrics_df.loc[regime]
            metric_card(regime, f"{row['Frequency (%)']:.0f}% of days",
                        f"Sharpe: {row['Sharpe']:.2f}", color)

    st.markdown('<div class="section-header">Full Statistics</div>', unsafe_allow_html=True)
    st.dataframe(
        metrics_df.style.background_gradient(subset=["Sharpe"], cmap="RdYlGn"),
        width='stretch')

    st.markdown('<div class="section-header">Sector Performance by Regime</div>',
                unsafe_allow_html=True)
    fig_sectors = make_subplots(rows=2, cols=2, subplot_titles=list(REGIME_COLORS.keys()))
    for (regime, color), (r, c) in zip(REGIME_COLORS.items(), [(1,1),(1,2),(2,1),(2,2)]):
        mask = regimes == regime
        ann_ret = returns[mask].mean() * 252 * 100
        bar_c = [C["primary"] if v > 0 else C["accent"] for v in ann_ret]
        fig_sectors.add_trace(
            go.Bar(x=ann_ret.values, y=ann_ret.index, orientation="h",
                   marker_color=bar_c, marker_line_color="white",
                   marker_line_width=0.5, showlegend=False),
            row=r, col=c)
    fig_sectors.update_layout(
        template="plotly_dark", paper_bgcolor=C["bg"], plot_bgcolor=C["bg"],
        height=580, title="Annualized Sector Returns by Regime",
        margin=dict(l=100, r=20, t=70, b=40),
    )
    fig_sectors.add_annotation(text="uid_9", x=0, y=0, opacity=0, showarrow=False, xref="paper", yref="paper")
    st.plotly_chart(fig_sectors, width='stretch', key="sector_perf")


# ═══════════════════════════════════════════════════════
# PAGE: TRANSITIONS
# ═══════════════════════════════════════════════════════
elif page == "Transitions":
    st.markdown("## Regime Transition Analysis")

    tab_matrix, tab_duration = st.tabs(["Transition Matrix", "Duration Statistics"])

    with tab_matrix:
        trans = pd.crosstab(
            regimes.shift(1).dropna(), regimes.iloc[1:], normalize="index") * 100
        fig_trans = go.Figure(go.Heatmap(
            z=trans.values, x=list(trans.columns), y=list(trans.index),
            colorscale="Blues", zmin=0, zmax=100,
            text=trans.values.round(1),
            texttemplate="%{text}%", textfont=dict(size=13),
            colorbar=dict(title="Probability (%)"),
        ))
        fig_trans.update_layout(
            template="plotly_dark", paper_bgcolor=C["bg"], plot_bgcolor=C["bg"],
            height=460, title="Regime Transition Probabilities (%)",
            xaxis=dict(title="To Regime"), yaxis=dict(title="From Regime"),
            margin=dict(l=130, r=20, t=60, b=80),
        )
        fig_trans.add_annotation(text="uid_10", x=0, y=0, opacity=0, showarrow=False, xref="paper", yref="paper")
        st.plotly_chart(fig_trans, width='stretch', key="transition_heatmap")
        persistence = np.diag(trans.values).mean()
        st.markdown(
            f'<div class="insight-box">Average regime persistence: '
            f'<strong>{persistence:.1f}%</strong> per day.</div>',
            unsafe_allow_html=True)

    with tab_duration:
        dur_dict = {r: [] for r in REGIME_COLORS}
        curr = regimes.iloc[0]; cnt = 1
        for r in regimes.iloc[1:]:
            if r == curr:
                cnt += 1
            else:
                dur_dict[curr].append(cnt); curr = r; cnt = 1
        dur_dict[curr].append(cnt)

        dur_rows = []
        for regime, durs in dur_dict.items():
            if durs:
                dur_rows.append({
                    "Regime": regime, "Occurrences": len(durs),
                    "Avg Duration (days)": round(np.mean(durs), 1),
                    "Median Duration": round(np.median(durs), 1),
                    "Max Duration (days)": int(np.max(durs)),
                })
        st.dataframe(pd.DataFrame(dur_rows).set_index("Regime"), width='stretch')

        fig_box = go.Figure()
        for regime, color in REGIME_COLORS.items():
            if dur_dict[regime]:
                fig_box.add_trace(go.Box(
                    y=dur_dict[regime], name=regime,
                    marker_color=color, line_color=color, boxmean=True,
                ))
        fig_box.update_layout(
            template="plotly_dark", paper_bgcolor=C["bg"], plot_bgcolor=C["bg"],
            height=400, title="Regime Duration Distribution",
            yaxis=dict(title="Duration (days)", gridcolor="#1e2a38"),
            margin=dict(l=50, r=20, t=50, b=80),
        )
        fig_box.add_annotation(text="uid_11", x=0, y=0, opacity=0, showarrow=False, xref="paper", yref="paper")
        st.plotly_chart(fig_box, width='stretch', key="duration_box")


# ═══════════════════════════════════════════════════════
# PAGE: ROLLING ANALYSIS
# ═══════════════════════════════════════════════════════
elif page == "Rolling Analysis":
    st.markdown(f"## Rolling SVD Analysis ({roll_window}-day window)")
    st.markdown("How does market structure evolve over time?")

    tab_pc1, tab_dim, tab_top3 = st.tabs(
        ["PC1 Dominance", "Effective Dimension", "Top-3 Variance"])

    with tab_pc1:
        animated_series(
            roll_stats["pc1_var"] * 100,
            f"Rolling PC1 Variance ({roll_window}-day window)",
            "PC1 Variance (%)", C["secondary"], "rgba(52,152,219,0.15)", n_steps,
            chart_id="pc1",
        )
        st.markdown(
            '<div class="insight-box">Spikes = market stress — sectors moving together, '
            'one factor dominates. A real-time systemic risk signal.</div>',
            unsafe_allow_html=True)

    with tab_dim:
        animated_series(
            roll_stats["eff_dim"],
            f"Rolling Effective Dimensionality ({roll_window}-day window)",
            "Effective Dim", C["purple"], "rgba(155,89,182,0.15)", n_steps,
            chart_id="effdim",
        )
        st.markdown(
            '<div class="insight-box">Low values (near 1–2) = correlated, systemic market. '
            'High values = healthy diversification across sectors.</div>',
            unsafe_allow_html=True)

    with tab_top3:
        animated_series(
            roll_stats["top3_var"] * 100,
            f"Rolling Top-3 PC Variance ({roll_window}-day window)",
            "Top-3 Variance (%)", C["primary"], "rgba(46,204,113,0.15)", n_steps,
            hline=90, hline_label="90%",
            chart_id="top3",
        )
        st.markdown(
            '<div class="insight-box">When top-3 PCs explain >90% of variance, '
            'a few macro themes dominate all sector moves.</div>',
            unsafe_allow_html=True)
