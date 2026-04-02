"""
Heart Disease Clinical Dashboard  ·  Streamlit + Plotly
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Recreates the 2-page Power BI report:
  Page 1 — Dashboard (KPIs, combo charts, ribbon, line, slicers)
  Page 2 — Details   (pivot, donut, area, stacked-area, slicer)

Run:
    pip install streamlit plotly pandas numpy
    streamlit run heart_dashboard.py
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# ══════════════════════════════════════════════════════════════
# 0 · PAGE CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Heart Disease Dashboard",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════
# 1 · GLOBAL CSS
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background: #0d1117;
    color: #c9d1d9;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(175deg, #0d1117 0%, #161b22 100%);
    border-right: 1px solid #21262d;
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div {
    color: #7d8590 !important;
    font-size: 0.78rem !important;
}

/* ── Selectbox / dropdown ── */
[data-testid="stSelectbox"] > div > div {
    background: #161b22 !important;
    border: 1px solid #21262d !important;
    border-radius: 8px !important;
    color: #c9d1d9 !important;
}

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #161b22 0%, #1c2128 100%);
    border: 1px solid #21262d;
    border-top: 3px solid #1f6feb;
    border-radius: 12px;
    padding: 18px 20px 14px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
}
[data-testid="metric-container"] label {
    color: #484f58 !important;
    font-size: 0.65rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #e6edf3 !important;
    font-size: 1.65rem !important;
    font-weight: 700 !important;
    font-family: 'JetBrains Mono', monospace !important;
    letter-spacing: -0.02em !important;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-size: 0.7rem !important;
}

/* ── Section divider ── */
.section-block {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 18px 20px 14px 20px;
    margin-bottom: 16px;
}
.section-title {
    font-size: 0.6rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #58a6ff;
    margin-bottom: 14px;
    padding-bottom: 8px;
    border-bottom: 1px solid #21262d;
}
.page-title {
    font-size: 1.55rem;
    font-weight: 700;
    color: #e6edf3;
    letter-spacing: -0.025em;
    margin-bottom: 2px;
}
.page-sub {
    font-size: 0.68rem;
    color: #30363d;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 20px;
}

/* ── Nav radio ── */
[data-testid="stRadio"] > div { gap: 5px; }
[data-testid="stRadio"] label {
    background: #161b22 !important;
    border: 1px solid #21262d !important;
    border-radius: 8px !important;
    padding: 9px 14px !important;
    color: #484f58 !important;
    font-size: 0.76rem !important;
    font-weight: 600 !important;
    width: 100%;
    transition: all 0.15s;
}
[data-testid="stRadio"] label:hover {
    background: #1c2128 !important;
    color: #58a6ff !important;
    border-color: #1f6feb55 !important;
}

/* ── HR ── */
hr { border-color: #21262d !important; margin: 12px 0 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #21262d; border-radius: 3px; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# 2 · DATA GENERATION  (UCI Heart Failure schema, 1 000 patients)
# ══════════════════════════════════════════════════════════════
@st.cache_data
def make_data(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n   = 1000

    age        = np.clip(rng.normal(60, 13, n).astype(int), 40, 95)
    gender     = rng.choice(["Male", "Female"], n, p=[0.65, 0.35])
    diabetes   = (rng.random(n) < 0.35).astype(int)
    anaemia    = (rng.random(n) < 0.43).astype(int)
    hbp        = (rng.random(n) < 0.35).astype(int)
    smoking    = (rng.random(n) < 0.32).astype(int)
    ef         = np.clip(rng.normal(38, 12, n).astype(int), 14, 80)
    creatinine = np.clip(rng.exponential(1.4, n), 0.5, 9.4).round(1)
    sodium     = np.clip(rng.normal(136, 4, n).astype(int), 113, 148)
    cpk        = np.clip(rng.exponential(580, n).astype(int), 23, 7861)
    platelets  = np.clip(rng.normal(263_000, 97_000, n), 25_100, 850_000).round(-2)
    follow_up  = np.clip(rng.exponential(130, n).astype(int), 4, 285)

    risk  = ((age - 40) * 0.018 + creatinine * 0.25 + (80 - ef) * 0.015
             + anaemia * 0.30 + hbp * 0.20 + diabetes * 0.15
             + smoking * 0.10 + rng.normal(0, 0.35, n))
    prob  = 1 / (1 + np.exp(-(risk - 2.2)))
    death = (rng.random(n) < prob).astype(int)

    def ag(a):
        if a < 50: return "Young (<50)"
        if a < 60: return "Middle-Aged (50–59)"
        if a < 70: return "Senior (60–69)"
        return "Elderly (70+)"

    AG_ORDER = ["Young (<50)", "Middle-Aged (50–59)", "Senior (60–69)", "Elderly (70+)"]

    df = pd.DataFrame({
        "age": age,
        "Age Group": pd.Categorical([ag(a) for a in age], categories=AG_ORDER, ordered=True),
        "Gender": gender,
        "DEATH_EVENT": death,
        "ejection_fraction": ef,
        "serum_creatinine": creatinine,
        "serum_sodium": sodium,
        "creatinine_phosphokinase": cpk,
        "platelets": platelets,
        "diabetes": diabetes,
        "anaemia": anaemia,
        "high_blood_pressure": hbp,
        "smoking": smoking,
        "time": follow_up,
    })
    return df


DF_RAW   = make_data()
AG_ORDER = ["Young (<50)", "Middle-Aged (50–59)", "Senior (60–69)", "Elderly (70+)"]

# ══════════════════════════════════════════════════════════════
# 3 · COLOUR TOKENS  (navy/steel palette, minimal red)
# ══════════════════════════════════════════════════════════════
BG      = "#0d1117"
SURFACE = "#161b22"
GRID    = "#1c2128"
TEXT    = "#7d8590"
WHITE   = "#e6edf3"

# Primary accent — blue family
BLUE    = "#58a6ff"
NAVY    = "#1f6feb"
SKY     = "#79c0ff"
CYAN    = "#39d353"

# Secondary accents — no dominant red
TEAL    = "#2dd4bf"
GREEN   = "#3fb950"
AMBER   = "#e3b341"
ORANGE  = "#f0883e"
PURPLE  = "#bc8cff"
INDIGO  = "#6e76ae"
ROSE    = "#ff7b72"      # rose (used sparingly, not dominant)
SLATE   = "#8b949e"

# Outcome colours — NOT harsh red
SURV_C  = "#3fb950"      # green  = survived
DEAD_C  = "#e3b341"      # amber  = deceased  (avoids full-red dominance)

PALETTE = [BLUE, TEAL, AMBER, GREEN, PURPLE, ORANGE, SKY, ROSE,
           INDIGO, CYAN, "#f0c0ff", "#70d8ff"]


def hex_rgba(hex_color: str, alpha: float) -> str:
    """Convert a '#rrggbb' hex string to 'rgba(r,g,b,alpha)'."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

AG_COLOR = {
    "Young (<50)":         BLUE,
    "Middle-Aged (50–59)": TEAL,
    "Senior (60–69)":      AMBER,
    "Elderly (70+)":       ORANGE,
}
GEN_COLOR = {"Male": BLUE, "Female": PURPLE}


# ══════════════════════════════════════════════════════════════
# 4 · LAYOUT HELPERS
# ══════════════════════════════════════════════════════════════
def _base(title: str = "", h: int = 340) -> dict:
    d = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=TEXT, size=11),
        margin=dict(l=10, r=10, t=46 if title else 16, b=10),
        height=h,
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=SLATE, size=10),
            orientation="h",
            y=1.10, x=0,
        ),
        xaxis=dict(
            gridcolor=GRID, linecolor=GRID, tickcolor=GRID,
            tickfont=dict(color=TEXT, size=10),
            showgrid=True, zeroline=False,
        ),
        yaxis=dict(
            gridcolor=GRID, linecolor=GRID, tickcolor=GRID,
            tickfont=dict(color=TEXT, size=10),
            showgrid=True, zeroline=False,
        ),
    )
    if title:
        d["title"] = dict(
            text=title,
            font=dict(color=WHITE, size=12, family="Inter"),
            x=0.0, y=0.99, xanchor="left", yanchor="top",
        )
    return d


def _pie(title: str = "", h: int = 310) -> dict:
    d = _base(title, h)
    d.pop("xaxis", None)
    d.pop("yaxis", None)
    return d


def section(label: str):
    """Render a styled section header."""
    st.markdown(
        f'<div class="section-title">{label}</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════
# 5 · SIDEBAR  (dropdown slicers)
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        '<div style="text-align:center;padding:24px 0 10px;">'
        '<div style="font-size:2.4rem;">🫀</div>'
        '<div style="font-size:1rem;font-weight:700;color:#e6edf3;'
        'letter-spacing:-0.02em;margin-top:7px;">Heart Disease</div>'
        '<div style="font-size:0.6rem;color:#21262d;margin-top:3px;'
        'letter-spacing:.1em;">CLINICAL ANALYTICS</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # ── Navigation ──────────────────────────────────────
    page = st.radio(
        "Navigation",
        ["📊  Dashboard", "📋  Details"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(
        '<div style="font-size:0.6rem;font-weight:700;letter-spacing:.18em;'
        'text-transform:uppercase;color:#58a6ff;margin-bottom:10px;">Filters</div>',
        unsafe_allow_html=True,
    )

    # ── Dropdown slicers ────────────────────────────────
    gender_opts  = ["All Genders"] + sorted(DF_RAW["Gender"].unique().tolist())
    ag_opts      = ["All Age Groups"] + AG_ORDER
    outcome_opts = ["All Patients", "Survived", "Deceased"]
    ef_opts      = ["All EF Ranges", "Low (< 30%)", "Medium (30–50%)", "High (> 50%)"]

    sel_gender  = st.selectbox("Gender",              gender_opts,  index=0)
    sel_ag      = st.selectbox("Age Group",           ag_opts,      index=0)
    sel_outcome = st.selectbox("Outcome",             outcome_opts, index=0)
    sel_ef      = st.selectbox("Ejection Fraction",   ef_opts,      index=0)

    st.markdown("---")
    st.markdown(
        f'<div style="font-size:0.62rem;color:#21262d;text-align:center;">'
        f'Dataset · {len(DF_RAW):,} patients<br>'
        f'UCI Heart Failure Clinical Records</div>',
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════════
# 6 · APPLY FILTERS
# ══════════════════════════════════════════════════════════════
df = DF_RAW.copy()

if sel_gender != "All Genders":
    df = df[df["Gender"] == sel_gender]

if sel_ag != "All Age Groups":
    df = df[df["Age Group"] == sel_ag]

if sel_outcome == "Survived":
    df = df[df["DEATH_EVENT"] == 0]
elif sel_outcome == "Deceased":
    df = df[df["DEATH_EVENT"] == 1]

if sel_ef == "Low (< 30%)":
    df = df[df["ejection_fraction"] < 30]
elif sel_ef == "Medium (30–50%)":
    df = df[df["ejection_fraction"].between(30, 50)]
elif sel_ef == "High (> 50%)":
    df = df[df["ejection_fraction"] > 50]

# Safety guard — never show empty
if df.empty:
    st.warning("⚠️ No records match the current filters — showing full dataset.")
    df = DF_RAW.copy()

# Ensure ordered category
df["Age Group"] = pd.Categorical(df["Age Group"], categories=AG_ORDER, ordered=True)


# ══════════════════════════════════════════════════════════════
# 7 · SHARED AGGREGATION
# ══════════════════════════════════════════════════════════════
def ag_summary(d: pd.DataFrame) -> pd.DataFrame:
    """Per-age-group summary used across multiple charts."""
    out = (
        d.groupby("Age Group", observed=True)
        .agg(
            Total          = ("DEATH_EVENT", "count"),
            Deaths         = ("DEATH_EVENT", "sum"),
            Avg_EF         = ("ejection_fraction", "mean"),
            Avg_Creatinine = ("serum_creatinine", "mean"),
            Avg_Sodium     = ("serum_sodium", "mean"),
            Diabetes       = ("diabetes", "sum"),
            Smoking        = ("smoking", "sum"),
            HBP            = ("high_blood_pressure", "sum"),
            Anaemia        = ("anaemia", "sum"),
        )
        .reset_index()
    )
    out["Survival"]      = out["Total"] - out["Deaths"]
    out["Survival_Rate"] = (out["Survival"] / out["Total"] * 100).round(1)
    out["Death_Rate"]    = (out["Deaths"]   / out["Total"] * 100).round(1)
    out["Age Group"]     = pd.Categorical(out["Age Group"], categories=AG_ORDER, ordered=True)
    return out.sort_values("Age Group")


# ══════════════════════════════════════════════════════════════
# ██████████  PAGE 1 — DASHBOARD  ██████████████████████████████
# ══════════════════════════════════════════════════════════════
if "Dashboard" in page:

    st.markdown('<div class="page-title">Heart Disease Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">🫀 Clinical Overview · Survival · Risk Factors</div>', unsafe_allow_html=True)

    ag = ag_summary(df)
    ag_x = ag["Age Group"].astype(str).tolist()

    total   = len(df)
    deaths  = int(df["DEATH_EVENT"].sum())
    survs   = total - deaths
    s_rate  = survs / total * 100 if total else 0
    avg_age_s = df[df["DEATH_EVENT"] == 0]["age"].mean() if survs else 0
    avg_ef    = df["ejection_fraction"].mean()
    avg_creat = df["serum_creatinine"].mean()

    # ── ① KPI CARDS ────────────────────────────────────────────
    st.markdown('<div class="section-block">', unsafe_allow_html=True)
    section("📌 Key Performance Indicators")
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Survival Rate",       f"{s_rate:.1f}%")
    k2.metric("Total Survival",      f"{survs:,}")
    k3.metric("Total Death",         f"{deaths:,}")
    k4.metric("Avg Age — Survived",  f"{avg_age_s:.1f} yrs")
    k5.metric("Avg Ejection Frac.",  f"{avg_ef:.1f}%")
    k6.metric("Avg Serum Creatinine",f"{avg_creat:.2f} mg/dL")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("")

    # ── ② COMBO CHARTS ─────────────────────────────────────────
    st.markdown('<div class="section-block">', unsafe_allow_html=True)
    section("📈 Death Event vs Clinical Markers by Age Group")
    cc1, cc2 = st.columns(2)

    with cc1:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Death Events",
            x=ag_x, y=ag["Deaths"].tolist(),
            marker_color=AMBER, opacity=0.80,
            hovertemplate="<b>%{x}</b><br>Deaths: %{y}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            name="Avg Serum Creatinine",
            x=ag_x, y=ag["Avg_Creatinine"].round(2).tolist(),
            mode="lines+markers", yaxis="y2",
            line=dict(color=TEAL, width=2.5),
            marker=dict(size=8, color=TEAL, line=dict(color=BG, width=2)),
            hovertemplate="<b>%{x}</b><br>Creatinine: %{y:.2f}<extra></extra>",
        ))
        lo = _base("Death Event and Serum Creatinine by Age Group", 320)
        lo["yaxis2"] = dict(
            overlaying="y", side="right",
            tickfont=dict(color=TEAL, size=10),
            gridcolor="rgba(0,0,0,0)", zeroline=False,
            title=dict(text="mg/dL", font=dict(color=TEAL, size=10)),
        )
        fig.update_layout(**lo)
        st.plotly_chart(fig, use_container_width=True)

    with cc2:
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            name="Death Events",
            x=ag_x, y=ag["Deaths"].tolist(),
            marker_color=INDIGO, opacity=0.80,
            hovertemplate="<b>%{x}</b><br>Deaths: %{y}<extra></extra>",
        ))
        fig2.add_trace(go.Scatter(
            name="Avg Ejection Fraction",
            x=ag_x, y=ag["Avg_EF"].round(1).tolist(),
            mode="lines+markers", yaxis="y2",
            line=dict(color=SKY, width=2.5),
            marker=dict(size=8, color=SKY, line=dict(color=BG, width=2)),
            hovertemplate="<b>%{x}</b><br>EF: %{y:.1f}%<extra></extra>",
        ))
        lo2 = _base("Death Event and Avg Ejection Fraction by Age Group", 320)
        lo2["yaxis2"] = dict(
            overlaying="y", side="right",
            tickfont=dict(color=SKY, size=10),
            gridcolor="rgba(0,0,0,0)", zeroline=False,
            title=dict(text="%", font=dict(color=SKY, size=10)),
        )
        fig2.update_layout(**lo2)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ── ③ SURVIVAL RATE LINE  +  RIBBON / GROUPED BAR ──────────
    st.markdown('<div class="section-block">', unsafe_allow_html=True)
    section("📉 Survival Rate · Risk Factor Impact by Age Group")
    rc1, rc2 = st.columns(2)

    with rc1:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            name="Survival Rate (%)",
            x=ag_x, y=ag["Survival_Rate"].tolist(),
            mode="lines+markers",
            line=dict(color=GREEN, width=3),
            marker=dict(size=10, color=GREEN, line=dict(color=BG, width=2)),
            hovertemplate="<b>%{x}</b><br>Survival Rate: %{y:.1f}%<extra></extra>",
        ))
        fig3.add_trace(go.Scatter(
            name="Diabetes Count",
            x=ag_x, y=ag["Diabetes"].tolist(),
            mode="lines+markers", yaxis="y2",
            line=dict(color=PURPLE, width=2, dash="dot"),
            marker=dict(size=7, color=PURPLE),
            hovertemplate="<b>%{x}</b><br>Diabetes: %{y}<extra></extra>",
        ))
        lo3 = _base("Survival Rate and No. of Diabetes by Age Group", 320)
        lo3["yaxis2"] = dict(
            overlaying="y", side="right",
            tickfont=dict(color=PURPLE, size=10),
            gridcolor="rgba(0,0,0,0)", zeroline=False,
        )
        fig3.update_layout(**lo3)
        st.plotly_chart(fig3, use_container_width=True)

    with rc2:
        # Grouped bar — Smoking, HBP, Anaemia, Diabetes by Age Group
        rf_data = {
            "Smoking":            (ag["Smoking"].tolist(),  ORANGE),
            "High Blood Pressure":(ag["HBP"].tolist(),      BLUE),
            "Anaemia":            (ag["Anaemia"].tolist(),   TEAL),
            "Diabetes":           (ag["Diabetes"].tolist(),  PURPLE),
        }
        fig4 = go.Figure()
        for label, (vals, color) in rf_data.items():
            fig4.add_trace(go.Bar(
                name=label, x=ag_x, y=vals,
                marker_color=color, opacity=0.85,
                hovertemplate=f"<b>%{{x}}</b><br>{label}: %{{y}}<extra></extra>",
            ))
        fig4.update_layout(
            **_base("Impact of Smoking, HBP, Anaemia & Diabetes by Age Group", 320),
            barmode="group",
        )
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ── ④ GENDER BREAKDOWN  +  EF DISTRIBUTION ─────────────────
    st.markdown('<div class="section-block">', unsafe_allow_html=True)
    section("👥 Gender Breakdown · Ejection Fraction Distribution")
    gc1, gc2 = st.columns(2)

    with gc1:
        gen_grp = (
            df.groupby(["Gender", "DEATH_EVENT"])
            .size()
            .reset_index(name="Count")
        )
        gen_grp["Outcome"] = gen_grp["DEATH_EVENT"].map({0: "Survived", 1: "Deceased"})
        fig5 = go.Figure()
        for outcome, color in [("Survived", SURV_C), ("Deceased", DEAD_C)]:
            sub = gen_grp[gen_grp["Outcome"] == outcome]
            if not sub.empty:
                fig5.add_trace(go.Bar(
                    name=outcome,
                    x=sub["Gender"].tolist(),
                    y=sub["Count"].tolist(),
                    marker_color=color,
                    opacity=0.88,
                    hovertemplate=f"<b>%{{x}}</b> — {outcome}<br>%{{y}} patients<extra></extra>",
                ))
        fig5.update_layout(
            **_base("Survival vs Death by Gender", 310),
            barmode="stack",
        )
        st.plotly_chart(fig5, use_container_width=True)

    with gc2:
        fig6 = go.Figure()
        for outcome, color in [("Survived", SURV_C), ("Deceased", DEAD_C)]:
            mask = df["DEATH_EVENT"] == (1 if outcome == "Deceased" else 0)
            vals = df.loc[mask, "ejection_fraction"].tolist()
            if vals:
                fig6.add_trace(go.Box(
                    name=outcome,
                    y=vals,
                    marker_color=color,
                    line_color=color,
                    fillcolor=hex_rgba(color, 0.16),
                    boxmean="sd",
                    hovertemplate=f"<b>{outcome}</b><br>EF: %{{y}}%<extra></extra>",
                ))
        fig6.update_layout(**_base("Ejection Fraction Distribution by Outcome", 310))
        fig6.update_yaxes(title_text="Ejection Fraction (%)",
                          title_font=dict(color=TEXT, size=10))
        st.plotly_chart(fig6, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ── ⑤ CLINICAL SCATTER ─────────────────────────────────────
    st.markdown('<div class="section-block">', unsafe_allow_html=True)
    section("🔬 Clinical Scatter — Serum Creatinine vs Age")

    sample = df.sample(min(800, len(df)), random_state=1)
    fig7   = go.Figure()
    for outcome, color in [("Survived", SURV_C), ("Deceased", DEAD_C)]:
        mask = sample["DEATH_EVENT"] == (1 if outcome == "Deceased" else 0)
        sub  = sample[mask]
        if not sub.empty:
            fig7.add_trace(go.Scattergl(
                name=outcome,
                x=sub["age"].tolist(),
                y=sub["serum_creatinine"].tolist(),
                mode="markers",
                marker=dict(color=color, size=5, opacity=0.6,
                            line=dict(color=BG, width=0.5)),
                hovertemplate="Age: %{x}<br>Creatinine: %{y} mg/dL<extra></extra>",
            ))
    fig7.update_layout(**_base("", 300))
    fig7.update_xaxes(title_text="Age (years)",
                      title_font=dict(color=TEXT, size=11))
    fig7.update_yaxes(title_text="Serum Creatinine (mg/dL)",
                      title_font=dict(color=TEXT, size=11))
    st.plotly_chart(fig7, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# ██████████  PAGE 2 — DETAILS  ████████████████████████████████
# ══════════════════════════════════════════════════════════════
else:

    st.markdown('<div class="page-title">Patient Details & Deep Dive</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">📋 Gender Breakdown · Age Analysis · Risk Stratification</div>', unsafe_allow_html=True)

    ag = ag_summary(df)

    total  = len(df)
    deaths = int(df["DEATH_EVENT"].sum())
    survs  = total - deaths
    s_rate = survs / total * 100 if total else 0
    avg_age_s = df[df["DEATH_EVENT"] == 0]["age"].mean() if survs else 0

    # ── ① KPIs ─────────────────────────────────────────────────
    st.markdown('<div class="section-block">', unsafe_allow_html=True)
    section("📌 Key Performance Indicators")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Patients",      f"{total:,}")
    k2.metric("Total Survival",      f"{survs:,}")
    k3.metric("Total Death",         f"{deaths:,}")
    k4.metric("Avg Age — Survived",  f"{avg_age_s:.1f} yrs")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("")

    # ── ② DONUT  +  AREA ───────────────────────────────────────
    st.markdown('<div class="section-block">', unsafe_allow_html=True)
    section("🍩 Total Death by Age Group  ·  Total Survival by Age")
    dc1, dc2 = st.columns([1, 1.4])

    with dc1:
        death_ag = (
            df[df["DEATH_EVENT"] == 1]["Age Group"]
            .value_counts()
            .reindex(AG_ORDER)
            .dropna()
            .reset_index()
        )
        death_ag.columns = ["Age Group", "Deaths"]
        labels = death_ag["Age Group"].astype(str).tolist()
        values = death_ag["Deaths"].tolist()
        colors = [AG_COLOR.get(l, SLATE) for l in labels]

        fig8 = go.Figure(go.Pie(
            labels=labels,
            values=values,
            hole=0.58,
            marker=dict(colors=colors),
            textinfo="percent+label",
            textfont=dict(color=WHITE, size=10),
            hovertemplate="<b>%{label}</b><br>%{value} deaths (%{percent})<extra></extra>",
            insidetextorientation="horizontal",
        ))
        fig8.add_annotation(
            text=f"<b>{deaths}</b><br><span style='font-size:9px;color:{TEXT}'>Deaths</span>",
            x=0.5, y=0.5,
            font=dict(color=WHITE, size=18, family="JetBrains Mono"),
            showarrow=False,
        )
        fig8.update_layout(**_pie("Total Death by Age Group", 330))
        st.plotly_chart(fig8, use_container_width=True)

    with dc2:
        surv_df = df[df["DEATH_EVENT"] == 0].copy()
        bins    = list(range(40, 100, 5))
        labels_bins = [f"{i}–{i+4}" for i in bins[:-1]]
        surv_df["Age Bin"] = pd.cut(
            surv_df["age"], bins=bins, labels=labels_bins, right=False
        )
        surv_age = (
            surv_df.groupby("Age Bin", observed=True)
            .size()
            .reset_index(name="Survivals")
        )
        surv_age["Age Bin"] = surv_age["Age Bin"].astype(str)

        fig9 = go.Figure(go.Scatter(
            x=surv_age["Age Bin"].tolist(),
            y=surv_age["Survivals"].tolist(),
            mode="lines+markers",
            fill="tozeroy",
            fillcolor=hex_rgba(GREEN, 0.12),
            line=dict(color=GREEN, width=2.5),
            marker=dict(size=7, color=GREEN, line=dict(color=BG, width=1.5)),
            hovertemplate="Age %{x}<br>Survivals: %{y}<extra></extra>",
        ))
        fig9.update_layout(**_base("Total Survival by Age", 330))
        fig9.update_xaxes(tickangle=-35, tickfont=dict(size=9),
                          title_text="Age Band",
                          title_font=dict(color=TEXT, size=10))
        fig9.update_yaxes(title_text="Survivors",
                          title_font=dict(color=TEXT, size=10))
        st.plotly_chart(fig9, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ── ③ STACKED AREA  +  RISK BY GENDER ──────────────────────
    st.markdown('<div class="section-block">', unsafe_allow_html=True)
    section("📊 Death Events by Gender  ·  Risk Factor Prevalence")
    sc1, sc2 = st.columns(2)

    with sc1:
        death_df = df[df["DEATH_EVENT"] == 1].copy()
        bins     = list(range(40, 100, 5))
        lbls     = [f"{i}–{i+4}" for i in bins[:-1]]
        death_df["Age Bin"] = pd.cut(
            death_df["age"], bins=bins, labels=lbls, right=False
        )
        death_gen = (
            death_df.groupby(["Age Bin", "Gender"], observed=True)
            .size()
            .reset_index(name="Deaths")
        )
        death_gen["Age Bin"] = death_gen["Age Bin"].astype(str)

        fig10 = go.Figure()
        for gender, color in [("Male", BLUE), ("Female", PURPLE)]:
            sub = death_gen[death_gen["Gender"] == gender]
            if not sub.empty:
                fig10.add_trace(go.Scatter(
                    name=gender,
                    x=sub["Age Bin"].tolist(),
                    y=sub["Deaths"].tolist(),
                    mode="lines",
                    stackgroup="one",
                    line=dict(color=color, width=1.5),
                    fillcolor=hex_rgba(color, 0.27),
                    hovertemplate=f"<b>{gender}</b><br>Age %{{x}}<br>Deaths: %{{y}}<extra></extra>",
                ))
        fig10.update_layout(**_base("Death Events by Gender & Age Band", 320))
        fig10.update_xaxes(tickangle=-35, tickfont=dict(size=9))
        st.plotly_chart(fig10, use_container_width=True)

    with sc2:
        risk_gen = (
            df.groupby("Gender")[["smoking", "high_blood_pressure", "anaemia", "diabetes"]]
            .mean()
            .mul(100)
            .round(1)
            .reset_index()
        )
        risk_melt = risk_gen.melt(
            id_vars="Gender", var_name="Risk Factor", value_name="Prevalence (%)"
        )
        rf_labels = {
            "smoking":            "Smoking",
            "high_blood_pressure":"High BP",
            "anaemia":            "Anaemia",
            "diabetes":           "Diabetes",
        }
        risk_melt["Risk Factor"] = risk_melt["Risk Factor"].map(rf_labels)

        fig11 = go.Figure()
        for gender, color in [("Male", BLUE), ("Female", PURPLE)]:
            sub = risk_melt[risk_melt["Gender"] == gender]
            if not sub.empty:
                fig11.add_trace(go.Bar(
                    name=gender,
                    x=sub["Risk Factor"].tolist(),
                    y=sub["Prevalence (%)"].tolist(),
                    marker_color=color,
                    opacity=0.88,
                    text=[f"{v:.0f}%" for v in sub["Prevalence (%)"]],
                    textposition="outside",
                    textfont=dict(color=WHITE, size=9),
                    hovertemplate=f"<b>%{{x}}</b> ({gender})<br>%{{y:.1f}}%<extra></extra>",
                ))
        fig11.update_layout(
            **_base("Risk Factor Prevalence by Gender (%)", 320),
            barmode="group",
        )
        fig11.update_yaxes(range=[0, 65])
        st.plotly_chart(fig11, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ── ④ SODIUM HISTOGRAM  +  SURVIVAL RATE BY AG × GENDER ────
    st.markdown('<div class="section-block">', unsafe_allow_html=True)
    section("💊 Serum Sodium Distribution  ·  Survival Rate by Age Group & Gender")
    hc1, hc2 = st.columns(2)

    with hc1:
        fig12 = go.Figure()
        for outcome, color in [("Survived", SURV_C), ("Deceased", DEAD_C)]:
            mask = df["DEATH_EVENT"] == (1 if outcome == "Deceased" else 0)
            vals = df.loc[mask, "serum_sodium"].tolist()
            if vals:
                fig12.add_trace(go.Histogram(
                    name=outcome, x=vals,
                    marker_color=color, opacity=0.70,
                    nbinsx=25,
                    hovertemplate=f"<b>{outcome}</b><br>Na: %{{x}} mEq/L<br>Count: %{{y}}<extra></extra>",
                ))
        fig12.update_layout(**_base("Serum Sodium Distribution by Outcome", 310), barmode="overlay")
        fig12.update_xaxes(title_text="Serum Sodium (mEq/L)",
                           title_font=dict(color=TEXT, size=10))
        st.plotly_chart(fig12, use_container_width=True)

    with hc2:
        ag_gen = (
            df.groupby(["Age Group", "Gender"], observed=True)
            .apply(lambda g: (1 - g["DEATH_EVENT"].mean()) * 100)
            .reset_index(name="Survival_Rate")
        )
        ag_gen["Age Group"] = pd.Categorical(ag_gen["Age Group"], categories=AG_ORDER, ordered=True)
        ag_gen = ag_gen.sort_values("Age Group")

        fig13 = go.Figure()
        for gender, color in [("Male", BLUE), ("Female", PURPLE)]:
            sub = ag_gen[ag_gen["Gender"] == gender]
            if not sub.empty:
                fig13.add_trace(go.Bar(
                    name=gender,
                    x=sub["Age Group"].astype(str).tolist(),
                    y=sub["Survival_Rate"].round(1).tolist(),
                    marker_color=color,
                    opacity=0.88,
                    text=[f"{v:.0f}%" for v in sub["Survival_Rate"].round(1)],
                    textposition="outside",
                    textfont=dict(color=WHITE, size=9),
                    hovertemplate=f"<b>%{{x}}</b> ({gender})<br>Survival: %{{y:.1f}}%<extra></extra>",
                ))
        fig13.update_layout(
            **_base("Survival Rate by Age Group & Gender", 310),
            barmode="group",
        )
        fig13.update_yaxes(range=[0, 120])
        st.plotly_chart(fig13, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ── ⑤ PIVOT TABLE ──────────────────────────────────────────
    st.markdown('<div class="section-block">', unsafe_allow_html=True)
    section("📑 Pivot Table — Gender × Age Group × Clinical Metrics")

    pivot = (
        df.groupby(["Gender", "Age Group"], observed=True)
        .agg(
            Patients   = ("DEATH_EVENT", "count"),
            Deaths     = ("DEATH_EVENT", "sum"),
            Survivals  = ("DEATH_EVENT", lambda x: (x == 0).sum()),
            Surv_Rate  = ("DEATH_EVENT", lambda x: round((1 - x.mean()) * 100, 1)),
            Avg_Age    = ("age", "mean"),
            Avg_EF     = ("ejection_fraction", "mean"),
            Avg_Creat  = ("serum_creatinine", "mean"),
            Avg_Sodium = ("serum_sodium", "mean"),
        )
        .reset_index()
    )
    pivot["Age Group"] = pd.Categorical(pivot["Age Group"], categories=AG_ORDER, ordered=True)
    pivot = pivot.sort_values(["Gender", "Age Group"]).reset_index(drop=True)
    pivot["Surv_Rate"] = pivot["Surv_Rate"].apply(lambda x: f"{x:.1f}%")
    pivot["Avg_Age"]   = pivot["Avg_Age"].round(1)
    pivot["Avg_EF"]    = pivot["Avg_EF"].round(1).apply(lambda x: f"{x}%")
    pivot["Avg_Creat"] = pivot["Avg_Creat"].round(2).apply(lambda x: f"{x} mg/dL")
    pivot["Avg_Sodium"]= pivot["Avg_Sodium"].round(1).apply(lambda x: f"{x} mEq/L")
    pivot.columns = [
        "Gender", "Age Group", "Patients", "Deaths", "Survivals",
        "Survival Rate", "Avg Age", "Avg EF", "Avg Creatinine", "Avg Sodium",
    ]
    st.dataframe(pivot, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── ⑥ PATIENT-LEVEL TABLE ──────────────────────────────────
    st.markdown('<div class="section-block">', unsafe_allow_html=True)
    section("🗃️ Patient-Level Records (top 300)")

    disp = df[[
        "age", "Age Group", "Gender", "DEATH_EVENT",
        "ejection_fraction", "serum_creatinine", "serum_sodium",
        "diabetes", "anaemia", "high_blood_pressure", "smoking", "time",
    ]].head(300).copy()
    disp["DEATH_EVENT"] = disp["DEATH_EVENT"].map({0: "✅ Survived", 1: "⚠️ Deceased"})
    disp.columns = [
        "Age", "Age Group", "Gender", "Outcome",
        "Ejection Frac (%)", "Serum Creatinine", "Serum Sodium",
        "Diabetes", "Anaemia", "High BP", "Smoking", "Follow-Up (days)",
    ]
    st.dataframe(disp, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(
    '<div style="text-align:center;font-size:0.6rem;color:#1c2128;padding:6px 0 10px;">'
    'Heart Disease Clinical Dashboard &nbsp;·&nbsp; Streamlit + Plotly &nbsp;·&nbsp; '
    '1,000 synthetic patients &nbsp;·&nbsp; UCI Heart Failure Clinical Records schema'
    '</div>',
    unsafe_allow_html=True,
)
