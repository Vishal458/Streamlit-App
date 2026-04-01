"""
Heart Disease Dashboard — Streamlit Version
Recreates the 2-page Power BI report with full interactivity.

Dataset: UCI Heart Failure Clinical Records (synthetic, 1,000 patients)

Install:
    pip install streamlit plotly pandas numpy

Run:
    streamlit run heart_dashboard.py
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# ──────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Dashboard",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────
# CSS / THEME
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.stApp { background: #0b0d14; color: #cdd4e4; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(170deg, #0e101a 0%, #10131f 100%);
    border-right: 1px solid #1c2235;
}
[data-testid="stSidebar"] * { color: #5a6880 !important; }

/* ── KPI Metric cards ── */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #11141f 0%, #141826 100%);
    border: 1px solid #1c2235;
    border-radius: 14px;
    padding: 20px 22px;
    box-shadow: 0 8px 40px rgba(0,0,0,0.5),
                inset 0 1px 0 rgba(255,100,100,0.06);
    position: relative;
    overflow: hidden;
}
[data-testid="metric-container"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #e84040, #ff7070);
    border-radius: 14px 14px 0 0;
}
[data-testid="metric-container"] label {
    color: #4a566a !important;
    font-size: 0.65rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-size: 1.8rem !important;
    font-weight: 700 !important;
    font-family: 'DM Mono', monospace !important;
    letter-spacing: -0.02em !important;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-size: 0.72rem !important;
}

/* ── Page title ── */
.page-title {
    font-size: 1.65rem;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: -0.03em;
    line-height: 1.2;
    margin-bottom: 3px;
}
.page-sub {
    font-size: 0.68rem;
    color: #2e3d54;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 20px;
}

/* ── Nav radio ── */
[data-testid="stRadio"] > div { gap: 5px; }
[data-testid="stRadio"] label {
    background: #11141f !important;
    border: 1px solid #1c2235 !important;
    border-radius: 9px !important;
    padding: 9px 14px !important;
    color: #4a566a !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    width: 100%;
    transition: all 0.15s;
}
[data-testid="stRadio"] label:hover {
    background: #161a28 !important;
    color: #ff7070 !important;
    border-color: #e8404044 !important;
}

/* ── Section label ── */
.sec-lbl {
    font-size: 0.6rem;
    font-weight: 700;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #e84040;
    margin-top: 24px;
    margin-bottom: 7px;
}

/* ── Chart wrapper card ── */
.chart-wrap {
    background: #11141f;
    border: 1px solid #1c2235;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 14px;
}

hr { border-color: #1c2235 !important; margin: 14px 0 !important; }
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0b0d14; }
::-webkit-scrollbar-thumb { background: #1c2235; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# DATA GENERATION  (UCI Heart Failure Clinical Records schema)
# ──────────────────────────────────────────────────────────────
@st.cache_data
def generate_data(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 1000

    # Age: right-skewed, 40–95
    age = np.clip(rng.normal(60, 13, n).astype(int), 40, 95)

    # Gender
    gender = rng.choice(["Male", "Female"], n, p=[0.65, 0.35])

    # Clinical features (correlated with age & death risk)
    diabetes          = (rng.random(n) < 0.35).astype(int)
    anaemia           = (rng.random(n) < 0.43).astype(int)
    high_blood_pressure = (rng.random(n) < 0.35).astype(int)
    smoking           = (rng.random(n) < 0.32).astype(int)

    ejection_fraction = np.clip(rng.normal(38, 12, n).astype(int), 14, 80)
    serum_creatinine  = np.clip(rng.exponential(1.4, n), 0.5, 9.4).round(1)
    serum_sodium      = np.clip(rng.normal(136, 4, n).astype(int), 113, 148)
    creatinine_phosphokinase = np.clip(rng.exponential(580, n).astype(int), 23, 7861)
    platelets         = np.clip(rng.normal(263_000, 97_000, n), 25_100, 850_000).round(-2)
    follow_up_days    = np.clip(rng.exponential(130, n).astype(int), 4, 285)

    # Death probability (logistic-like, influenced by risk factors)
    risk_score = (
        (age - 40) * 0.018
        + serum_creatinine * 0.25
        + (80 - ejection_fraction) * 0.015
        + anaemia * 0.30
        + high_blood_pressure * 0.20
        + diabetes * 0.15
        + smoking * 0.10
        + rng.normal(0, 0.35, n)
    )
    death_prob   = 1 / (1 + np.exp(-(risk_score - 2.2)))
    death_event  = (rng.random(n) < death_prob).astype(int)

    # Age Group
    def age_group(a):
        if a < 50:  return "Young (<50)"
        if a < 60:  return "Middle-Aged (50-59)"
        if a < 70:  return "Senior (60-69)"
        return "Elderly (70+)"

    age_groups = [age_group(a) for a in age]

    df = pd.DataFrame({
        "age":                    age,
        "Age Group":              age_groups,
        "Gender":                 gender,
        "DEATH_EVENT":            death_event,
        "ejection_fraction":      ejection_fraction,
        "serum_creatinine":       serum_creatinine,
        "serum_sodium":           serum_sodium,
        "creatinine_phosphokinase": creatinine_phosphokinase,
        "platelets":              platelets,
        "diabetes":               diabetes,
        "anaemia":                anaemia,
        "high_blood_pressure":    high_blood_pressure,
        "smoking":                smoking,
        "time":                   follow_up_days,
    })
    return df


df_raw = generate_data()

# Age-group ordered category
AG_ORDER = ["Young (<50)", "Middle-Aged (50-59)", "Senior (60-69)", "Elderly (70+)"]

# ──────────────────────────────────────────────────────────────
# PLOTLY THEME HELPERS
# ──────────────────────────────────────────────────────────────
BG     = "#0b0d14"
GRID   = "#171a28"
TEXT   = "#8494a8"
WHITE  = "#ffffff"
RED    = "#e84040"
RED2   = "#ff7070"
ORANGE = "#f5843c"
AMBER  = "#f5c230"
BLUE   = "#4a8cf5"
TEAL   = "#3ac4d8"
GREEN  = "#38d980"
PURPLE = "#a87cf0"
ROSE   = "#f07898"

PALETTE = [RED, BLUE, ORANGE, TEAL, AMBER, GREEN, PURPLE, ROSE,
           "#f04488", "#50b8f0", "#f0c840", "#70e090"]

GENDER_COLOR = {"Male": BLUE, "Female": ROSE}
DEATH_COLOR  = {0: GREEN, 1: RED}
AG_COLOR     = dict(zip(AG_ORDER, [TEAL, AMBER, ORANGE, RED]))


def bl(title="", h=340):
    """Base layout for all charts."""
    d = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, sans-serif", color=TEXT, size=11),
        margin=dict(l=8, r=8, t=42 if title else 14, b=10),
        height=h,
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=TEXT, size=10),
            orientation="h", y=1.08,
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
            font=dict(color=WHITE, size=13, family="DM Sans"),
            x=0.01, y=0.98,
        )
    return d


def pie_bl(title="", h=310):
    d = bl(title, h)
    d.pop("xaxis", None)
    d.pop("yaxis", None)
    return d


# ──────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div style="text-align:center;padding:26px 0 12px;">'
        '<div style="font-size:2.5rem;">🫀</div>'
        '<div style="font-size:1.05rem;font-weight:700;color:#fff;letter-spacing:-0.02em;margin-top:7px;">'
        'Heart Disease</div>'
        '<div style="font-size:0.63rem;color:#1e2840;margin-top:2px;letter-spacing:.1em;">'
        'CLINICAL ANALYTICS DASHBOARD</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    st.markdown('<div class="sec-lbl">Navigation</div>', unsafe_allow_html=True)
    page = st.radio(
        "",
        ["📊  Dashboard", "📋  Details"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown('<div class="sec-lbl">Filters</div>', unsafe_allow_html=True)

    all_genders   = sorted(df_raw["Gender"].unique().tolist())
    all_ag        = [ag for ag in AG_ORDER if ag in df_raw["Age Group"].unique()]
    all_death_lbl = {"All Patients": [0, 1], "Survived (0)": [0], "Deceased (1)": [1]}

    sel_gender = st.multiselect("Gender",          options=all_genders, default=all_genders)
    sel_ag     = st.multiselect("Age Group",       options=all_ag,      default=all_ag)
    sel_death_lbl = st.selectbox("Outcome Filter", options=list(all_death_lbl.keys()), index=0)
    sel_death  = all_death_lbl[sel_death_lbl]

    ef_min, ef_max = int(df_raw["ejection_fraction"].min()), int(df_raw["ejection_fraction"].max())
    sel_ef = st.slider("Ejection Fraction (%)", ef_min, ef_max, (ef_min, ef_max))

    st.markdown("---")
    st.markdown(
        f'<div style="font-size:0.63rem;color:#1a2435;text-align:center;">'
        f'Dataset: {len(df_raw):,} patients<br>'
        f'Heart Failure Clinical Records</div>',
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────────────────────
# APPLY FILTERS (with empty-guard)
# ──────────────────────────────────────────────────────────────
sel_gender = sel_gender or all_genders
sel_ag     = sel_ag     or all_ag

df = df_raw[
    df_raw["Gender"].isin(sel_gender) &
    df_raw["Age Group"].isin(sel_ag) &
    df_raw["DEATH_EVENT"].isin(sel_death) &
    df_raw["ejection_fraction"].between(sel_ef[0], sel_ef[1])
].copy()

if df.empty:
    st.warning("⚠️ No data matches the current filters — showing full dataset.")
    df = df_raw.copy()

# Ensure Age Group is a proper ordered category
df["Age Group"] = pd.Categorical(df["Age Group"], categories=AG_ORDER, ordered=True)

# Derived aggregation helpers
def ag_agg(d: pd.DataFrame) -> pd.DataFrame:
    """Per age-group summary."""
    grp = d.groupby("Age Group", observed=True)
    out = grp.agg(
        Total=("DEATH_EVENT", "count"),
        Deaths=("DEATH_EVENT", "sum"),
        Avg_EF=("ejection_fraction", "mean"),
        Avg_Creatinine=("serum_creatinine", "mean"),
        Diabetes_Count=("diabetes", "sum"),
        Smoking_Count=("smoking", "sum"),
        HBP_Count=("high_blood_pressure", "sum"),
        Anaemia_Count=("anaemia", "sum"),
    ).reset_index()
    out["Survival"]      = out["Total"] - out["Deaths"]
    out["Survival_Rate"] = (out["Survival"] / out["Total"] * 100).round(1)
    out["Death_Rate"]    = (out["Deaths"]   / out["Total"] * 100).round(1)
    out["Age Group"]     = pd.Categorical(out["Age Group"], categories=AG_ORDER, ordered=True)
    out = out.sort_values("Age Group")
    return out


# ══════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════
if "Dashboard" in page:
    st.markdown('<div class="page-title">Heart Disease Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">🫀 Survival · Risk Factors · Clinical Metrics</div>', unsafe_allow_html=True)

    # ── KPI Cards ─────────────────────────────────────────────
    total_pts    = len(df)
    total_death  = int(df["DEATH_EVENT"].sum())
    total_surv   = total_pts - total_death
    surv_rate    = total_surv / total_pts * 100 if total_pts else 0
    avg_age_surv = df[df["DEATH_EVENT"] == 0]["age"].mean()
    avg_ef       = df["ejection_fraction"].mean()
    avg_creat    = df["serum_creatinine"].mean()

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Survival Rate",       f"{surv_rate:.1f}%",      delta=f"{surv_rate-68:.1f}% vs baseline")
    c2.metric("Total Survival",      f"{total_surv:,}",        delta=f"{total_surv/total_pts*100:.0f}% of cohort")
    c3.metric("Total Death",         f"{total_death:,}",       delta=f"-{total_death/total_pts*100:.0f}%", delta_color="inverse")
    c4.metric("Avg Age (Survived)",  f"{avg_age_surv:.1f} yrs")
    c5.metric("Avg Ejection Frac.",  f"{avg_ef:.1f}%")
    c6.metric("Avg Serum Creatinine",f"{avg_creat:.2f} mg/dL")

    st.markdown("")
    ag = ag_agg(df)

    # ── Row 1: Two combo charts ────────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        # Combo: Deaths (bar) + Avg Serum Creatinine (line) by Age Group
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=ag["Age Group"].astype(str).tolist(),
            y=ag["Deaths"].tolist(),
            name="Death Events",
            marker_color=RED,
            opacity=0.85,
            hovertemplate="<b>%{x}</b><br>Deaths: %{y}<extra></extra>",
            yaxis="y1",
        ))
        fig.add_trace(go.Scatter(
            x=ag["Age Group"].astype(str).tolist(),
            y=ag["Avg_Creatinine"].round(2).tolist(),
            name="Avg Serum Creatinine",
            mode="lines+markers",
            line=dict(color=TEAL, width=2.5),
            marker=dict(size=8, color=TEAL, line=dict(color=BG, width=1.5)),
            hovertemplate="<b>%{x}</b><br>Avg Creatinine: %{y:.2f} mg/dL<extra></extra>",
            yaxis="y2",
        ))
        combo_l = bl("Death Event and Serum Creatinine by Age Group", 330)
        combo_l["yaxis2"] = dict(
            overlaying="y", side="right",
            gridcolor="rgba(0,0,0,0)",
            tickfont=dict(color=TEAL, size=10),
            zeroline=False,
            title_text="Avg Creatinine (mg/dL)",
            title_font=dict(color=TEAL, size=10),
        )
        fig.update_layout(**combo_l)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        # Combo: Deaths (bar) + Avg Ejection Fraction (line) by Age Group
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=ag["Age Group"].astype(str).tolist(),
            y=ag["Deaths"].tolist(),
            name="Death Events",
            marker_color=ORANGE,
            opacity=0.85,
            hovertemplate="<b>%{x}</b><br>Deaths: %{y}<extra></extra>",
            yaxis="y1",
        ))
        fig2.add_trace(go.Scatter(
            x=ag["Age Group"].astype(str).tolist(),
            y=ag["Avg_EF"].round(1).tolist(),
            name="Avg Ejection Fraction",
            mode="lines+markers",
            line=dict(color=AMBER, width=2.5),
            marker=dict(size=8, color=AMBER, line=dict(color=BG, width=1.5)),
            hovertemplate="<b>%{x}</b><br>Avg EF: %{y:.1f}%<extra></extra>",
            yaxis="y2",
        ))
        combo_l2 = bl("Death Event and Avg Ejection Fraction by Age Group", 330)
        combo_l2["yaxis2"] = dict(
            overlaying="y", side="right",
            gridcolor="rgba(0,0,0,0)",
            tickfont=dict(color=AMBER, size=10),
            zeroline=False,
            title_text="Avg Ejection Fraction (%)",
            title_font=dict(color=AMBER, size=10),
        )
        fig2.update_layout(**combo_l2)
        st.plotly_chart(fig2, use_container_width=True)

    # ── Row 2: Survival Rate line + Ribbon (grouped bar) ──────
    col_c, col_d = st.columns(2)

    with col_c:
        # Line: Survival Rate + Diabetes count by Age Group
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=ag["Age Group"].astype(str).tolist(),
            y=ag["Survival_Rate"].tolist(),
            name="Survival Rate (%)",
            mode="lines+markers",
            line=dict(color=GREEN, width=3),
            marker=dict(size=10, color=GREEN, line=dict(color=BG, width=2)),
            hovertemplate="<b>%{x}</b><br>Survival Rate: %{y:.1f}%<extra></extra>",
            yaxis="y1",
        ))
        fig3.add_trace(go.Scatter(
            x=ag["Age Group"].astype(str).tolist(),
            y=ag["Diabetes_Count"].tolist(),
            name="Diabetes Count",
            mode="lines+markers",
            line=dict(color=PURPLE, width=2, dash="dot"),
            marker=dict(size=7, color=PURPLE),
            hovertemplate="<b>%{x}</b><br>Diabetes: %{y}<extra></extra>",
            yaxis="y2",
        ))
        surv_line_l = bl("Survival Rate and No. of Diabetes by Age Group", 330)
        surv_line_l["yaxis2"] = dict(
            overlaying="y", side="right",
            gridcolor="rgba(0,0,0,0)",
            tickfont=dict(color=PURPLE, size=10),
            zeroline=False,
        )
        fig3.update_layout(**surv_line_l)
        st.plotly_chart(fig3, use_container_width=True)

    with col_d:
        # Ribbon/Grouped bar: Smoking, HBP, Anaemia, Diabetes by Age Group
        ag_str = ag["Age Group"].astype(str).tolist()
        fig4 = go.Figure()
        risk_factors = [
            ("Smoking",            "smoking",           ORANGE),
            ("High Blood Pressure","high_blood_pressure", RED2),
            ("Anaemia",            "anaemia",           AMBER),
            ("Diabetes",           "diabetes",          BLUE),
        ]
        for label, col, color in risk_factors:
            counts = [
                int(df[(df["Age Group"] == ag_val) & (df[col] == 1)].shape[0])
                for ag_val in AG_ORDER
                if ag_val in df["Age Group"].values
            ]
            visible_ags = [ag for ag in AG_ORDER if ag in df["Age Group"].values]
            fig4.add_trace(go.Bar(
                name=label,
                x=visible_ags,
                y=counts,
                marker_color=color,
                opacity=0.88,
                hovertemplate=f"<b>%{{x}}</b><br>{label}: %{{y}}<extra></extra>",
            ))
        fig4.update_layout(
            **bl("Impact of Smoking, HBP, Anaemia & Diabetes by Age Group", 330),
            barmode="group",
        )
        st.plotly_chart(fig4, use_container_width=True)

    # ── Row 3: Survival vs Death by Gender + EF distribution ──
    col_e, col_f = st.columns(2)

    with col_e:
        # Stacked bar: Survival vs Death by Gender
        gen_agg = df.groupby(["Gender", "DEATH_EVENT"]).size().reset_index(name="Count")
        gen_agg["Outcome"] = gen_agg["DEATH_EVENT"].map({0: "Survived", 1: "Deceased"})
        fig5 = px.bar(
            gen_agg, x="Gender", y="Count", color="Outcome",
            color_discrete_map={"Survived": GREEN, "Deceased": RED},
            barmode="stack",
            labels={"Count": "Patients", "Gender": ""},
        )
        fig5.update_layout(**bl("Survival vs Death by Gender", 310))
        fig5.update_traces(
            hovertemplate="<b>%{x}</b> — %{fullData.name}<br>%{y} patients<extra></extra>"
        )
        st.plotly_chart(fig5, use_container_width=True)

    with col_f:
        # Box: Ejection Fraction by Outcome
        df_box = df.copy()
        df_box["Outcome"] = df_box["DEATH_EVENT"].map({0: "Survived", 1: "Deceased"})
        fig6 = go.Figure()
        for outcome, color in [("Survived", GREEN), ("Deceased", RED)]:
            vals = df_box[df_box["Outcome"] == outcome]["ejection_fraction"].tolist()
            if vals:
                fig6.add_trace(go.Box(
                    y=vals,
                    name=outcome,
                    marker_color=color,
                    line_color=color,
                    fillcolor=color + "33",
                    boxmean="sd",
                    hovertemplate=f"<b>{outcome}</b><br>EF: %{{y}}%<extra></extra>",
                ))
        fig6.update_layout(**bl("Ejection Fraction Distribution by Outcome", 310))
        st.plotly_chart(fig6, use_container_width=True)

    # ── Row 4: Serum Creatinine vs Age scatter ─────────────────
    st.markdown('<div class="sec-lbl">Clinical Scatter — Serum Creatinine vs Age (coloured by Outcome)</div>', unsafe_allow_html=True)
    df_sc = df.copy()
    df_sc["Outcome"] = df_sc["DEATH_EVENT"].map({0: "Survived", 1: "Deceased"})
    df_sc = df_sc.sample(min(800, len(df_sc)), random_state=1)

    fig7 = go.Figure()
    for outcome, color in [("Survived", GREEN), ("Deceased", RED)]:
        sub = df_sc[df_sc["Outcome"] == outcome]
        fig7.add_trace(go.Scattergl(
            x=sub["age"].tolist(),
            y=sub["serum_creatinine"].tolist(),
            mode="markers",
            name=outcome,
            marker=dict(color=color, size=5, opacity=0.65,
                        line=dict(color=BG, width=0.5)),
            hovertemplate="Age: %{x}<br>Creatinine: %{y}<extra></extra>",
        ))
    fig7.update_layout(**bl("", 280))
    fig7.update_xaxes(title_text="Age (years)", title_font=dict(color=TEXT, size=11))
    fig7.update_yaxes(title_text="Serum Creatinine (mg/dL)", title_font=dict(color=TEXT, size=11))
    st.plotly_chart(fig7, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# PAGE 2 — DETAILS
# ══════════════════════════════════════════════════════════════
else:
    st.markdown('<div class="page-title">Patient Details & Deep-Dive</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">📋 Gender Breakdown · Age · Risk Stratification</div>', unsafe_allow_html=True)

    ag = ag_agg(df)

    # ── KPIs ──────────────────────────────────────────────────
    total_pts    = len(df)
    total_death  = int(df["DEATH_EVENT"].sum())
    total_surv   = total_pts - total_death
    surv_rate    = total_surv / total_pts * 100 if total_pts else 0
    avg_age_surv = df[df["DEATH_EVENT"] == 0]["age"].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Patients",     f"{total_pts:,}")
    c2.metric("Total Survival",     f"{total_surv:,}")
    c3.metric("Total Death",        f"{total_death:,}", delta_color="inverse")
    c4.metric("Avg Age (Survived)", f"{avg_age_surv:.1f} yrs")

    st.markdown("")

    # ── Row 1: Donut + Area ───────────────────────────────────
    col_a, col_b = st.columns([1, 1.4])

    with col_a:
        # Donut: Total Death by Age Group
        death_by_ag = df[df["DEATH_EVENT"] == 1]["Age Group"].value_counts().reset_index()
        death_by_ag.columns = ["Age Group", "Deaths"]
        death_by_ag["Age Group"] = pd.Categorical(
            death_by_ag["Age Group"], categories=AG_ORDER, ordered=True
        )
        death_by_ag = death_by_ag.sort_values("Age Group")

        fig8 = go.Figure(go.Pie(
            labels=death_by_ag["Age Group"].astype(str).tolist(),
            values=death_by_ag["Deaths"].tolist(),
            hole=0.55,
            marker=dict(colors=[AG_COLOR.get(ag, RED) for ag in death_by_ag["Age Group"].astype(str)]),
            textinfo="percent+label",
            textfont=dict(color=TEXT, size=10),
            hovertemplate="<b>%{label}</b><br>%{value} deaths (%{percent})<extra></extra>",
        ))
        fig8.add_annotation(
            text=f"<b>{total_death}</b><br><span style='font-size:10px'>Deaths</span>",
            x=0.5, y=0.5,
            font=dict(color=WHITE, size=16, family="DM Mono"),
            showarrow=False,
        )
        fig8.update_layout(**pie_bl("Total Death by Age Group", 340))
        st.plotly_chart(fig8, use_container_width=True)

    with col_b:
        # Area: Total Survival by age (binned)
        surv_df = df[df["DEATH_EVENT"] == 0].copy()
        age_bins = list(range(40, 100, 5))
        surv_df["Age Bin"] = pd.cut(
            surv_df["age"], bins=age_bins,
            labels=[f"{i}-{i+4}" for i in age_bins[:-1]],
            right=False,
        )
        surv_age = surv_df.groupby("Age Bin", observed=True).size().reset_index(name="Survivals")
        surv_age["Age Bin"] = surv_age["Age Bin"].astype(str)

        fig9 = go.Figure(go.Scatter(
            x=surv_age["Age Bin"].tolist(),
            y=surv_age["Survivals"].tolist(),
            mode="lines+markers",
            fill="tozeroy",
            fillcolor=GREEN + "22",
            line=dict(color=GREEN, width=2.5),
            marker=dict(size=7, color=GREEN, line=dict(color=BG, width=1.5)),
            hovertemplate="Age %{x}<br>Survivals: %{y}<extra></extra>",
        ))
        fig9.update_layout(**bl("Total Survival by Age", 340))
        fig9.update_xaxes(tickangle=-35, tickfont=dict(size=9))
        st.plotly_chart(fig9, use_container_width=True)

    # ── Row 2: Stacked area (Death Event by Gender) + HBP/Smoke bar ─
    col_c, col_d = st.columns(2)

    with col_c:
        # Stacked area: Cumulative death count by gender across age
        death_df = df[df["DEATH_EVENT"] == 1].copy()
        death_df["Age Bin"] = pd.cut(
            death_df["age"], bins=list(range(40, 100, 5)),
            labels=[f"{i}-{i+4}" for i in range(40, 95, 5)],
            right=False,
        )
        death_gen = (
            death_df.groupby(["Age Bin", "Gender"], observed=True)
            .size()
            .reset_index(name="Deaths")
        )
        death_gen["Age Bin"] = death_gen["Age Bin"].astype(str)

        fig10 = go.Figure()
        for gender, color in [("Male", BLUE), ("Female", ROSE)]:
            sub = death_gen[death_gen["Gender"] == gender]
            if not sub.empty:
                fig10.add_trace(go.Scatter(
                    x=sub["Age Bin"].tolist(),
                    y=sub["Deaths"].tolist(),
                    name=gender,
                    mode="lines",
                    stackgroup="one",
                    line=dict(color=color, width=1.5),
                    fillcolor=color + "55",
                    hovertemplate=f"<b>{gender}</b><br>Age %{{x}}<br>Deaths: %{{y}}<extra></extra>",
                ))
        fig10.update_layout(**bl("Death Events by Gender & Age Band", 330))
        fig10.update_xaxes(tickangle=-35, tickfont=dict(size=9))
        st.plotly_chart(fig10, use_container_width=True)

    with col_d:
        # Grouped bar: Risk factor prevalence by Gender
        risk_gen = df.groupby("Gender").agg(
            Smoking=("smoking", "mean"),
            HBP=("high_blood_pressure", "mean"),
            Anaemia=("anaemia", "mean"),
            Diabetes=("diabetes", "mean"),
        ).reset_index()
        risk_melt = risk_gen.melt(id_vars="Gender", var_name="Risk Factor", value_name="Prevalence")
        risk_melt["Prevalence"] = (risk_melt["Prevalence"] * 100).round(1)

        fig11 = px.bar(
            risk_melt, x="Risk Factor", y="Prevalence", color="Gender",
            barmode="group",
            color_discrete_map={"Male": BLUE, "Female": ROSE},
            text=risk_melt["Prevalence"].apply(lambda x: f"{x:.0f}%"),
            labels={"Prevalence": "Prevalence (%)", "Risk Factor": ""},
        )
        fig11.update_traces(
            textposition="outside",
            textfont=dict(color=TEXT, size=9),
            hovertemplate="<b>%{x}</b> (%{fullData.name})<br>%{y:.1f}%<extra></extra>",
        )
        fig11.update_layout(**bl("Risk Factor Prevalence by Gender", 330))
        st.plotly_chart(fig11, use_container_width=True)

    # ── Row 3: Serum Sodium + Survival Rate heatmap ───────────
    col_e, col_f = st.columns(2)

    with col_e:
        # Histogram: Serum Sodium by Outcome
        df_hs = df.copy()
        df_hs["Outcome"] = df_hs["DEATH_EVENT"].map({0: "Survived", 1: "Deceased"})
        fig12 = go.Figure()
        for outcome, color in [("Survived", GREEN), ("Deceased", RED)]:
            vals = df_hs[df_hs["Outcome"] == outcome]["serum_sodium"].tolist()
            if vals:
                fig12.add_trace(go.Histogram(
                    x=vals, name=outcome,
                    marker_color=color, opacity=0.72,
                    nbinsx=25,
                    hovertemplate=f"<b>{outcome}</b><br>Na: %{{x}} mEq/L<br>Count: %{{y}}<extra></extra>",
                ))
        fig12.update_layout(**bl("Serum Sodium Distribution by Outcome", 310), barmode="overlay")
        fig12.update_xaxes(title_text="Serum Sodium (mEq/L)", title_font=dict(color=TEXT, size=10))
        st.plotly_chart(fig12, use_container_width=True)

    with col_f:
        # Bar: Survival Rate by Age Group × Gender (grouped)
        ag_gen = (
            df.groupby(["Age Group", "Gender"], observed=True)
            .apply(lambda g: pd.Series({
                "Survival_Rate": (1 - g["DEATH_EVENT"].mean()) * 100,
                "Count": len(g),
            }))
            .reset_index()
        )
        ag_gen["Age Group"] = pd.Categorical(ag_gen["Age Group"], categories=AG_ORDER, ordered=True)
        ag_gen = ag_gen.sort_values("Age Group")

        fig13 = px.bar(
            ag_gen, x="Age Group", y="Survival_Rate", color="Gender",
            barmode="group",
            color_discrete_map={"Male": BLUE, "Female": ROSE},
            text=ag_gen["Survival_Rate"].apply(lambda x: f"{x:.0f}%"),
            labels={"Survival_Rate": "Survival Rate (%)", "Age Group": ""},
        )
        fig13.update_traces(
            textposition="outside",
            textfont=dict(color=TEXT, size=9),
            hovertemplate="<b>%{x}</b> (%{fullData.name})<br>Survival: %{y:.1f}%<extra></extra>",
        )
        fig13.update_yaxes(range=[0, 115])
        fig13.update_layout(**bl("Survival Rate by Age Group & Gender", 310))
        st.plotly_chart(fig13, use_container_width=True)

    # ── Pivot Table ────────────────────────────────────────────
    st.markdown('<div class="sec-lbl">Pivot Table — Gender × Age Group × Clinical Metrics</div>', unsafe_allow_html=True)

    pivot = (
        df.groupby(["Gender", "Age Group"], observed=True)
        .agg(
            Total_Patients=("DEATH_EVENT", "count"),
            Total_Death=("DEATH_EVENT", "sum"),
            Total_Survival=("DEATH_EVENT", lambda x: (x == 0).sum()),
            Survival_Rate=("DEATH_EVENT", lambda x: f"{(1 - x.mean()) * 100:.1f}%"),
            Avg_Age_Survival=("age", lambda x: f"{x[df.loc[x.index,'DEATH_EVENT']==0].mean():.1f}" if (df.loc[x.index,"DEATH_EVENT"]==0).any() else "—"),
            Avg_EF=("ejection_fraction", lambda x: f"{x.mean():.1f}%"),
            Avg_Creatinine=("serum_creatinine", lambda x: f"{x.mean():.2f}"),
        )
        .reset_index()
    )
    pivot["Age Group"] = pd.Categorical(pivot["Age Group"], categories=AG_ORDER, ordered=True)
    pivot = pivot.sort_values(["Gender", "Age Group"])
    pivot.columns = [
        "Gender", "Age Group", "Total Patients",
        "Deaths", "Survivals", "Survival Rate",
        "Avg Age (Survived)", "Avg EF (%)", "Avg Creatinine",
    ]
    st.dataframe(pivot, use_container_width=True, hide_index=True)

    # ── Full patient table ─────────────────────────────────────
    st.markdown('<div class="sec-lbl">Patient-Level Data</div>', unsafe_allow_html=True)
    disp = df[[
        "age", "Age Group", "Gender", "DEATH_EVENT",
        "ejection_fraction", "serum_creatinine", "serum_sodium",
        "diabetes", "anaemia", "high_blood_pressure", "smoking", "time",
    ]].copy()
    disp["DEATH_EVENT"] = disp["DEATH_EVENT"].map({0: "✅ Survived", 1: "❌ Deceased"})
    disp.columns = [
        "Age", "Age Group", "Gender", "Outcome",
        "Ejection Frac. (%)", "Serum Creatinine", "Serum Sodium",
        "Diabetes", "Anaemia", "High BP", "Smoking", "Follow-Up (days)",
    ]
    st.dataframe(disp.head(200), use_container_width=True, hide_index=True)


# ──────────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div style="text-align:center;font-size:0.63rem;color:#1a2235;padding:6px;">'
    'Heart Disease Dashboard &nbsp;·&nbsp; Built with Streamlit &amp; Plotly &nbsp;·&nbsp; '
    '1,000 synthetic patients &nbsp;·&nbsp; UCI Heart Failure Clinical Records schema'
    '</div>',
    unsafe_allow_html=True,
)
