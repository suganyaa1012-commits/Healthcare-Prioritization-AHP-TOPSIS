import io
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# =========================================================
# Streamlit Dashboard: Healthcare Resource Prioritization
# AHP + TOPSIS
# =========================================================

st.set_page_config(
    page_title="Healthcare Resource Prioritization Model",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------
# Theme / styling
# -----------------------------
CUSTOM_CSS = """
<style>
    .stApp {
        background: linear-gradient(180deg, #f7fbff 0%, #eef7ff 100%);
    }
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #0f62fe 0%, #8a3ffc 45%, #fa4d56 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
    }
    .sub-title {
        font-size: 1rem;
        color: #4b5563;
        margin-top: 0;
    }
    .hero-box {
        background: linear-gradient(135deg, rgba(15,98,254,0.12), rgba(138,63,252,0.10), rgba(250,77,86,0.10));
        border: 1px solid rgba(15, 98, 254, 0.12);
        border-radius: 22px;
        padding: 1rem 1.2rem;
        box-shadow: 0 10px 30px rgba(15, 98, 254, 0.08);
    }
    .metric-card {
        background: white;
        padding: 1rem 1rem;
        border-radius: 18px;
        border: 1px solid rgba(15, 98, 254, 0.08);
        box-shadow: 0 8px 24px rgba(15, 98, 254, 0.06);
    }
    .small-note {
        color: #6b7280;
        font-size: 0.9rem;
    }
    .section-card {
        background: white;
        border-radius: 20px;
        padding: 1rem 1rem 0.7rem 1rem;
        border: 1px solid rgba(15, 98, 254, 0.08);
        box-shadow: 0 8px 24px rgba(15, 98, 254, 0.05);
        margin-bottom: 1rem;
    }
    .stDataFrame, .stPlotlyChart {
        border-radius: 18px;
        overflow: hidden;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------
# Default model configuration
# -----------------------------
CRITERIA = [
    "Hospital Beds",
    "Life Expectancy",
    "Death Rate",
    "Disease Burden",
    "Population Density",
    "Poverty (Illiteracy %)",
]
BENEFIT_CRITERIA = ["Hospital Beds", "Life Expectancy"]
COST_CRITERIA = ["Death Rate", "Disease Burden", "Population Density", "Poverty (Illiteracy %)"]

PAIRWISE_MATRIX = np.array(
    [
        [1, 0.5, 0.333, 0.2, 0.333, 0.25],
        [2, 1, 0.5, 0.333, 0.5, 0.333],
        [3, 2, 1, 0.5, 2, 1],
        [5, 3, 2, 1, 3, 2],
        [3, 2, 0.5, 0.333, 1, 0.5],
        [4, 3, 1, 0.5, 2, 1],
    ],
    dtype=float,
)


def compute_ahp_weights(pairwise: np.ndarray) -> Tuple[np.ndarray, float]:
    """Compute AHP weights using principal eigenvector and consistency ratio."""
    eigenvalues, eigenvectors = np.linalg.eig(pairwise)
    max_eig_idx = np.argmax(np.real(eigenvalues))
    principal = np.real(eigenvectors[:, max_eig_idx])
    principal = np.abs(principal)
    weights = principal / principal.sum()

    n = pairwise.shape[0]
    lambda_max = np.real(eigenvalues[max_eig_idx])
    ci = (lambda_max - n) / (n - 1)
    # Random Index for n=6
    ri = 1.24
    cr = float(ci / ri) if ri != 0 else np.nan
    return weights, cr


DEFAULT_WEIGHTS, DEFAULT_CR = compute_ahp_weights(PAIRWISE_MATRIX)


# -----------------------------
# Helper functions
# -----------------------------
def get_sample_file_path() -> str | None:
    candidates = [
        "Healthcare_TOPSIS_Formula_Backed.xlsx",
        "Healthcare_TOPSIS_Ranked_Fixed.xlsx",
        "Healthcare_TOPSIS_Cleaned_ Final.xlsx",
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def load_excel_safely(uploaded_file) -> Tuple[pd.ExcelFile, List[str]]:
    if uploaded_file is None:
        sample = get_sample_file_path()
        if sample is None:
            raise FileNotFoundError(
                "No sample file found. Please upload an Excel file with your healthcare dataset."
            )
        xls = pd.ExcelFile(sample)
        return xls, xls.sheet_names

    xls = pd.ExcelFile(uploaded_file)
    return xls, xls.sheet_names


def infer_raw_sheet(sheet_names: List[str]) -> str:
    preferred = ["Input_Data", "Sheet1", "Raw_Data", "Data", "Cleaned_Data"]
    for name in preferred:
        if name in sheet_names:
            return name
    return sheet_names[0]


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    if "State" not in df.columns:
        raise ValueError("The dataset must contain a 'State' column.")

    df["State"] = df["State"].astype(str).str.strip()
    df = df[df["State"].notna()]
    df = df[df["State"] != "nan"]
    df = df.drop_duplicates(subset=["State"], keep="first")

    missing = [c for c in CRITERIA if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing required criteria columns: " + ", ".join(missing)
        )

    for col in CRITERIA:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fill missing values with median per column for stability.
    for col in CRITERIA:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    return df.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def topsis_rank(
    df: pd.DataFrame,
    weights: np.ndarray,
    benefit_cols: List[str],
    cost_cols: List[str],
) -> pd.DataFrame:
    result = df.copy()
    x = result[CRITERIA].astype(float).to_numpy()

    # Vector normalization
    denom = np.sqrt((x ** 2).sum(axis=0))
    denom[denom == 0] = 1.0
    normalized = x / denom

    weighted = normalized * weights

    ideal_best = np.array([
        weighted[:, i].max() if CRITERIA[i] in benefit_cols else weighted[:, i].min()
        for i in range(len(CRITERIA))
    ])
    ideal_worst = np.array([
        weighted[:, i].min() if CRITERIA[i] in benefit_cols else weighted[:, i].max()
        for i in range(len(CRITERIA))
    ])

    dist_best = np.sqrt(((weighted - ideal_best) ** 2).sum(axis=1))
    dist_worst = np.sqrt(((weighted - ideal_worst) ** 2).sum(axis=1))

    closeness = np.divide(
        dist_worst,
        dist_best + dist_worst,
        out=np.zeros_like(dist_worst),
        where=(dist_best + dist_worst) != 0,
    )

    result["TOPSIS Score"] = np.round(closeness, 4)
    result["Priority Index"] = np.round(1 - closeness, 4)
    result["Priority Rank"] = result["Priority Index"].rank(
        ascending=False, method="min"
    ).astype(int)
    result["Score Rank"] = result["TOPSIS Score"].rank(
        ascending=False, method="min"
    ).astype(int)

    result = result.sort_values(["Priority Rank", "State"]).reset_index(drop=True)
    return result


def dataframe_to_excel_bytes(
    raw_df: pd.DataFrame,
    ranked_df: pd.DataFrame,
    weights: np.ndarray,
    cr: float,
) -> bytes:
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        raw_df.to_excel(writer, sheet_name="Raw_Data", index=False)
        pd.DataFrame(
            {
                "Criteria": CRITERIA,
                "Weight": weights,
                "Type": ["Benefit" if c in BENEFIT_CRITERIA else "Cost" for c in CRITERIA],
            }
        ).to_excel(writer, sheet_name="AHP_Weights", index=False)
        summary_cols = ["Priority Rank", "State", "TOPSIS Score", "Priority Index", "Score Rank"]
        ranked_df[summary_cols].to_excel(writer, sheet_name="Priority_Summary", index=False)
        ranked_df.to_excel(writer, sheet_name="Full_Ranking", index=False)
        notes = pd.DataFrame(
            {
                "Item": ["AHP Consistency Ratio", "Interpretation"],
                "Value": [round(cr, 4), "Higher Priority Index = Higher healthcare need"],
            }
        )
        notes.to_excel(writer, sheet_name="Notes", index=False)
    return bio.getvalue()


def make_metric_card(title: str, value: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div style="font-size:0.9rem;color:#6b7280;">{title}</div>
            <div style="font-size:1.7rem;font-weight:800;color:#111827;line-height:1.2;">{value}</div>
            <div style="font-size:0.85rem;color:#6b7280;">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.markdown("## 🏥 Dashboard Controls")
st.sidebar.caption("Upload your Excel file or use the sample workbook placed next to `app.py`.")
uploaded_file = st.sidebar.file_uploader(
    "Upload Excel dataset", type=["xlsx", "xls"]
)

try:
    xls, sheet_names = load_excel_safely(uploaded_file)
except Exception as e:
    st.error(str(e))
    st.stop()

selected_sheet = st.sidebar.selectbox(
    "Choose sheet", sheet_names, index=sheet_names.index(infer_raw_sheet(sheet_names))
)

use_ahp_weights = st.sidebar.checkbox("Use AHP weights", value=True)
show_manual_weights = st.sidebar.checkbox("Tune weights manually", value=False)

# Load and clean data
try:
    raw_df = pd.read_excel(xls, sheet_name=selected_sheet)
    df = clean_dataframe(raw_df)
except Exception as e:
    st.error(f"Could not read the selected sheet: {e}")
    st.stop()

# Weight selection
weights = DEFAULT_WEIGHTS.copy()
cr = DEFAULT_CR

if not use_ahp_weights:
    weights_input = []
    st.sidebar.markdown("### Manual weights")
    for i, crit in enumerate(CRITERIA):
        default_slider = float(DEFAULT_WEIGHTS[i] * 100)
        weights_input.append(
            st.sidebar.slider(
                crit,
                min_value=0.0,
                max_value=100.0,
                value=default_slider,
                step=0.5,
            )
        )
    weights = np.array(weights_input, dtype=float)
    if weights.sum() == 0:
        weights = np.ones(len(CRITERIA), dtype=float)
    weights = weights / weights.sum()
    cr = float("nan")

elif show_manual_weights:
    st.sidebar.markdown("### Weight override")
    adjusted = []
    for i, crit in enumerate(CRITERIA):
        adjusted.append(
            st.sidebar.slider(
                crit,
                min_value=0.0,
                max_value=100.0,
                value=float(DEFAULT_WEIGHTS[i] * 100),
                step=0.5,
            )
        )
    adjusted = np.array(adjusted, dtype=float)
    if adjusted.sum() > 0:
        weights = adjusted / adjusted.sum()

# Run ranking
ranked = topsis_rank(df, weights, BENEFIT_CRITERIA, COST_CRITERIA)

# Sidebar filters
st.sidebar.markdown("### Filters")
top_n = st.sidebar.slider("Show top N states", min_value=5, max_value=min(36, len(ranked)), value=min(10, len(ranked)))
selected_states = st.sidebar.multiselect(
    "Focus on specific states",
    options=ranked["State"].tolist(),
    default=[],
)

filtered_ranked = ranked.copy()
if selected_states:
    filtered_ranked = filtered_ranked[filtered_ranked["State"].isin(selected_states)]

# -----------------------------
# Main header
# -----------------------------
st.markdown('<div class="main-title">Healthcare Resource Prioritization Model</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">AHP + TOPSIS dashboard for state-level healthcare need ranking in India</div>',
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="hero-box">
        <b>Business goal:</b> identify which states should receive healthcare resources first.<br>
        <b>Logic:</b> AHP assigns importance to criteria, TOPSIS ranks states, and the <b>Priority Index</b> flips the score so higher values mean higher healthcare need.
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

# -----------------------------
# KPI row
# -----------------------------
col1, col2, col3, col4 = st.columns(4)
with col1:
    make_metric_card("States analyzed", str(len(ranked)), "Unique states in the dataset")
with col2:
    make_metric_card("Criteria used", str(len(CRITERIA)), "2 benefit + 4 cost criteria")
with col3:
    make_metric_card("AHP CR", f"{cr:.4f}" if not np.isnan(cr) else "Manual", "Consistency ratio")
with col4:
    top_state = ranked.iloc[0]["State"] if len(ranked) else "N/A"
    make_metric_card("Highest priority", top_state, "Top healthcare need")

st.write("")

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Priority Ranking",
    "🧠 Method & Weights",
    "📈 Visual Insights",
    "⬇️ Download",
])

with tab1:
    c1, c2 = st.columns([1.35, 1])

    with c1:
        st.markdown("### Priority ranking")
        display_cols = ["Priority Rank", "State", "TOPSIS Score", "Priority Index", "Score Rank"] + CRITERIA
        st.dataframe(
            filtered_ranked[display_cols].head(top_n),
            use_container_width=True,
            hide_index=True,
        )

    with c2:
        st.markdown("### Top priority states")
        top_table = ranked[["Priority Rank", "State", "Priority Index", "TOPSIS Score"]].head(top_n).copy()
        top_table["Priority Index"] = top_table["Priority Index"].round(4)
        fig = px.bar(
            top_table.sort_values("Priority Index", ascending=True),
            x="Priority Index",
            y="State",
            orientation="h",
            text="Priority Index",
            color="Priority Index",
            color_continuous_scale=["#dbeafe", "#60a5fa", "#7c3aed", "#ef4444"],
            title="Top states by healthcare need",
        )
        fig.update_layout(
            height=520,
            margin=dict(l=10, r=10, t=60, b=10),
            coloraxis_showscale=False,
            paper_bgcolor="white",
            plot_bgcolor="white",
        )
        fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Interpretation")
    st.info(
        "A higher Priority Index means the state is more likely to need urgent healthcare intervention. "
        "This happens when a state has lower infrastructure, weaker outcomes, and higher demographic or socio-economic pressure."
    )

    st.markdown("### Bottom priority states")
    bottom_cols = ["Priority Rank", "State", "Priority Index", "TOPSIS Score"]
    st.dataframe(
        ranked[bottom_cols].sort_values("Priority Rank", ascending=False).head(top_n),
        use_container_width=True,
        hide_index=True,
    )

with tab2:
    st.markdown("### Model setup")
    left, right = st.columns(2)

    with left:
        st.markdown("#### Criteria classification")
        criteria_df = pd.DataFrame(
            {
                "Criteria": CRITERIA,
                "Type": ["Benefit" if c in BENEFIT_CRITERIA else "Cost" for c in CRITERIA],
                "Weight": np.round(weights, 4),
            }
        )
        st.dataframe(criteria_df, use_container_width=True, hide_index=True)

    with right:
        st.markdown("#### AHP weights")
        ahp_df = pd.DataFrame({"Criteria": CRITERIA, "AHP Weight": np.round(weights, 4)})
        fig = px.pie(
            ahp_df,
            names="Criteria",
            values="AHP Weight",
            title="Relative importance of criteria",
            hole=0.35,
        )
        fig.update_layout(height=430, margin=dict(l=10, r=10, t=60, b=10), paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### How the ranking works")
    st.markdown(
        """
        1. Normalize every criterion so all variables are comparable.
        2. Apply AHP weights to reflect policy importance.
        3. Identify the ideal best and ideal worst state.
        4. Measure the distance of each state from both ideals.
        5. Compute the TOPSIS closeness score.
        6. Convert it to a **Priority Index = 1 - Closeness Score** so that higher values mean higher need.
        """
    )

    with st.expander("View the exact AHP pairwise matrix"):
        pairwise_df = pd.DataFrame(PAIRWISE_MATRIX, index=CRITERIA, columns=CRITERIA)
        st.dataframe(pairwise_df, use_container_width=True)

with tab3:
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### Distribution of Priority Index")
        fig = px.histogram(
            ranked,
            x="Priority Index",
            nbins=12,
            title="How healthcare need is spread across states",
        )
        fig.update_layout(height=420, paper_bgcolor="white", plot_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("### Priority Index vs TOPSIS Score")
        fig = px.scatter(
            ranked,
            x="TOPSIS Score",
            y="Priority Index",
            text="State",
            title="Inverse relationship used by the dashboard",
        )
        fig.update_traces(textposition="top center")
        fig.update_layout(height=420, paper_bgcolor="white", plot_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Ranked overview")
    ranked_chart = ranked[["State", "Priority Index"]].sort_values("Priority Index", ascending=True)
    fig = px.bar(
        ranked_chart,
        x="Priority Index",
        y="State",
        orientation="h",
        title="All states ranked by healthcare need",
    )
    fig.update_layout(height=max(550, 18 * len(ranked_chart)), paper_bgcolor="white", plot_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Data preview")
    st.dataframe(df.head(10), use_container_width=True, hide_index=True)

with tab4:
    st.markdown("### Export your results")
    excel_bytes = dataframe_to_excel_bytes(df, ranked, weights, cr)
    st.download_button(
        label="Download ranked workbook (.xlsx)",
        data=excel_bytes,
        file_name="Healthcare_TOPSIS_Ranked_Output.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

    csv_bytes = ranked.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download ranked table (.csv)",
        data=csv_bytes,
        file_name="Healthcare_TOPSIS_Ranked_Output.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.markdown("### Final ranked output")
    st.dataframe(ranked[["Priority Rank", "State", "Priority Index", "TOPSIS Score"]], use_container_width=True, hide_index=True)

    st.markdown("### Project note")
    st.success(
        "You can present this dashboard as a policy decision support tool for state-level healthcare allocation. "
        "It is transparent, reproducible, and easy to explain in a viva or report."
    )


# Footer
st.markdown("---")
st.caption(
    "Built for Applied Business Analysis: AHP weights + TOPSIS ranking + Priority Index interpretation for healthcare resource allocation."
)
