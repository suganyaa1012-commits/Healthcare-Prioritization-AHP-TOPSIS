# ==============================
# app.py  (FINAL DEPLOYABLE VERSION)
# Healthcare Prioritization Dashboard
# Works on Streamlit Cloud
# ==============================

import io
import os
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Healthcare Prioritization Dashboard",
    page_icon="🏥",
    layout="wide"
)

# ---------------------------------------------------
# CUSTOM STYLE
# ---------------------------------------------------
st.markdown("""
<style>
.main-title{
font-size:42px;
font-weight:800;
color:#1565c0;
}
.sub{
font-size:18px;
color:#666;
}
.card{
background:#ffffff;
padding:20px;
border-radius:15px;
box-shadow:0px 4px 12px rgba(0,0,0,0.08);
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------
st.markdown('<p class="main-title">🏥 Healthcare Resource Prioritization Model</p>', unsafe_allow_html=True)
st.markdown('<p class="sub">Using AHP + TOPSIS to Rank Indian States for Healthcare Need</p>', unsafe_allow_html=True)
st.markdown("---")


# ---------------------------------------------------
# FILE LOADER
# ---------------------------------------------------
uploaded_file = st.sidebar.file_uploader("Upload Excel File", type=["xlsx"])

DEFAULT_FILE = "Healthcare_TOPSIS_Ranked_Fixed.xlsx"

if uploaded_file:
    df = pd.read_excel(uploaded_file)
elif os.path.exists(DEFAULT_FILE):
    df = pd.read_excel(DEFAULT_FILE)
else:
    st.error("Please upload your Excel dataset.")
    st.stop()

# ---------------------------------------------------
# CLEAN DATA
# ---------------------------------------------------
df.columns = df.columns.str.strip()

# if ranking already exists use it
if "Priority Index" not in df.columns:
    if "TOPSIS Score" in df.columns:
        df["Priority Index"] = 1 - df["TOPSIS Score"]
    else:
        st.error("Dataset must contain TOPSIS Score column.")
        st.stop()

if "Priority Rank" not in df.columns:
    df["Priority Rank"] = df["Priority Index"].rank(
        ascending=False, method="dense"
    ).astype(int)

df = df.sort_values("Priority Rank")

# ---------------------------------------------------
# SIDEBAR FILTERS
# ---------------------------------------------------
st.sidebar.header("Filters")

top_n = st.sidebar.slider(
    "Top N States",
    5,
    min(36, len(df)),
    10
)

selected_states = st.sidebar.multiselect(
    "Select States",
    df["State"].tolist()
)

filtered = df.copy()

if selected_states:
    filtered = filtered[filtered["State"].isin(selected_states)]

# ---------------------------------------------------
# KPI CARDS
# ---------------------------------------------------
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("States Analyzed", len(df))

with c2:
    st.metric("Top Priority State", df.iloc[0]["State"])

with c3:
    st.metric("Highest Priority Index", round(df["Priority Index"].max(), 4))

with c4:
    st.metric("Average Priority", round(df["Priority Index"].mean(), 4))

st.markdown("---")

# ---------------------------------------------------
# TABLE
# ---------------------------------------------------
st.subheader("📊 State Priority Ranking")

show_cols = [
    "Priority Rank",
    "State",
    "Priority Index",
    "TOPSIS Score"
]

extra_cols = [
    "Hospital Beds",
    "Life Expectancy",
    "Death Rate",
    "Disease Burden",
    "Population Density",
    "Poverty (Illiteracy %)"
]

for col in extra_cols:
    if col in df.columns:
        show_cols.append(col)

st.dataframe(
    filtered[show_cols].head(top_n),
    use_container_width=True
)

# ---------------------------------------------------
# BAR CHART
# ---------------------------------------------------
st.subheader("🔥 Top Priority States")

fig = px.bar(
    filtered.head(top_n).sort_values("Priority Index"),
    x="Priority Index",
    y="State",
    orientation="h",
    text="Priority Index",
    color="Priority Index",
    height=500
)

fig.update_traces(texttemplate='%{text:.3f}')
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------
# HISTOGRAM
# ---------------------------------------------------
st.subheader("📈 Priority Index Distribution")

fig2 = px.histogram(
    df,
    x="Priority Index",
    nbins=10,
    height=450
)

st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------------
# SCATTER
# ---------------------------------------------------
st.subheader("📍 Priority vs TOPSIS Score")

fig3 = px.scatter(
    df,
    x="TOPSIS Score",
    y="Priority Index",
    text="State",
    height=550
)

fig3.update_traces(textposition="top center")
st.plotly_chart(fig3, use_container_width=True)

# ---------------------------------------------------
# DOWNLOADS
# ---------------------------------------------------
st.subheader("⬇ Download Results")

csv = df.to_csv(index=False).encode("utf-8")

st.download_button(
    "Download CSV",
    csv,
    "Healthcare_Priority_Output.csv",
    "text/csv"
)

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------
st.markdown("---")
st.success(
"""
This dashboard helps policymakers identify states requiring urgent healthcare intervention using a data-driven approach.
Higher Priority Index = Higher Healthcare Need.
"""
)
