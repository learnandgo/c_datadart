import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Page config
st.set_page_config(
    page_title="DataDart",
    page_icon="🎯",
    layout="wide"
)

# Custom CSS for highlighted section headings
st.markdown("""
<style>
.section-header {
    background: linear-gradient(90deg, #1a4a7a 0%, #1a6aaa 100%);
    color: white;
    padding: 10px 20px;
    border-radius: 8px;
    font-size: 18px;
    font-weight: 600;
    margin: 20px 0 10px 0;
    letter-spacing: 0.5px;
}
.metric-card {
    background-color: #f0f7ff;
    border-left: 4px solid #4a90d9;
    padding: 10px 16px;
    border-radius: 6px;
    margin-bottom: 8px;
}
</style>
""", unsafe_allow_html=True)

BLUE = "#4a90d9"
BLUE_SCALE = ["#d0e8ff", "#4a90d9", "#1a4a7a"]

def section_header(title: str):
    st.markdown(f'<div class="section-header">📌 {title}</div>', unsafe_allow_html=True)

# ── Title ──────────────────────────────────────────────────────────────────────
st.title("🎯 DataDart")
st.caption("Upload your data. Get instant insights.")
st.divider()

# ── File uploader ──────────────────────────────────────────────────────────────
file = st.file_uploader(
    "Upload your data file",
    type=["csv", "xlsx", "xls"],
    help="Supported formats: CSV, Excel"
)

if file:

    # Load
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    numeric_df  = df.select_dtypes(include="number")
    char_df     = df.select_dtypes(include="object")
    numeric_cols = numeric_df.columns.tolist()
    char_cols    = char_df.columns.tolist()

    # ── Overall Table Summary ──────────────────────────────────────────────────
    section_header("Overall Table Summary")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Rows",             f"{df.shape[0]:,}")
    c2.metric("Columns",          df.shape[1])
    c3.metric("Numeric cols",     len(numeric_cols))
    c4.metric("Categorical cols", len(char_cols))
    c5.metric("Missing values",   f"{df.isnull().sum().sum():,}")

    st.success(f"✅ **{file.name}** loaded successfully!")

    # ── Data Preview ──────────────────────────────────────────────────────────
    section_header("Data Preview")
    n_rows = st.slider("Rows to preview", 5, min(500, df.shape[0]), 50, step=5)
    st.dataframe(df.head(n_rows), use_container_width=True)

    # ── Numeric Column Statistics ──────────────────────────────────────────────
    section_header("Numeric Column Statistics")

    if not numeric_df.empty:

        # Build extended stats table including median and mode
        desc = numeric_df.describe().T
        desc["median"] = numeric_df.median()
        desc["mode"]   = numeric_df.mode().iloc[0]
        desc["skew"]   = numeric_df.skew()

        # Reorder columns nicely
        desc = desc[["count", "mean", "median", "mode",
                     "std", "min", "25%", "75%", "max", "skew"]]
        desc.columns = ["Count", "Mean", "Median", "Mode",
                        "Std Dev", "Min", "25%", "75%", "Max", "Skew"]

        st.dataframe(desc.style.format("{:.2f}"), use_container_width=True)
    else:
        st.info("No numeric columns found.")

    # ── Visual Insights Dashboard ──────────────────────────────────────────────
    section_header("Visual Insights Dashboard")

    if not numeric_df.empty:

        # ── Categorical filter ─────────────────────────────────────────────────
        filtered_df = df.copy()

        if char_cols:
            st.markdown("**Filter data by categorical field:**")
            filter_col1, filter_col2 = st.columns([1, 3])
            with filter_col1:
                filter_cat = st.selectbox(
                    "Category column",
                    ["None"] + char_cols,
                    key="filter_cat"
                )
            with filter_col2:
                if filter_cat != "None":
                    options = df[filter_cat].dropna().unique().tolist()
                    selected_vals = st.multiselect(
                        f"Filter by {filter_cat}",
                        options,
                        default=options,
                        key="filter_vals"
                    )
                    if selected_vals:
                        filtered_df = df[df[filter_cat].isin(selected_vals)]
                    st.caption(f"Showing {len(filtered_df):,} of {len(df):,} rows")

        st.divider()

        # ── Chart grid ────────────────────────────────────────────────────────
        # Histograms — 2 per row
        st.markdown("##### Distributions")
        num_charts = len(numeric_cols)
        cols_per_row = 2
        rows_needed  = -(-num_charts // cols_per_row)  # ceiling division

        for row in range(rows_needed):
            grid_cols = st.columns(cols_per_row)
            for col_idx in range(cols_per_row):
                chart_idx = row * cols_per_row + col_idx
                if chart_idx >= num_charts:
                    break
                col_name = numeric_cols[chart_idx]
                with grid_cols[col_idx]:
                    fig = px.histogram(
                        filtered_df,
                        x=col_name,
                        title=f"{col_name}",
                        color_discrete_sequence=[BLUE],
                        nbins=30
                    )
                    fig.add_vline(
                        x=filtered_df[col_name].median(),
                        line_dash="dash",
                        line_color="#1a4a7a",
                        annotation_text="Median",
                        annotation_position="top right"
                    )
                    fig.update_layout(
                        margin=dict(l=10, r=10, t=40, b=20),
                        height=280,
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # Box plots — 2 per row
        st.markdown("##### Box Plots")
        for row in range(rows_needed):
            grid_cols = st.columns(cols_per_row)
            for col_idx in range(cols_per_row):
                chart_idx = row * cols_per_row + col_idx
                if chart_idx >= num_charts:
                    break
                col_name = numeric_cols[chart_idx]
                with grid_cols[col_idx]:
                    fig = px.box(
                        filtered_df,
                        y=col_name,
                        title=f"{col_name}",
                        color_discrete_sequence=[BLUE],
                        points="outliers"
                    )
                    fig.update_layout(
                        margin=dict(l=10, r=10, t=40, b=20),
                        height=280
                    )
                    st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # ── Correlation heatmap ────────────────────────────────────────────────
        if len(numeric_cols) >= 2:
            st.markdown("##### Correlations Between Numeric Variables")

            corr = filtered_df[numeric_cols].corr().round(2)

            fig_corr = px.imshow(
                corr,
                text_auto=True,
                color_continuous_scale=["#ffffff", "#4a90d9", "#1a4a7a"],
                zmin=-1, zmax=1,
                title="Correlation Matrix"
            )
            fig_corr.update_layout(
                height=400,
                margin=dict(l=10, r=10, t=40, b=20)
            )
            st.plotly_chart(fig_corr, use_container_width=True)

            # Highlight strong correlations in plain English
            st.markdown("**Notable correlations (|r| > 0.5):**")
            found_any = False
            for i in range(len(corr.columns)):
                for j in range(i+1, len(corr.columns)):
                    r = corr.iloc[i, j]
                    if abs(r) > 0.5:
                        found_any = True
                        direction = "positive" if r > 0 else "negative"
                        strength  = "strong" if abs(r) > 0.75 else "moderate"
                        st.markdown(
                            f"- **{corr.columns[i]}** and **{corr.columns[j]}** "
                            f"have a {strength} {direction} correlation "
                            f"(**r = {r:.2f}**)"
                        )
            if not found_any:
                st.info("No strong correlations found between numeric variables (|r| > 0.5).")

    else:
        st.info("No numeric columns available for visual insights.")

    # ── Distribution: Categorical Fields ──────────────────────────────────────
    section_header("Distribution: Categorical Fields")

    if not char_df.empty:
        for col in char_cols:
            st.markdown(f"#### `{col}`")

            counts = df[col].value_counts().reset_index()
            counts.columns = [col, "Count"]
            counts["Percentage %"] = (
                counts["Count"] / counts["Count"].sum() * 100
            ).round(2)

            left, right = st.columns([1, 2])

            with left:
                st.dataframe(
                    counts.head(20),
                    use_container_width=True,
                    hide_index=True
                )

            with right:
                plot_data = counts.head(20)
                fig = px.bar(
                    plot_data,
                    x="Count",
                    y=col,
                    orientation="h",
                    title=f"Top values in '{col}'",
                    color="Count",
                    color_continuous_scale=BLUE_SCALE,
                    text="Percentage %"
                )
                fig.update_traces(
                    texttemplate="%{text}%",
                    textposition="outside"
                )
                fig.update_layout(
                    showlegend=False,
                    coloraxis_showscale=False,
                    yaxis=dict(autorange="reversed"),
                    margin=dict(l=10, r=40, t=40, b=20),
                    height=max(300, len(plot_data) * 28)
                )
                st.plotly_chart(fig, use_container_width=True)

            st.divider()
    else:
        st.info("No categorical columns found.")