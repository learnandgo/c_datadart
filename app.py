import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import anthropic

st.set_page_config(
    page_title="DataDart",
    page_icon="🎯",
    layout="wide"
)

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
}
.exec-box {
    background-color: #f0f7ff;
    border-left: 5px solid #1a6aaa;
    padding: 16px 20px;
    border-radius: 6px;
    margin-bottom: 12px;
    font-size: 15px;
    line-height: 1.7;
}
.kpi-box {
    background-color: #f0f7ff;
    border: 1px solid #c0d8f0;
    border-radius: 8px;
    padding: 14px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

BLUE = "#4a90d9"
BLUE_SCALE = ["#d0e8ff", "#4a90d9", "#1a4a7a"]

def section_header(title: str):
    st.markdown(f'<div class="section-header">📌 {title}</div>',
                unsafe_allow_html=True)

def build_profile_text(df):
    """Build compact data profile string to send to Claude."""
    numeric_df = df.select_dtypes(include="number")
    char_df    = df.select_dtypes(include="object")
    lines = []
    lines.append(f"Rows: {df.shape[0]:,}, Columns: {df.shape[1]}")
    lines.append(f"Numeric columns: {numeric_df.columns.tolist()}")
    lines.append(f"Categorical columns: {char_df.columns.tolist()}")
    lines.append("\nNumeric summary:")
    lines.append(numeric_df.describe().round(2).to_string())
    lines.append("\nCategorical top values:")
    for col in char_df.columns[:6]:
        top = df[col].value_counts().head(5).to_dict()
        lines.append(f"  {col}: {top}")
    lines.append(f"\nMissing values per column:\n{df.isnull().sum().to_string()}")
    return "\n".join(lines)

def call_claude(prompt: str) -> str:
    """Call Claude API and return text response."""
    client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
    msg = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}]
    )
    return msg.content[0].text

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
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    numeric_df   = df.select_dtypes(include="number")
    char_df      = df.select_dtypes(include="object")
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

    # FIX 1: horizontal scroll via container + full column display
    st.dataframe(
        df.head(n_rows),
        use_container_width=False,   # let it expand horizontally
        width=None,                  # no fixed width cap
        height=400                   # fixed height with vertical scroll
    )
    st.caption(f"Showing {n_rows} rows × {df.shape[1]} columns — scroll right to see all columns →")

    # ── Numeric Column Statistics ──────────────────────────────────────────────
    section_header("Numeric Column Statistics")
    if not numeric_df.empty:
        desc = numeric_df.describe().T
        desc["median"] = numeric_df.median()
        desc["mode"]   = numeric_df.mode().iloc[0]
        desc["skew"]   = numeric_df.skew()
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
        filtered_df = df.copy()

        # FIX 3: single dropdown per categorical field (selectbox not multiselect)
        if char_cols:
            st.markdown("**Filter data by categorical field:**")
            filter_cols = st.columns(min(len(char_cols), 4))
            for i, cat_col in enumerate(char_cols[:4]):
                with filter_cols[i]:
                    options = ["All"] + sorted(
                        df[cat_col].dropna().unique().tolist()
                    )
                    selected = st.selectbox(
                        cat_col,
                        options,
                        key=f"filter_{cat_col}"
                    )
                    if selected != "All":
                        filtered_df = filtered_df[
                            filtered_df[cat_col] == selected
                        ]

            st.caption(
                f"Showing {len(filtered_df):,} of {len(df):,} rows"
            )

        st.divider()

        # FIX 2: Histograms with clear bar separation (bargap)
        st.markdown("##### Distributions")
        cols_per_row = 2
        rows_needed  = -(-len(numeric_cols) // cols_per_row)

        for row in range(rows_needed):
            grid_cols = st.columns(cols_per_row)
            for col_idx in range(cols_per_row):
                chart_idx = row * cols_per_row + col_idx
                if chart_idx >= len(numeric_cols):
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
                    # FIX 2: bargap adds clear separation between bars
                    fig.update_layout(
                        bargap=0.15,
                        margin=dict(l=10, r=10, t=40, b=20),
                        height=280,
                        showlegend=False
                    )
                    fig.add_vline(
                        x=float(filtered_df[col_name].median()),
                        line_dash="dash",
                        line_color="#1a4a7a",
                        annotation_text="Median",
                        annotation_position="top right"
                    )
                    st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # Box plots
        st.markdown("##### Box Plots")
        for row in range(rows_needed):
            grid_cols = st.columns(cols_per_row)
            for col_idx in range(cols_per_row):
                chart_idx = row * cols_per_row + col_idx
                if chart_idx >= len(numeric_cols):
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

        # Correlation heatmap
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
                            f"- **{corr.columns[i]}** and "
                            f"**{corr.columns[j]}** — "
                            f"{strength} {direction} correlation "
                            f"(**r = {r:.2f}**)"
                        )
            if not found_any:
                st.info("No strong correlations found (|r| > 0.5).")

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
                    bargap=0.2,
                    showlegend=False,
                    coloraxis_showscale=False,
                    yaxis=dict(autorange="reversed"),
                    margin=dict(l=10, r=40, t=40, b=20),
                    height=max(300, len(plot_data) * 32)
                )
                st.plotly_chart(fig, use_container_width=True)
            st.divider()
    else:
        st.info("No categorical columns found.")

    # ── AI Executive Summary ───────────────────────────────────────────────────
    section_header("AI Executive Summary")
    st.markdown(
        "Claude analyzes your data and generates an intelligent summary, "
        "key findings, and analysis tables — just like a data analyst would."
    )

    if st.button("🤖 Generate AI Summary", type="primary"):

        profile = build_profile_text(df)

        with st.spinner("Claude is analyzing your data..."):

            # ── Executive Summary ──────────────────────────────────────────
            exec_prompt = f"""
You are a senior data analyst. Analyze this dataset profile and write:
1. A 2-3 sentence executive summary of what this dataset contains
2. 6-8 key bullet point insights — be specific with numbers from the data
3. 2-3 watch-out flags or data quality observations

Format exactly like this:
EXECUTIVE SUMMARY
[your summary here]

KEY INSIGHTS
- [insight with specific numbers]
- [insight with specific numbers]
(continue for all insights)

WATCH-OUTS
- [flag]
- [flag]

Dataset profile:
{profile}
"""
            exec_text = call_claude(exec_prompt)

            # ── Analysis Tables ────────────────────────────────────────────
            tables_prompt = f"""
You are a senior data analyst. Based on this dataset profile, generate 3 analysis tables.
For each table:
- Pick the most meaningful grouping from the categorical columns
- Aggregate the most relevant numeric columns
- Show top 5-6 rows only
- Format as a proper markdown table

Generate exactly 3 tables with clear titles.

Dataset profile:
{profile}
"""
            tables_text = call_claude(tables_prompt)

            # ── Visual Insights Text ───────────────────────────────────────
            visual_prompt = f"""
You are a senior data analyst. Based on this dataset, write a visual insights narrative:
1. What are the most interesting distributions in this data?
2. What relationships or trends stand out?
3. What would you recommend visualizing first and why?
4. Any anomalies or outliers worth investigating?

Be specific with column names and numbers from the profile.
Use clear section headers and bullet points.

Dataset profile:
{profile}
"""
            visual_text = call_claude(visual_prompt)

        # ── Render Executive Summary ───────────────────────────────────────
        st.markdown("#### 📋 Executive Summary")
        st.markdown(
            f'<div class="exec-box">{exec_text}</div>',
            unsafe_allow_html=True
        )

        # ── Render Analysis Tables ─────────────────────────────────────────
        st.markdown("#### 📊 Analysis Tables")
        st.markdown(tables_text)

        # ── Render Visual Insights ─────────────────────────────────────────
        st.markdown("#### 👁️ Visual Insights Narrative")
        st.markdown(visual_text)