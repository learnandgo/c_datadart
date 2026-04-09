import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import anthropic

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DataDart",
    page_icon="🎯",
    layout="wide"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
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
    letter-spacing: 0.4px;
}

.exec-box {
    background-color: #f0f7ff;
    border-left: 5px solid #1a6aaa;
    padding: 16px 20px;
    border-radius: 6px;
    margin-bottom: 12px;
    font-size: 15px;
    line-height: 1.8;
    white-space: pre-wrap;
}

.hero-banner {
    background: linear-gradient(135deg, #0d2b4e 0%, #1a5a9a 100%);
    color: white;
    padding: 28px 36px;
    border-radius: 12px;
    margin-bottom: 24px;
}
.hero-banner h1 {
    font-size: 32px;
    font-weight: 700;
    margin: 0 0 6px 0;
    color: white;
}
.hero-banner p {
    font-size: 16px;
    margin: 4px 0;
    opacity: 0.88;
    color: white;
}
.hero-tag {
    display: inline-block;
    background: rgba(255,255,255,0.18);
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 13px;
    margin: 6px 4px 0 0;
    color: white;
}

.dataframe-container {
    overflow-x: auto;
    overflow-y: auto;
    max-height: 420px;
    width: 100%;
    display: block;
    border: 1px solid #d0e0f0;
    border-radius: 6px;
    padding: 4px;
}
.dataframe-container table {
    border-collapse: collapse;
    font-size: 13px;
    white-space: nowrap;
    width: max-content;
}
.dataframe-container th {
    background-color: #1a6aaa;
    color: white;
    padding: 7px 14px;
    text-align: left;
    position: sticky;
    top: 0;
    z-index: 1;
}
.dataframe-container td {
    padding: 6px 14px;
    border-bottom: 1px solid #e8f0f8;
}
.dataframe-container tr:nth-child(even) {
    background-color: #f5f9ff;
}
.dataframe-container tr:hover {
    background-color: #e0eeff;
}

.upload-prompt {
    border: 2px dashed #4a90d9;
    border-radius: 12px;
    padding: 36px;
    text-align: center;
    color: #4a6a8a;
    background-color: #f8fbff;
    margin-top: 10px;
}
.upload-prompt h3 {
    color: #1a4a7a;
    font-size: 20px;
    margin-bottom: 8px;
}

</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
BLUE       = "#4a90d9"
BLUE_SCALE = ["#d0e8ff", "#4a90d9", "#1a4a7a"]

# ── Helper functions ───────────────────────────────────────────────────────────
def section_header(title: str):
    st.markdown(
        f'<div class="section-header">📌 {title}</div>',
        unsafe_allow_html=True
    )

def build_profile_text(df: pd.DataFrame) -> str:
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
    for col in char_df.columns[:8]:
        top = df[col].value_counts().head(5).to_dict()
        lines.append(f"  {col}: {top}")
    lines.append(
        f"\nMissing values per column:\n{df.isnull().sum().to_string()}"
    )
    return "\n".join(lines)

def call_claude(prompt: str) -> str:
    """Call Claude API — with friendly error if key is missing."""
    if "ANTHROPIC_API_KEY" not in st.secrets:
        st.error(
            "⚠️ Anthropic API key not found. "
            "Please add ANTHROPIC_API_KEY to your Streamlit secrets."
        )
        st.stop()
    client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
    msg = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}]
    )
    return msg.content[0].text

def render_ai_summary(df: pd.DataFrame):
    """Run all three Claude calls and render results."""
    profile = build_profile_text(df)

    with st.spinner("Claude is reading your data..."):
        exec_text = call_claude(f"""
You are a senior data analyst. Analyze this dataset profile and write:
1. A 2-3 sentence executive summary of what this dataset contains
2. 6-8 key bullet point insights — be specific with real numbers from the data
3. 2-3 watch-out flags or data quality observations

Format exactly like this (keep the labels as headings):

EXECUTIVE SUMMARY
[your 2-3 sentence summary]

KEY INSIGHTS
• [insight with specific numbers]
• [insight with specific numbers]
(continue for all insights)

WATCH-OUTS
• [flag or data quality note]
• [flag or data quality note]

Dataset profile:
{profile}
""")

    with st.spinner("Generating analysis tables..."):
        tables_text = call_claude(f"""
You are a senior data analyst. Based on this dataset profile, generate
3 meaningful analysis tables. For each:
- Pick the most relevant grouping from the categorical columns
- Aggregate the most important numeric columns
- Show top 5-6 rows only
- Format as a proper markdown table with aligned columns
- Add a short 1-line insight below each table

Generate exactly 3 tables with clear bold titles.

Dataset profile:
{profile}
""")

    with st.spinner("Writing visual insights narrative..."):
        visual_text = call_claude(f"""
You are a senior data analyst writing a visual insights report section.
Cover these four areas with clear headers and bullet points:

1. KEY DISTRIBUTIONS — what stands out in the numeric distributions?
2. RELATIONSHIPS & TRENDS — what correlations or patterns are notable?
3. RECOMMENDED VISUALIZATIONS — what charts would reveal the most insight and why?
4. ANOMALIES & OUTLIERS — anything unusual worth investigating?

Be specific — use actual column names and numbers from the profile.

Dataset profile:
{profile}
""")

    # Render results
    st.markdown("#### 📋 Executive Summary")
    st.markdown(
        f'<div class="exec-box">{exec_text}</div>',
        unsafe_allow_html=True
    )
    st.divider()

    st.markdown("#### 📊 Analysis Tables")
    st.markdown(tables_text)
    st.divider()

    st.markdown("#### 👁️ Visual Insights Narrative")
    st.markdown(visual_text)


# ══════════════════════════════════════════════════════════════════════════════
# HERO BANNER — always visible at top
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-banner">
    <h1>🎯 DataDart</h1>
    <p>Upload any dataset and get instant AI-powered analysis — stats, charts,
       distributions, correlations, and an executive summary written by Claude.</p>
    <span class="hero-tag">📁 CSV</span>
    <span class="hero-tag">📊 Excel</span>
    <span class="hero-tag">📈 Auto Charts</span>
    <span class="hero-tag">🤖 AI Insights</span>
    <span class="hero-tag">🔗 Correlations</span>
    <span class="hero-tag">📤 Export coming soon</span>
</div>
""", unsafe_allow_html=True)

# ── How it works strip ─────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.info("**Step 1** 📁\nUpload CSV or Excel file")
c2.info("**Step 2** 📊\nAuto stats + charts generated")
c3.info("**Step 3** 🔍\nExplore distributions & correlations")
c4.info("**Step 4** 🤖\nGenerate AI executive summary")

st.divider()

# ── File uploader ──────────────────────────────────────────────────────────────
file = st.file_uploader(
    "📂 Drop your data file here or click to browse",
    type=["csv", "xlsx", "xls"],
    help="Supported: CSV, Excel (.xlsx, .xls)"
)

# ══════════════════════════════════════════════════════════════════════════════
# if/else — NO st.stop() so footer always renders
# ══════════════════════════════════════════════════════════════════════════════
if not file:
    # ── Empty state ────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="upload-prompt">
        <h3>👆 Upload a file to get started</h3>
        <p>DataDart will instantly profile your data, generate charts,
        detect correlations, and let Claude write an executive summary.</p>
        <p style="font-size:13px; margin-top:12px; opacity:0.7;">
        Supported formats: CSV · Excel (.xlsx / .xls)
        &nbsp;|&nbsp; Your data never leaves your session.
        </p>
    </div>
    """, unsafe_allow_html=True)

else:
    # ══════════════════════════════════════════════════════════════════════════
    # DATA LOADED — runs only after file is uploaded
    # ══════════════════════════════════════════════════════════════════════════
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
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Rows",             f"{df.shape[0]:,}")
    m2.metric("Columns",          df.shape[1])
    m3.metric("Numeric cols",     len(numeric_cols))
    m4.metric("Categorical cols", len(char_cols))
    m5.metric("Missing values",   f"{df.isnull().sum().sum():,}")
    st.success(f"✅ **{file.name}** loaded successfully!")

    # ── AI Executive Summary ───────────────────────────────────────────────────
    section_header("🤖 AI Executive Summary")
    st.markdown(
        "Claude reads your dataset profile and writes an executive summary, "
        "analysis tables, and visual insights — tailored to your specific data."
    )
    if st.button("✨ Generate AI Summary", type="primary", use_container_width=True):
        render_ai_summary(df)
    st.divider()

    # ── Data Preview ───────────────────────────────────────────────────────────
    section_header("Data Preview")
    n_rows = st.slider("Rows to preview", 5, min(500, df.shape[0]), 50, step=5)
    preview_html = df.head(n_rows).to_html(index=False)
    st.markdown(
        f'<div class="dataframe-container">{preview_html}</div>',
        unsafe_allow_html=True
    )
    st.caption(
        f"Showing {n_rows} of {df.shape[0]:,} rows × {df.shape[1]} columns"
        f" — scroll right → to see all columns"
    )

    # ── Numeric Column Statistics ──────────────────────────────────────────────
    section_header("Numeric Column Statistics")
    if not numeric_df.empty:
        desc = numeric_df.describe().T
        desc["median"] = numeric_df.median()
        desc["mode"]   = numeric_df.mode().iloc[0]
        desc["skew"]   = numeric_df.skew()
        desc = desc[[
            "count", "mean", "median", "mode",
            "std", "min", "25%", "75%", "max", "skew"
        ]]
        desc.columns = [
            "Count", "Mean", "Median", "Mode",
            "Std Dev", "Min", "25%", "75%", "Max", "Skew"
        ]
        st.dataframe(desc.style.format("{:.2f}"), use_container_width=True)
    else:
        st.info("No numeric columns found.")

    # ── Visual Insights Dashboard ──────────────────────────────────────────────
    section_header("Visual Insights Dashboard")

    if not numeric_df.empty:
        filtered_df = df.copy()

        # Multiselect filters — one per categorical column up to 4
        if char_cols:
            st.markdown("**🔽 Filter data by categorical fields:**")
            filter_cols_ui = st.columns(min(len(char_cols), 4))
            for i, cat_col in enumerate(char_cols[:4]):
                with filter_cols_ui[i]:
                    options = sorted(df[cat_col].dropna().unique().tolist())
                    selected = st.multiselect(
                        label=cat_col,
                        options=options,
                        default=options,
                        key=f"filter_{cat_col}"
                    )
                    if selected:
                        filtered_df = filtered_df[
                            filtered_df[cat_col].isin(selected)
                        ]
            st.caption(
                f"📊 Showing **{len(filtered_df):,}** of **{len(df):,}** "
                f"rows after filters"
            )

        st.divider()

        # Histograms grid — 2 per row
        st.markdown("##### 📊 Distributions")
        cols_per_row = 2
        rows_needed  = -(-len(numeric_cols) // cols_per_row)

        for row in range(rows_needed):
            grid = st.columns(cols_per_row)
            for col_idx in range(cols_per_row):
                idx = row * cols_per_row + col_idx
                if idx >= len(numeric_cols):
                    break
                col_name = numeric_cols[idx]
                with grid[col_idx]:
                    fig = px.histogram(
                        filtered_df, x=col_name,
                        title=col_name,
                        color_discrete_sequence=[BLUE],
                        nbins=30
                    )
                    fig.update_layout(
                        bargap=0.15,
                        height=280,
                        margin=dict(l=10, r=10, t=40, b=20),
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

        # Box plots grid — 2 per row
        st.markdown("##### 📦 Box Plots")
        for row in range(rows_needed):
            grid = st.columns(cols_per_row)
            for col_idx in range(cols_per_row):
                idx = row * cols_per_row + col_idx
                if idx >= len(numeric_cols):
                    break
                col_name = numeric_cols[idx]
                with grid[col_idx]:
                    fig = px.box(
                        filtered_df, y=col_name,
                        title=col_name,
                        color_discrete_sequence=[BLUE],
                        points="outliers"
                    )
                    fig.update_layout(
                        height=280,
                        margin=dict(l=10, r=10, t=40, b=20)
                    )
                    st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # Correlation heatmap
        if len(numeric_cols) >= 2:
            st.markdown("##### 🔗 Correlations Between Numeric Variables")
            corr = filtered_df[numeric_cols].corr().round(2)
            fig_corr = px.imshow(
                corr,
                text_auto=True,
                color_continuous_scale=["#ffffff", "#4a90d9", "#1a4a7a"],
                zmin=-1, zmax=1,
                title="Correlation Matrix"
            )
            fig_corr.update_layout(
                height=420,
                margin=dict(l=10, r=10, t=40, b=20)
            )
            st.plotly_chart(fig_corr, use_container_width=True)

            st.markdown("**Notable correlations (|r| > 0.5):**")
            found_any = False
            for i in range(len(corr.columns)):
                for j in range(i + 1, len(corr.columns)):
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

    else:
        st.info("No numeric columns available for Visual Insights.")

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
                    margin=dict(l=10, r=50, t=40, b=20),
                    height=max(320, len(plot_data) * 32)
                )
                st.plotly_chart(fig, use_container_width=True)
            st.divider()
    else:
        st.info("No categorical columns found.")

# ══════════════════════════════════════════════════════════════════════════════
# FOOTER — zero indentation, outside if/else, ALWAYS renders
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.markdown("""
<div style="text-align: center; padding: 16px 0 8px 0;
            color: #888; font-size: 13px;">
    © 2026 <strong>DataDart</strong> · Built with 🎯 by Sreena ·
    Powered by <strong>Claude AI</strong> + <strong>Streamlit</strong><br>
    <span style="font-size: 11px; opacity: 0.7;">
    Your data never leaves your session · No data is stored or shared
    </span>
</div>
""", unsafe_allow_html=True)
