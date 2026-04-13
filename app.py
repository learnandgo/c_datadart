import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import anthropic
import json

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
.dataframe-container tr:nth-child(even) { background-color: #f5f9ff; }
.dataframe-container tr:hover { background-color: #e0eeff; }

.upload-prompt {
    border: 2px dashed #4a90d9;
    border-radius: 12px;
    padding: 36px;
    text-align: center;
    color: #4a6a8a;
    background-color: #f8fbff;
    margin-top: 10px;
}
.upload-prompt h3 { color: #1a4a7a; font-size: 20px; margin-bottom: 8px; }

.filter-title {
    font-weight: 600;
    font-size: 13px;
    color: #1a4a7a;
    margin-bottom: 6px;
}

.text-summary-box {
    background-color: #f0fff4;
    border-left: 5px solid #2ecc71;
    padding: 14px 18px;
    border-radius: 6px;
    font-size: 14px;
    line-height: 1.7;
    margin-top: 8px;
    white-space: pre-wrap;
}

.sentiment-positive {
    background-color: #f0fff4;
    border-left: 6px solid #27ae60;
    padding: 14px 18px;
    border-radius: 6px;
    font-size: 14px;
    line-height: 1.8;
    margin: 8px 0;
}
.sentiment-negative {
    background-color: #fff5f5;
    border-left: 6px solid #e74c3c;
    padding: 14px 18px;
    border-radius: 6px;
    font-size: 14px;
    line-height: 1.8;
    margin: 8px 0;
}
.sentiment-neutral {
    background-color: #f8f9fa;
    border-left: 6px solid #95a5a6;
    padding: 14px 18px;
    border-radius: 6px;
    font-size: 14px;
    line-height: 1.8;
    margin: 8px 0;
}
.sentiment-mixed {
    background-color: #fffbf0;
    border-left: 6px solid #f39c12;
    padding: 14px 18px;
    border-radius: 6px;
    font-size: 14px;
    line-height: 1.8;
    margin: 8px 0;
}
.issues-box {
    background-color: #fff5f5;
    border: 1px solid #fadbd8;
    border-left: 6px solid #c0392b;
    padding: 14px 18px;
    border-radius: 6px;
    font-size: 14px;
    line-height: 1.8;
    margin: 8px 0;
}
.improvements-box {
    background-color: #fef9e7;
    border: 1px solid #fdebd0;
    border-left: 6px solid #e67e22;
    padding: 14px 18px;
    border-radius: 6px;
    font-size: 14px;
    line-height: 1.8;
    margin: 8px 0;
}
.themes-box {
    background-color: #eaf4fb;
    border: 1px solid #d6eaf8;
    border-left: 6px solid #2980b9;
    padding: 14px 18px;
    border-radius: 6px;
    font-size: 14px;
    line-height: 1.8;
    margin: 8px 0;
}
.ai-cta-box {
    background: linear-gradient(135deg, #1a4a7a 0%, #2980b9 100%);
    border-radius: 12px;
    padding: 24px 28px;
    margin: 12px 0 20px 0;
    text-align: center;
}
.ai-cta-box h3 { color: white; font-size: 20px; margin: 0 0 8px 0; }
.ai-cta-box p  { color: rgba(255,255,255,0.85); font-size: 14px; margin: 0 0 16px 0; }

.sentiment-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 600;
    margin-bottom: 10px;
}
.badge-positive { background: #d5f5e3; color: #1e8449; }
.badge-negative { background: #fadbd8; color: #922b21; }
.badge-neutral  { background: #e8e8e8; color: #555; }
.badge-mixed    { background: #fef9e7; color: #b7770d; }

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
    lines.append(f"\nMissing values:\n{df.isnull().sum().to_string()}")
    return "\n".join(lines)

def call_claude(prompt: str) -> str:
    if "ANTHROPIC_API_KEY" not in st.secrets:
        st.error("⚠️ Anthropic API key not found. Add ANTHROPIC_API_KEY to Streamlit secrets.")
        st.stop()
    client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
    msg = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    )
    return msg.content[0].text

def render_ai_summary(df: pd.DataFrame):
    profile = build_profile_text(df)
    with st.spinner("Claude is reading your data..."):
        exec_text = call_claude(f"""
You are a senior data analyst. Analyze this dataset profile and write:
1. A 2-3 sentence executive summary of what this dataset contains
2. 6-8 key bullet point insights with specific numbers from the data
3. 2-3 watch-out flags or data quality observations

Format exactly like this:

EXECUTIVE SUMMARY
[your 2-3 sentence summary]

KEY INSIGHTS
[bullet insights]

WATCH-OUTS
[bullet flags]

Dataset profile:
{profile}
""")
    with st.spinner("Generating analysis tables..."):
        tables_text = call_claude(f"""
You are a senior data analyst. Generate 3 meaningful analysis tables.
For each: pick a relevant categorical grouping, aggregate key numeric columns,
show top 5-6 rows, format as markdown table, add a 1-line insight below.
Use clear bold titles.

Dataset profile:
{profile}
""")
    with st.spinner("Writing visual insights narrative..."):
        visual_text = call_claude(f"""
You are a senior data analyst. Cover these four areas with headers and bullets:
1. KEY DISTRIBUTIONS
2. RELATIONSHIPS AND TRENDS
3. RECOMMENDED VISUALIZATIONS
4. ANOMALIES AND OUTLIERS
Be specific with column names and numbers.

Dataset profile:
{profile}
""")
    st.markdown("#### 📋 Executive Summary")
    st.markdown(f'<div class="exec-box">{exec_text}</div>', unsafe_allow_html=True)
    st.divider()
    st.markdown("#### 📊 Analysis Tables")
    st.markdown(tables_text)
    st.divider()
    st.markdown("#### 👁️ Visual Insights Narrative")
    st.markdown(visual_text)

def render_categorical_filter(df: pd.DataFrame, short_char_cols: list) -> pd.DataFrame:
    filtered_df = df.copy()
    if not short_char_cols:
        st.info("No short categorical columns available for filtering.")
        return filtered_df

    st.markdown("**🔽 Filter data by categorical fields:**")
    cols_to_show = short_char_cols[:4]
    filter_cols  = st.columns(len(cols_to_show))

    for i, cat_col in enumerate(cols_to_show):
        with filter_cols[i]:
            all_options = sorted(df[cat_col].dropna().unique().tolist())
            st.markdown(f'<div class="filter-title">{cat_col}</div>', unsafe_allow_html=True)
            search = st.text_input(
                "Search", key=f"search_{cat_col}",
                placeholder="Type to search...",
                label_visibility="collapsed"
            )
            filtered_options = (
                [o for o in all_options if search.lower() in str(o).lower()]
                if search else all_options
            )
            btn1, btn2 = st.columns(2)
            select_all  = btn1.button("All",  key=f"all_{cat_col}",  use_container_width=True)
            select_none = btn2.button("None", key=f"none_{cat_col}", use_container_width=True)
            state_key = f"checked_{cat_col}"
            if state_key not in st.session_state:
                st.session_state[state_key] = set(all_options)
            if select_all:
                st.session_state[state_key] = set(all_options)
            if select_none:
                st.session_state[state_key] = set()
            checkbox_container = st.container(height=220)
            with checkbox_container:
                for option in filtered_options:
                    checked = option in st.session_state[state_key]
                    new_val = st.checkbox(
                        str(option), value=checked,
                        key=f"chk_{cat_col}_{option}"
                    )
                    if new_val:
                        st.session_state[state_key].add(option)
                    else:
                        st.session_state[state_key].discard(option)
            selected = st.session_state[state_key]
            if selected:
                filtered_df = filtered_df[filtered_df[cat_col].isin(selected)]

    st.caption(f"📊 Showing **{len(filtered_df):,}** of **{len(df):,}** rows after filters")
    return filtered_df

def summarize_categorical_field(col: str, all_values: list, counts_dict: dict) -> str:
    top_values_str = "\n".join(
        [f"  {v}: {c} occurrences" for v, c in list(counts_dict.items())[:15]]
    )
    return call_claude(f"""
You are a data analyst. Summarize this categorical field in 2-4 plain English sentences.
Cover: what the field represents, dominant values, concentration or spread, interesting patterns.

Field name: {col}
Total unique values: {len(all_values)}
Top values by frequency:
{top_values_str}

Write only the summary, no preamble or headers.
""")

def analyze_long_text_sentiment(col: str, sample_values: list) -> dict:
    samples_str = "\n".join([f"  - {str(v)[:300]}" for v in sample_values[:30]])
    raw = call_claude(f"""
You are a data analyst specializing in text analysis and sentiment.
Analyze these sample values and return ONLY a valid JSON object with exactly these keys:

{{
  "sentiment": "positive or negative or neutral or mixed",
  "sentiment_summary": "2-3 sentences describing the overall tone",
  "themes": "2-3 sentences describing key topics and themes",
  "issues": "2-3 sentences describing problems or complaints. Write None identified if none.",
  "improvements": "2-3 sentences describing improvement suggestions. Write None identified if none."
}}

Field name: {col}
Sample values:
{samples_str}

Return ONLY the JSON. No markdown, no explanation.
""")
    try:
        clean = raw.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(clean)
    except Exception:
        return {
            "sentiment": "mixed",
            "sentiment_summary": raw[:300],
            "themes": "Could not parse structured response.",
            "issues": "Could not parse structured response.",
            "improvements": "Could not parse structured response."
        }

def render_sentiment_result(result: dict):
    sentiment = result.get("sentiment", "neutral").lower()
    badge_map = {
        "positive": ("badge-positive", "✅ Positive"),
        "negative": ("badge-negative", "❌ Negative"),
        "neutral":  ("badge-neutral",  "➖ Neutral"),
        "mixed":    ("badge-mixed",    "🔀 Mixed"),
    }
    box_map = {
        "positive": "sentiment-positive",
        "negative": "sentiment-negative",
        "neutral":  "sentiment-neutral",
        "mixed":    "sentiment-mixed",
    }
    badge_class, badge_label = badge_map.get(sentiment, ("badge-neutral", "➖ Neutral"))
    box_class = box_map.get(sentiment, "sentiment-neutral")

    st.markdown(
        f'<span class="sentiment-badge {badge_class}">{badge_label} Sentiment</span>',
        unsafe_allow_html=True
    )
    st.markdown(
        f'<div class="{box_class}"><strong>Overall Sentiment</strong><br>'
        f'{result.get("sentiment_summary","")}</div>',
        unsafe_allow_html=True
    )
    if result.get("themes"):
        st.markdown(
            f'<div class="themes-box"><strong>🔵 Key Themes</strong><br>'
            f'{result.get("themes","")}</div>',
            unsafe_allow_html=True
        )
    issues = result.get("issues", "None identified")
    if issues and issues.strip().lower() != "none identified":
        st.markdown(
            f'<div class="issues-box"><strong>🔴 Issues and Complaints</strong><br>'
            f'{issues}</div>',
            unsafe_allow_html=True
        )
    improvements = result.get("improvements", "None identified")
    if improvements and improvements.strip().lower() != "none identified":
        st.markdown(
            f'<div class="improvements-box"><strong>🟠 Improvement Suggestions</strong><br>'
            f'{improvements}</div>',
            unsafe_allow_html=True
        )


# ══════════════════════════════════════════════════════════════════════════════
# HERO BANNER
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
# MAIN CONTENT
# ══════════════════════════════════════════════════════════════════════════════
if not file:
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
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    numeric_df   = df.select_dtypes(include="number")
    char_df      = df.select_dtypes(include="object")
    numeric_cols = numeric_df.columns.tolist()
    char_cols    = char_df.columns.tolist()

    short_char_cols = [
        col for col in char_cols
        if df[col].dropna().astype(str).str.len().mean() <= 100
    ]
    long_char_cols = [
        col for col in char_cols
        if df[col].dropna().astype(str).str.len().mean() > 100
    ]

    # ── Overall Table Summary ──────────────────────────────────────────────────
    section_header("Overall Table Summary")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Rows",             f"{df.shape[0]:,}")
    m2.metric("Columns",          df.shape[1])
    m3.metric("Numeric cols",     len(numeric_cols))
    m4.metric("Categorical cols", len(short_char_cols))
    m5.metric("Free-text cols",   len(long_char_cols))
    m6.metric("Missing values",   f"{df.isnull().sum().sum():,}")
    st.success(f"✅ **{file.name}** loaded successfully!")

    # ── AI Executive Summary — prominent CTA ──────────────────────────────────
    section_header("🤖 AI Executive Summary")
    st.markdown("""
    <div class="ai-cta-box">
        <h3>🤖 Let Claude Analyze Your Data</h3>
        <p>Get an instant executive summary, analysis tables, and visual insights
        narrative — all written by Claude AI based on your specific dataset.</p>
    </div>
    """, unsafe_allow_html=True)

    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
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
        f" — scroll right to see all columns"
    )

    # ── Numeric Column Statistics ──────────────────────────────────────────────
    section_header("Numeric Column Statistics")
    if not numeric_df.empty:
        desc = numeric_df.describe().T
        desc["median"] = numeric_df.median()
        desc["mode"]   = numeric_df.mode().iloc[0]
        desc["skew"]   = numeric_df.skew()
        desc = desc[[
            "count","mean","median","mode",
            "std","min","25%","75%","max","skew"
        ]]
        desc.columns = [
            "Count","Mean","Median","Mode",
            "Std Dev","Min","25%","75%","Max","Skew"
        ]
        st.dataframe(desc.style.format("{:.2f}"), use_container_width=True)
    else:
        st.info("No numeric columns found.")

    # ── Visual Insights Dashboard ──────────────────────────────────────────────
    section_header("Visual Insights Dashboard")

    if not numeric_df.empty:
        filtered_df = render_categorical_filter(df, short_char_cols)
        st.divider()

        # Histograms
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
                        filtered_df, x=col_name, title=col_name,
                        color_discrete_sequence=[BLUE], nbins=30
                    )
                    fig.update_layout(
                        bargap=0.15, height=280,
                        margin=dict(l=10,r=10,t=40,b=20),
                        showlegend=False
                    )
                    fig.add_vline(
                        x=float(filtered_df[col_name].median()),
                        line_dash="dash", line_color="#1a4a7a",
                        annotation_text="Median",
                        annotation_position="top right"
                    )
                    st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # Box plots
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
                        filtered_df, y=col_name, title=col_name,
                        color_discrete_sequence=[BLUE], points="outliers"
                    )
                    fig.update_layout(height=280, margin=dict(l=10,r=10,t=40,b=20))
                    st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # Correlation heatmap
        if len(numeric_cols) >= 2:
            st.markdown("##### 🔗 Correlations Between Numeric Variables")
            corr = filtered_df[numeric_cols].corr().round(2)
            fig_corr = px.imshow(
                corr, text_auto=True,
                color_continuous_scale=["#ffffff","#4a90d9","#1a4a7a"],
                zmin=-1, zmax=1, title="Correlation Matrix"
            )
            fig_corr.update_layout(height=420, margin=dict(l=10,r=10,t=40,b=20))
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
                            f"- **{corr.columns[i]}** and **{corr.columns[j]}** "
                            f"— {strength} {direction} correlation (**r = {r:.2f}**)"
                        )
            if not found_any:
                st.info("No strong correlations found (|r| > 0.5).")
    else:
        st.info("No numeric columns available for Visual Insights.")

    # ── Distribution: Categorical Fields ──────────────────────────────────────
    section_header("Distribution: Categorical Fields")

    if short_char_cols:
        for col in short_char_cols:
            st.markdown(f"#### `{col}`")
            counts = df[col].value_counts().reset_index()
            counts.columns = [col, "Count"]
            counts["Percentage %"] = (
                counts["Count"] / counts["Count"].sum() * 100
            ).round(2)
            all_values  = df[col].dropna().unique().tolist()
            counts_dict = df[col].value_counts().to_dict()

            left, right = st.columns([1, 2])
            with left:
                st.dataframe(counts.head(20), use_container_width=True, hide_index=True)
            with right:
                plot_data = counts.head(20)
                fig = px.bar(
                    plot_data, x="Count", y=col, orientation="h",
                    title=f"Top values in '{col}'",
                    color="Count", color_continuous_scale=BLUE_SCALE,
                    text="Percentage %"
                )
                fig.update_traces(texttemplate="%{text}%", textposition="outside")
                fig.update_layout(
                    bargap=0.2, showlegend=False, coloraxis_showscale=False,
                    yaxis=dict(autorange="reversed"),
                    margin=dict(l=10,r=50,t=40,b=20),
                    height=max(320, len(plot_data) * 32)
                )
                st.plotly_chart(fig, use_container_width=True)

            with st.expander(f"🤖 AI Summary of '{col}' field", expanded=False):
                if st.button(f"Summarize '{col}' field", key=f"summarize_{col}"):
                    with st.spinner(f"Claude is analyzing '{col}'..."):
                        summary = summarize_categorical_field(col, all_values, counts_dict)
                    st.markdown(
                        f'<div class="text-summary-box">{summary}</div>',
                        unsafe_allow_html=True
                    )
            st.divider()
    else:
        st.info("No short categorical columns found.")

    # ── Long Text Fields — Sentiment Analysis ──────────────────────────────────
    if long_char_cols:
        section_header("📝 Long Text Fields — Sentiment & Theme Analysis")
        st.markdown(
            "These fields contain long free-text values (avg > 100 characters). "
            "Claude reads a sample and returns color-coded sentiment, themes, "
            "issues, and improvement suggestions."
        )

        for col in long_char_cols:
            st.markdown(f"#### `{col}`")
            col_data = df[col].dropna()

            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Total values",  f"{len(col_data):,}")
            s2.metric("Unique values", f"{col_data.nunique():,}")
            s3.metric("Avg length",    f"{col_data.astype(str).str.len().mean():.0f} chars")
            s4.metric("Max length",    f"{col_data.astype(str).str.len().max():,} chars")

            with st.expander("👀 Preview sample values", expanded=False):
                sample_preview = col_data.sample(
                    min(5, len(col_data)), random_state=42
                ).tolist()
                for idx_s, val in enumerate(sample_preview, 1):
                    st.markdown(f"**Sample {idx_s}:** {str(val)[:500]}")
                    st.divider()

            st.markdown(f"""
            <div style="background:#fff8f0; border:1px solid #f0c080;
                        border-radius:10px; padding:16px 20px; margin:10px 0;">
                <strong>🤖 Sentiment Analysis available for this field</strong><br>
                <span style="font-size:13px; color:#666;">
                Click below to let Claude analyze tone, themes, issues and
                improvement suggestions from a sample of {min(30, len(col_data))} values.
                </span>
            </div>
            """, unsafe_allow_html=True)

            if st.button(
                f"🔍 Analyze Sentiment in '{col}'",
                key=f"sentiment_{col}",
                type="primary",
                use_container_width=True
            ):
                sample_values = col_data.sample(
                    min(30, len(col_data)), random_state=42
                ).tolist()
                with st.spinner(f"Claude is reading and analyzing '{col}'..."):
                    result = analyze_long_text_sentiment(col, sample_values)
                render_sentiment_result(result)

            st.divider()

# ── Footer — zero indentation, always renders ──────────────────────────────────
st.divider()
st.markdown("""
<div style="text-align: center; padding: 16px 0 8px 0;
            color: #888; font-size: 13px;">
    © 2026 <strong>DataDart</strong> · 🎯 Datadart — All rights reserved. ·
    Powered by <strong>Claude AI</strong> + <strong>Streamlit</strong><br>
    <span style="font-size: 11px; opacity: 0.7;">
    Your data never leaves your session · No data is stored or shared.
    </span>
</div>
""", unsafe_allow_html=True)
