import streamlit as st
import pandas as pd
import plotly.express as px

# Page config — must be first Streamlit command
st.set_page_config(
    page_title="DataDart",
    page_icon="🎯",
    layout="wide"
)

# Title
st.title("🎯 DataDart")
st.caption("Upload your data. Get instant insights.")
st.divider()

# File uploader
file = st.file_uploader(
    "Upload your data file",
    type=["csv", "xlsx", "xls"],
    help="Supported formats: CSV, Excel"
)

# Only run if file is uploaded
if file:

    # Load the file
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    # Show basic info
    st.success(f"✅ {file.name} loaded successfully!")

    col1, col2 = st.columns(2)
    col1.metric("Rows", f"{df.shape[0]:,}")
    col2.metric("Columns", df.shape[1])

    st.divider()

    # Data preview
    st.subheader("🔍 Data Preview")
    st.dataframe(df.head(50), use_container_width=True)

    st.divider()

    # Summary stats — numeric
    st.subheader("📊 Numeric Column Statistics")
    numeric_df = df.select_dtypes(include="number")
    if not numeric_df.empty:
        st.dataframe(numeric_df.describe().T, use_container_width=True)
    else:
        st.info("No numeric columns found.")

    st.divider()

    # Numeric chart
    st.subheader("📈 Numeric Distribution")
    if not numeric_df.empty:
        num_col = st.selectbox(
            "Pick a numeric column to visualize",
            numeric_df.columns.tolist(),
            key="num_col"
        )
        fig = px.histogram(
            df, x=num_col,
            title=f"Distribution of {num_col}",
            color_discrete_sequence=["#2ecc71"]
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Character/categorical columns
    st.subheader("🔤 Character Field Distributions")
    char_df = df.select_dtypes(include="object")

    if not char_df.empty:

        for col in char_df.columns:

            st.markdown(f"#### `{col}`")

            # Value counts table
            counts = (
                df[col]
                .value_counts()
                .reset_index()
            )
            counts.columns = [col, "Count"]
            counts["Percentage %"] = (
                counts["Count"] / counts["Count"].sum() * 100
            ).round(2)

            # Side by side — table and chart
            left, right = st.columns([1, 2])

            with left:
                st.dataframe(
                    counts.head(20),
                    use_container_width=True,
                    hide_index=True
                )

            with right:
                # Only show top 20 for readability
                plot_data = counts.head(20)
                fig = px.bar(
                    plot_data,
                    x="Count",
                    y=col,
                    orientation="h",
                    title=f"Top values in '{col}'",
                    color="Count",
                    color_continuous_scale=[
                        "#1a7a3c", "#2ecc71"
                    ],
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
                    margin=dict(l=10, r=40, t=40, b=20)
                )
                st.plotly_chart(fig, use_container_width=True)

            st.divider()

    else:
        st.info("No character/categorical columns found.")