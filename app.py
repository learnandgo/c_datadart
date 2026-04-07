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

    # Summary stats
    st.subheader("📊 Summary Statistics")
    st.dataframe(df.describe(), use_container_width=True)

    st.divider()

    # Auto chart
    st.subheader("📈 Quick Chart")
    numeric_cols = df.select_dtypes("number").columns.tolist()

    if numeric_cols:
        col = st.selectbox("Pick a column to visualize", numeric_cols)
        fig = px.histogram(df, x=col, title=f"Distribution of {col}")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No numeric columns found for charting.")