import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Dimensionality Reduction", page_icon="📉", layout="wide")
st.title("📉 Dimensionality Reduction")
st.markdown("---")

# ✅ FIXED: direct loading from root
@st.cache_data
def load_data():
    return pd.read_parquet("processed_small.parquet")

@st.cache_data
def load_tsne():
    return pd.read_parquet("tsne_result.parquet")

try:
    df = load_data()
    tsne_df = load_tsne()
except:
    st.error("Required data files not found.")
    st.stop()

tab1, tab2 = st.tabs(["PCA", "t-SNE"])

# ─── PCA ───
with tab1:
    st.subheader("Principal Component Analysis")

    col1, col2, col3 = st.columns(3)

    pca_cols = [c for c in ["PCA_1", "PCA_2", "PCA_3"] if c in df.columns]

    if pca_cols:
        variances = [df[c].var() for c in pca_cols]
        total_var = sum(variances)
        explained = [v / total_var for v in variances]
        cumulative = np.cumsum(explained)

        col1.metric("PC1 Variance", f"{explained[0]*100:.1f}%")
        col2.metric("PC2 Variance", f"{explained[1]*100:.1f}%")
        col3.metric("Cumulative", f"{cumulative[-1]*100:.1f}%")

        fig_scree = go.Figure()
        fig_scree.add_trace(go.Bar(
            x=[f"PC{i+1}" for i in range(len(explained))],
            y=[e*100 for e in explained],
            name="Individual"
        ))
        fig_scree.add_trace(go.Scatter(
            x=[f"PC{i+1}" for i in range(len(cumulative))],
            y=[c*100 for c in cumulative],
            mode="lines+markers",
            name="Cumulative"
        ))
        st.plotly_chart(fig_scree, use_container_width=True)

        sample_df = df.sample(min(10000, len(df)), random_state=42)

        fig_2d = px.scatter(sample_df, x="PCA_1", y="PCA_2", opacity=0.4)
        st.plotly_chart(fig_2d, use_container_width=True)

        fig_3d = px.scatter_3d(sample_df, x="PCA_1", y="PCA_2", z="PCA_3", opacity=0.4)
        st.plotly_chart(fig_3d, use_container_width=True)


# ─── t-SNE ───
with tab2:
    st.subheader("t-SNE Visualization")

    color_by = st.selectbox("Color by", ["Primary_Type", "Time_of_Day", "Season"])

    tsne_df["Label"] = tsne_df[color_by]

    fig_tsne = px.scatter(
        tsne_df,
        x="tSNE_1",
        y="tSNE_2",
        color="Label",
        opacity=0.5
    )
    st.plotly_chart(fig_tsne, use_container_width=True)

    col1, col2 = st.columns(2)
    col1.metric("Sample Size", f"{len(tsne_df):,}")
    col2.metric("Unique Labels", tsne_df["Label"].nunique())
