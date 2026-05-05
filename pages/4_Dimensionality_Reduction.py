import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(page_title="Dimensionality Reduction", page_icon="📉", layout="wide")
st.title("📉 Dimensionality Reduction")
st.markdown("---")

DATA_PATH = os.path.join("data", "processed.parquet")
TSNE_PATH = os.path.join("data", "tsne_result.parquet")
PCA_LOADINGS_PATH = os.path.join("data", "pca_loadings.csv")

@st.cache_data
def load_data():
    return pd.read_parquet(DATA_PATH)

@st.cache_data
def load_tsne():
    return pd.read_parquet(TSNE_PATH)

if not os.path.exists(DATA_PATH):
    st.warning("Run `python data_pipeline.py` first.")
    st.stop()

df = load_data()

tab1, tab2 = st.tabs(["PCA", "t-SNE"])

# ─── PCA ───
with tab1:
    st.subheader("Principal Component Analysis")
    st.markdown("Reduced 12 engineered features to 3 principal components.")

    col1, col2, col3 = st.columns(3)
    # Compute explained variance from MLflow or directly from data
    if os.path.exists(PCA_LOADINGS_PATH):
        loadings = pd.read_csv(PCA_LOADINGS_PATH, index_col=0)
        st.success("PCA loadings loaded from pipeline output.")
    else:
        st.info("PCA loadings file not found. Re-run pipeline.")
        loadings = None

    # Scree plot using PCA columns in df
    pca_cols = [c for c in ["PCA_1", "PCA_2", "PCA_3"] if c in df.columns]
    if pca_cols:
        variances = [df[c].var() for c in pca_cols]
        total_var = sum(variances)
        explained = [v / total_var for v in variances]
        cumulative = np.cumsum(explained)

        col1.metric("PC1 Variance", f"{explained[0]*100:.1f}%")
        col2.metric("PC2 Variance", f"{explained[1]*100:.1f}%")
        col3.metric("Cumulative (PC1+PC2+PC3)", f"{cumulative[-1]*100:.1f}%")

        fig_scree = go.Figure()
        fig_scree.add_trace(go.Bar(x=[f"PC{i+1}" for i in range(len(explained))],
                                   y=[e*100 for e in explained],
                                   name="Individual", marker_color="#3498db"))
        fig_scree.add_trace(go.Scatter(x=[f"PC{i+1}" for i in range(len(cumulative))],
                                       y=[c*100 for c in cumulative],
                                       mode="lines+markers", name="Cumulative",
                                       marker_color="#e74c3c"))
        fig_scree.update_layout(title="Scree Plot - Explained Variance",
                                yaxis_title="Variance (%)", xaxis_title="Component")
        st.plotly_chart(fig_scree, use_container_width=True)

        # 2D scatter PC1 vs PC2
        sample_df = df.sample(min(10000, len(df)), random_state=42)
        top6 = df["Primary Type"].value_counts().head(6).index
        sample_df["Type_Label"] = sample_df["Primary Type"].apply(
            lambda x: x if x in top6 else "OTHER")

        fig_2d = px.scatter(sample_df, x="PCA_1", y="PCA_2", color="Type_Label",
                            opacity=0.4, title="PCA 2D Projection - Colored by Crime Type",
                            height=500)
        st.plotly_chart(fig_2d, use_container_width=True)

        # 3D scatter
        fig_3d = px.scatter_3d(sample_df, x="PCA_1", y="PCA_2", z="PCA_3",
                                color="Type_Label", opacity=0.4,
                                title="PCA 3D Projection", height=600)
        st.plotly_chart(fig_3d, use_container_width=True)

    # Feature loadings table
    if loadings is not None:
        st.subheader("Feature Loadings (Absolute Values)")
        st.dataframe(loadings.round(4), use_container_width=True)

        st.subheader("Top Feature Contributions to PC1")
        pc1 = loadings.loc["PC1"].sort_values(ascending=False)
        fig_feat = px.bar(x=pc1.values[:8], y=pc1.index[:8],
                          orientation="h", color=pc1.values[:8],
                          color_continuous_scale="Blues",
                          title="Most Important Features (PC1)")
        fig_feat.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_feat, use_container_width=True)


# ─── t-SNE ───
with tab2:
    st.subheader("t-SNE 2D Visualization")
    st.markdown("5,000 record sample. Similar crimes group together in 2D space.")

    if not os.path.exists(TSNE_PATH):
        st.warning("t-SNE result not found. Re-run `python data_pipeline.py`.")
        st.stop()

    tsne_df = load_tsne()

    color_by = st.selectbox("Color by", ["Primary_Type", "Time_of_Day", "Season"])

    if color_by == "Primary_Type":
        top8 = tsne_df["Primary_Type"].value_counts().head(8).index
        tsne_df["Label"] = tsne_df["Primary_Type"].apply(lambda x: x if x in top8 else "OTHER")
    else:
        tsne_df["Label"] = tsne_df[color_by]

    fig_tsne = px.scatter(tsne_df, x="tSNE_1", y="tSNE_2", color="Label",
                          opacity=0.5, title=f"t-SNE Projection - Colored by {color_by}",
                          height=600)
    st.plotly_chart(fig_tsne, use_container_width=True)

    col1, col2 = st.columns(2)
    col1.metric("Sample Size", f"{len(tsne_df):,}")
    col2.metric("Unique Labels", tsne_df["Label"].nunique())
