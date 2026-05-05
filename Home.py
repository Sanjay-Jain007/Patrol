import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="PatrolIQ - Smart Safety Analytics",
    page_icon="🚔",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🚔 PatrolIQ - Smart Safety Analytics Platform")
st.markdown("---")

st.markdown("""
### Urban Crime Intelligence for Chicago PD

This platform analyzes crime records from Chicago to help law enforcement
make data-driven decisions on patrol deployment and resource allocation.

Navigate using the sidebar:

- 📊 **EDA** — Crime distribution, temporal and geographic trends
- 🗺️ **Geographic Clustering** — KMeans, DBSCAN, Hierarchical hotspot maps
- 🕐 **Temporal Clustering** — Time-based crime pattern analysis
- 📉 **Dimensionality Reduction** — PCA and t-SNE visualizations
- 🧪 **MLflow Tracker** — Experiment results and model comparison
""")

st.markdown("---")

# ✅ NEW: Directly load preprocessed data
@st.cache_data
def load_data():
    return pd.read_parquet("processed_small.parquet")

try:
    df = load_data()
    st.success("✅ Data loaded successfully. All pages are ready to use.")
except Exception as e:
    st.error("❌ Data file missing or failed to load.")
    st.stop()

col1, col2, col3 = st.columns(3)
col1.info("**Dataset**\n\nChicago Crime Records\nSampled dataset\nOptimized for fast loading")
col2.info("**ML Techniques**\n\nKMeans · DBSCAN · Hierarchical\nPCA · t-SNE\nMLflow Tracking")
col3.info("**Goal**\n\nIdentify crime hotspots\nTemporal patterns\nOptimize patrol routes")

st.markdown("---")
st.markdown("**Data Source:** [Chicago Data Portal](https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2)")
st.caption("PatrolIQ | GUVI Capstone Project")
