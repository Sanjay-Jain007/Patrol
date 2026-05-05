import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Geographic Clustering", page_icon="🗺️", layout="wide")
st.title("🗺️ Geographic Crime Hotspot Clustering")
st.markdown("---")

# ✅ FIXED: load from processed_small.parquet
@st.cache_data
def load_data():
    return pd.read_parquet("processed_small.parquet")

try:
    df = load_data()
except:
    st.error("Data file not found.")
    st.stop()

# Sample for display speed
plot_df = df.sample(min(50000, len(df)), random_state=42).copy()

tab1, tab2, tab3, tab4 = st.tabs(["KMeans", "DBSCAN", "Hierarchical", "Comparison"])

# ─── KMeans ───
with tab1:
    st.subheader("KMeans Geographic Clustering (k=7)")
    st.markdown("Divides Chicago into 7 distinct crime concentration zones.")

    fig = px.scatter_mapbox(plot_df, lat="Latitude", lon="Longitude",
                            color="KMeans_Geo_Cluster",
                            zoom=10, height=600,
                            mapbox_style="carto-positron",
                            title="KMeans Crime Clusters",
                            opacity=0.5,
                            color_continuous_scale="Rainbow")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Cluster Summary")
    summary = df.groupby("KMeans_Geo_Cluster").agg(
        Total_Crimes=("Primary Type", "count"),
        Top_Crime=("Primary Type", lambda x: x.value_counts().index[0]),
        Arrest_Rate=("Arrest", "mean"),
        Avg_Severity=("Crime_Severity_Score", "mean"),
        Avg_Lat=("Latitude", "mean"),
        Avg_Lon=("Longitude", "mean")
    ).reset_index()
    summary["Arrest_Rate"] = (summary["Arrest_Rate"] * 100).round(1)
    summary["Avg_Severity"] = summary["Avg_Severity"].round(2)
    summary["Avg_Lat"] = summary["Avg_Lat"].round(4)
    summary["Avg_Lon"] = summary["Avg_Lon"].round(4)
    st.dataframe(summary, use_container_width=True)

    st.subheader("Crime Type Distribution per Cluster")
    top5 = df["Primary Type"].value_counts().head(5).index
    heat = df[df["Primary Type"].isin(top5)].groupby(
        ["KMeans_Geo_Cluster", "Primary Type"]).size().reset_index(name="Count")
    fig2 = px.bar(heat, x="KMeans_Geo_Cluster", y="Count", color="Primary Type",
                  barmode="stack", title="Top 5 Crime Types per Cluster")
    st.plotly_chart(fig2, use_container_width=True)


# ─── DBSCAN ───
with tab2:
    st.subheader("DBSCAN Density-Based Clustering")
    st.markdown("Points labeled **-1** are noise/outliers.")

    df_db = df[df["DBSCAN_Geo_Cluster"] != -1].copy()
    df_noise = df[df["DBSCAN_Geo_Cluster"] == -1].copy()

    col1, col2, col3 = st.columns(3)
    col1.metric("Dense Clusters Found", df_db["DBSCAN_Geo_Cluster"].nunique())
    col2.metric("Points in Clusters", f"{len(df_db):,}")
    col3.metric("Noise Points", f"{len(df_noise):,}")

    plot_db = df_db.sample(min(30000, len(df_db)), random_state=42)
    fig3 = px.scatter_mapbox(plot_db, lat="Latitude", lon="Longitude",
                              color="DBSCAN_Geo_Cluster",
                              zoom=10, height=600,
                              mapbox_style="carto-positron",
                              title="DBSCAN Dense Crime Clusters",
                              opacity=0.5,
                              color_continuous_scale="Viridis")
    st.plotly_chart(fig3, use_container_width=True)


# ─── Hierarchical ───
with tab3:
    st.subheader("Hierarchical Clustering (k=7)")

    df_hc = df[df["HC_Geo_Cluster"] != -1].copy()

    fig4 = px.scatter_mapbox(df_hc, lat="Latitude", lon="Longitude",
                              color="HC_Geo_Cluster",
                              zoom=10, height=600,
                              mapbox_style="carto-positron",
                              title="Hierarchical Clusters",
                              opacity=0.6,
                              color_continuous_scale="Plasma")
    st.plotly_chart(fig4, use_container_width=True)

    summary_hc = df_hc.groupby("HC_Geo_Cluster").agg(
        Count=("Primary Type", "count"),
        Top_Crime=("Primary Type", lambda x: x.value_counts().index[0]),
        Arrest_Rate=("Arrest", "mean")
    ).reset_index()
    summary_hc["Arrest_Rate"] = (summary_hc["Arrest_Rate"] * 100).round(1)
    st.dataframe(summary_hc, use_container_width=True)


# ─── Comparison ───
with tab4:
    st.subheader("Algorithm Comparison")

    try:
        import mlflow
        from mlflow.tracking import MlflowClient
        import os

        MLFLOW_URI = os.path.abspath("mlruns")
        mlflow.set_tracking_uri(f"file://{MLFLOW_URI}")
        client = MlflowClient()

        exp = client.get_experiment_by_name("Geographic_Clustering")
        if exp:
            runs = client.search_runs(experiment_ids=[exp.experiment_id])
            rows = []
            for r in runs:
                rows.append({
                    "Algorithm": r.data.params.get("algorithm", r.info.run_name),
                    "Silhouette": round(r.data.metrics.get("silhouette_score", 0), 4),
                    "Davies-Bouldin": round(r.data.metrics.get("davies_bouldin_score", 0), 4)
                        if "davies_bouldin_score" in r.data.metrics else "N/A"
                })
            cmp_df = pd.DataFrame(rows)
            st.dataframe(cmp_df, use_container_width=True)

    except:
        st.info("MLflow data not available in deployment.")
