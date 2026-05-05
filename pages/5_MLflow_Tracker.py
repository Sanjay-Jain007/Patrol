import streamlit as st
import pandas as pd
import plotly.express as px
import mlflow
from mlflow.tracking import MlflowClient
import os

st.set_page_config(page_title="MLflow Tracker", page_icon="🧪", layout="wide")
st.title("🧪 MLflow Experiment Tracker")
st.markdown("---")

MLFLOW_URI = os.path.abspath("mlruns")
mlflow.set_tracking_uri(f"file://{MLFLOW_URI}")
client = MlflowClient()

st.markdown("All experiments are logged during `data_pipeline.py` execution.")

try:
    experiments = client.search_experiments()
    exp_names = [e.name for e in experiments if e.name not in ["Default", "Test"]]

    if not exp_names:
        st.info("No experiments found. Run `python data_pipeline.py` first.")
        st.stop()

    selected = st.selectbox("Select Experiment", exp_names)
    exp = client.get_experiment_by_name(selected)
    runs = client.search_runs(experiment_ids=[exp.experiment_id], order_by=["start_time DESC"])

    if not runs:
        st.info("No runs found for this experiment.")
        st.stop()

    st.subheader(f"Runs — {selected} ({len(runs)} total)")

    rows = []
    for r in runs:
        row = {"Run": r.info.run_name or r.info.run_id[:8], "Status": r.info.status}
        row.update({k: v for k, v in r.data.params.items()})
        row.update({k: round(v, 4) for k, v in r.data.metrics.items()})
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    st.dataframe(summary_df, use_container_width=True)

    metric_cols = [c for c in summary_df.columns
                   if c not in ["Run", "Status"] and summary_df[c].dtype != object]

    if metric_cols:
        st.subheader("Metric Comparison")
        selected_metric = st.selectbox("Select metric", metric_cols)
        plot_df = summary_df[["Run", selected_metric]].dropna()
        if not plot_df.empty:
            fig = px.bar(plot_df, x="Run", y=selected_metric, color="Run",
                         title=f"{selected_metric} across runs",
                         color_discrete_sequence=px.colors.qualitative.Set1)
            fig.update_layout(xaxis_tickangle=-20, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.caption(f"MLflow tracking dir: `{MLFLOW_URI}`")
    st.caption("To open MLflow UI: `mlflow ui --backend-store-uri ./mlruns`  →  localhost:5000")

except Exception as e:
    st.error(f"Could not load experiments: {e}")
    st.info("Run `python data_pipeline.py` first.")
