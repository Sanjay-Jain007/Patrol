import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="MLflow Tracker", page_icon="🧪", layout="wide")
st.title("🧪 MLflow Experiment Tracker")
st.markdown("---")

st.info("MLflow tracking is available only in local environment.")

try:
    import mlflow
    from mlflow.tracking import MlflowClient
    import os

    MLFLOW_URI = os.path.abspath("mlruns")
    mlflow.set_tracking_uri(f"file://{MLFLOW_URI}")
    client = MlflowClient()

    experiments = client.search_experiments()
    exp_names = [e.name for e in experiments if e.name not in ["Default", "Test"]]

    if not exp_names:
        st.warning("No MLflow experiments found in deployment.")
        st.stop()

    selected = st.selectbox("Select Experiment", exp_names)
    exp = client.get_experiment_by_name(selected)
    runs = client.search_runs(experiment_ids=[exp.experiment_id])

    rows = []
    for r in runs:
        row = {"Run": r.info.run_name or r.info.run_id[:8]}
        row.update(r.data.metrics)
        rows.append(row)

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)

    if not df.empty:
        metric = st.selectbox("Metric", df.columns[1:])
        fig = px.bar(df, x="Run", y=metric)
        st.plotly_chart(fig, use_container_width=True)

except:
    st.warning("MLflow data not available in deployed app.")
