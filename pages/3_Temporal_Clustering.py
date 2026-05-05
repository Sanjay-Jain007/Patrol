import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Temporal Clustering", page_icon="🕐", layout="wide")
st.title("🕐 Temporal Crime Pattern Clustering")
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

# ─── Peak hour analysis ───
st.subheader("Crime Frequency by Hour")
hourly = df.groupby("Hour").agg(
    Count=("Primary Type", "count"),
    Avg_Severity=("Crime_Severity_Score", "mean")
).reset_index()

fig1 = go.Figure()
fig1.add_trace(go.Bar(x=hourly["Hour"], y=hourly["Count"],
                      name="Crime Count", marker_color="#e74c3c", opacity=0.7))
fig1.add_trace(go.Scatter(x=hourly["Hour"],
                           y=hourly["Avg_Severity"] * hourly["Count"].max() / 10,
                           name="Avg Severity (scaled)", mode="lines+markers",
                           marker_color="#3498db", yaxis="y2"))
fig1.update_layout(
    title="Hourly Crime Count vs Severity",
    xaxis_title="Hour of Day",
    yaxis=dict(title="Crime Count"),
    yaxis2=dict(title="Avg Severity", overlaying="y", side="right")
)
st.plotly_chart(fig1, use_container_width=True)

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Weekday vs Weekend")
    wd = df.groupby(["Hour", "Is_Weekend"]).size().reset_index(name="Count")
    wd["Type"] = wd["Is_Weekend"].map({0: "Weekday", 1: "Weekend"})
    fig2 = px.line(wd, x="Hour", y="Count", color="Type", markers=True,
                   title="Weekday vs Weekend Crime by Hour",
                   color_discrete_map={"Weekday": "#3498db", "Weekend": "#e74c3c"})
    st.plotly_chart(fig2, use_container_width=True)

with col2:
    st.subheader("Seasonal Crime Trend")
    seasonal = df.groupby("Season").size().reset_index(name="Count")
    fig3 = px.bar(seasonal, x="Season", y="Count", color="Season",
                  title="Total Crimes by Season",
                  color_discrete_sequence=px.colors.qualitative.Set2)
    st.plotly_chart(fig3, use_container_width=True)

st.markdown("---")

st.subheader("Time of Day Profile")
tod_order = ["Late Night", "Morning", "Afternoon", "Evening"]
top5 = df["Primary Type"].value_counts().head(5).index
tod = df[df["Primary Type"].isin(top5)].groupby(
    ["Time_of_Day", "Primary Type"]).size().reset_index(name="Count")
fig4 = px.bar(tod, x="Time_of_Day", y="Count", color="Primary Type",
              barmode="stack", category_orders={"Time_of_Day": tod_order},
              title="Top 5 Crime Types by Time of Day")
st.plotly_chart(fig4, use_container_width=True)

st.markdown("---")

st.subheader("Temporal Cluster Profiles (KMeans k=4)")
profile = df.groupby("Temporal_Cluster").agg(
    Total_Crimes=("Primary Type", "count"),
    Avg_Hour=("Hour", "mean"),
    Avg_Month=("Month", "mean"),
    Weekend_Pct=("Is_Weekend", "mean"),
    Top_Crime=("Primary Type", lambda x: x.value_counts().index[0]),
    Avg_Severity=("Crime_Severity_Score", "mean")
).reset_index()
profile["Weekend_Pct"] = (profile["Weekend_Pct"] * 100).round(1)
profile["Avg_Hour"] = profile["Avg_Hour"].round(1)
profile["Avg_Severity"] = profile["Avg_Severity"].round(2)
st.dataframe(profile, use_container_width=True)

hour_cluster = df.groupby(["Hour", "Temporal_Cluster"]).size().reset_index(name="Count")
fig5 = px.line(hour_cluster, x="Hour", y="Count",
               color=hour_cluster["Temporal_Cluster"].astype(str),
               markers=True, title="Temporal Clusters by Hour of Day",
               labels={"color": "Cluster"})
st.plotly_chart(fig5, use_container_width=True)

st.markdown("---")

st.subheader("High-Risk Time Slots")
risk = df.groupby("Hour").agg(
    Crime_Count=("Primary Type", "count"),
    Violent_Crimes=("Crime_Severity_Score", lambda x: (x >= 7).sum()),
    Avg_Severity=("Crime_Severity_Score", "mean")
).reset_index()
risk["Risk_Score"] = (risk["Crime_Count"] / risk["Crime_Count"].max() * 0.5 +
                      risk["Avg_Severity"] / 10 * 0.5).round(3)
st.dataframe(risk.sort_values("Risk_Score", ascending=False).head(8), use_container_width=True)
