import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(page_title="EDA", page_icon="📊", layout="wide")
st.title("📊 Exploratory Data Analysis")
st.markdown("---")

DATA_PATH = os.path.join("data", "processed.parquet")

@st.cache_data
def load_data():
    return pd.read_parquet(DATA_PATH)

if not os.path.exists(DATA_PATH):
    st.warning("Run `python data_pipeline.py` first to generate processed data.")
    st.stop()

df = load_data()

# Overview metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Records", f"{len(df):,}")
col2.metric("Crime Types", df["Primary Type"].nunique())
col3.metric("Arrest Rate", f"{df['Arrest'].mean()*100:.1f}%")
col4.metric("Domestic Cases", f"{df['Domestic'].mean()*100:.1f}%")

st.markdown("---")

# Crime type distribution
st.subheader("Crime Type Distribution")
crime_counts = df["Primary Type"].value_counts().reset_index()
crime_counts.columns = ["Crime Type", "Count"]
fig1 = px.bar(crime_counts.head(15), x="Count", y="Crime Type",
              orientation="h", color="Count", color_continuous_scale="Reds",
              title="Top 15 Crime Types")
fig1.update_layout(yaxis={"categoryorder": "total ascending"}, height=500)
st.plotly_chart(fig1, use_container_width=True)

st.markdown("---")

col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Crimes by Hour of Day")
    hourly = df.groupby("Hour").size().reset_index(name="Count")
    fig2 = px.line(hourly, x="Hour", y="Count", markers=True,
                   title="Crime Frequency by Hour")
    fig2.update_traces(line_color="#e74c3c")
    st.plotly_chart(fig2, use_container_width=True)

with col_b:
    st.subheader("Crimes by Day of Week")
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    daily = df.groupby("Day_of_Week").size().reset_index(name="Count")
    daily["Day_of_Week"] = pd.Categorical(daily["Day_of_Week"], categories=day_order, ordered=True)
    daily = daily.sort_values("Day_of_Week")
    fig3 = px.bar(daily, x="Day_of_Week", y="Count", color="Count",
                  color_continuous_scale="Blues", title="Crime by Day of Week")
    st.plotly_chart(fig3, use_container_width=True)

st.markdown("---")

col_c, col_d = st.columns(2)

with col_c:
    st.subheader("Monthly Crime Trend")
    month_map = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                 7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    monthly = df.groupby("Month").size().reset_index(name="Count")
    monthly["Month_Name"] = monthly["Month"].map(month_map)
    fig4 = px.bar(monthly, x="Month_Name", y="Count", color="Count",
                  color_continuous_scale="Oranges", title="Crimes Per Month")
    st.plotly_chart(fig4, use_container_width=True)

with col_d:
    st.subheader("Crime by Season")
    seasonal = df.groupby("Season").size().reset_index(name="Count")
    fig5 = px.pie(seasonal, names="Season", values="Count",
                  title="Crime Distribution by Season",
                  color_discrete_sequence=px.colors.qualitative.Set2)
    st.plotly_chart(fig5, use_container_width=True)

st.markdown("---")

st.subheader("Arrest Rate by Crime Type (Top 10)")
arrest_df = df.groupby("Primary Type").agg(
    Total=("Arrest", "count"),
    Arrested=("Arrest", "sum")
).reset_index()
arrest_df["Arrest_Rate"] = (arrest_df["Arrested"] / arrest_df["Total"] * 100).round(1)
arrest_df = arrest_df.sort_values("Total", ascending=False).head(10)
fig6 = px.bar(arrest_df, x="Primary Type", y="Arrest_Rate",
              color="Arrest_Rate", color_continuous_scale="Greens",
              title="Arrest Rate % by Crime Type")
fig6.update_layout(xaxis_tickangle=-30)
st.plotly_chart(fig6, use_container_width=True)

st.markdown("---")

st.subheader("Geographic Crime Distribution (10K sample)")
geo_sample = df.sample(min(10000, len(df)), random_state=42)
fig7 = px.scatter_mapbox(geo_sample, lat="Latitude", lon="Longitude",
                          color="Crime_Severity_Score",
                          color_continuous_scale="YlOrRd",
                          zoom=10, height=500,
                          mapbox_style="carto-positron",
                          title="Crime Locations - Color by Severity",
                          opacity=0.4)
st.plotly_chart(fig7, use_container_width=True)

st.markdown("---")

st.subheader("Hour × Day Heatmap")
heatmap_data = df.groupby(["Day_of_Week", "Hour"]).size().unstack(fill_value=0)
day_order2 = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
heatmap_data = heatmap_data.reindex([d for d in day_order2 if d in heatmap_data.index])
fig8 = px.imshow(heatmap_data, color_continuous_scale="Reds",
                 labels=dict(x="Hour of Day", y="Day of Week", color="Crimes"),
                 title="Crime Frequency Heatmap")
fig8.update_layout(height=400)
st.plotly_chart(fig8, use_container_width=True)
