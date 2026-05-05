import pandas as pd
import numpy as np
import os
import warnings
import mlflow
import mlflow.sklearn

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
CSV_FILE = "chicago_crimes.csv"      # put your downloaded CSV here, rename to this
SAMPLE_SIZE = 500000
RANDOM_STATE = 42
OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MLFLOW_URI = os.path.abspath("mlruns")
mlflow.set_tracking_uri(f"file://{MLFLOW_URI}")

SEVERITY_MAP = {
    "HOMICIDE": 10,
    "CRIM SEXUAL ASSAULT": 9,
    "KIDNAPPING": 9,
    "ROBBERY": 8,
    "ASSAULT": 7,
    "BATTERY": 7,
    "ARSON": 7,
    "WEAPONS VIOLATION": 6,
    "BURGLARY": 6,
    "MOTOR VEHICLE THEFT": 5,
    "THEFT": 4,
    "CRIMINAL DAMAGE": 4,
    "STALKING": 4,
    "INTIMIDATION": 4,
    "NARCOTICS": 3,
    "SEX OFFENSE": 5,
    "OFFENSE INVOLVING CHILDREN": 6,
    "CRIMINAL TRESPASS": 2,
    "DECEPTIVE PRACTICE": 3,
    "PUBLIC PEACE VIOLATION": 2,
    "GAMBLING": 1,
    "LIQUOR LAW VIOLATION": 1,
    "INTERFERENCE WITH PUBLIC OFFICER": 2,
    "OBSCENITY": 2,
}


# ─────────────────────────────────────────────
# STEP 1: LOAD AND SAMPLE
# ─────────────────────────────────────────────
def load_and_sample():
    print(f"\n[STEP 1] Reading {CSV_FILE} ...")
    df = pd.read_csv(CSV_FILE, low_memory=False)
    print(f"  Full dataset: {len(df):,} records")

    if len(df) > SAMPLE_SIZE:
        df = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE).reset_index(drop=True)
        print(f"  Sampled: {len(df):,} records")
    else:
        print(f"  Dataset smaller than 500K, using all {len(df):,} records")

    return df


# ─────────────────────────────────────────────
# STEP 2: CLEAN DATA
# ─────────────────────────────────────────────
def clean_data(df):
    print("\n[STEP 2] Cleaning data ...")
    before = len(df)

    df = df.dropna(subset=["Latitude", "Longitude"])
    df = df[(df["Latitude"] >= 41.6) & (df["Latitude"] <= 42.0)]
    df = df[(df["Longitude"] >= -87.9) & (df["Longitude"] <= -87.5)]

    for col in ["Primary Type", "Description", "Location Description"]:
        if col in df.columns:
            df[col] = df[col].fillna("UNKNOWN")

    for col in ["District", "Ward", "Community Area", "Beat"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    for col in ["Arrest", "Domestic"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.upper().map({"TRUE": 1, "FALSE": 0}).fillna(0).astype(int)

    print(f"  Before: {before:,}  |  After: {len(df):,}  |  Dropped: {before - len(df):,}")
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────
# STEP 3: FEATURE ENGINEERING
# ─────────────────────────────────────────────
def engineer_features(df):
    print("\n[STEP 3] Engineering features ...")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).reset_index(drop=True)

    df["Hour"] = df["Date"].dt.hour
    df["Day_of_Week"] = df["Date"].dt.day_name()
    df["Day_of_Week_Num"] = df["Date"].dt.dayofweek
    df["Month"] = df["Date"].dt.month
    df["Year"] = df["Date"].dt.year
    df["Is_Weekend"] = df["Day_of_Week_Num"].isin([5, 6]).astype(int)

    def season(month):
        if month in [12, 1, 2]: return "Winter"
        elif month in [3, 4, 5]: return "Spring"
        elif month in [6, 7, 8]: return "Summer"
        else: return "Fall"

    def time_of_day(hour):
        if hour < 6: return "Late Night"
        elif hour < 12: return "Morning"
        elif hour < 18: return "Afternoon"
        else: return "Evening"

    df["Season"] = df["Month"].apply(season)
    df["Time_of_Day"] = df["Hour"].apply(time_of_day)
    df["Crime_Severity_Score"] = df["Primary Type"].map(SEVERITY_MAP).fillna(3).astype(int)

    le = LabelEncoder()
    df["Crime_Type_Encoded"] = le.fit_transform(df["Primary Type"].astype(str))

    print(f"  Final shape: {df.shape}")
    return df


# ─────────────────────────────────────────────
# STEP 4: GEOGRAPHIC CLUSTERING
# ─────────────────────────────────────────────
def run_geographic_clustering(df):
    print("\n[STEP 4] Geographic Clustering ...")

    coords = df[["Latitude", "Longitude"]].values
    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(coords)

    mlflow.set_experiment("Geographic_Clustering")

    # --- KMeans ---
    print("  Running KMeans (k=7) ...")
    with mlflow.start_run(run_name="KMeans_Geo_k7"):
        km = KMeans(n_clusters=7, random_state=RANDOM_STATE, n_init=10)
        df["KMeans_Geo_Cluster"] = km.fit_predict(coords_scaled)
        sil_km = silhouette_score(coords_scaled, df["KMeans_Geo_Cluster"])
        db_km = davies_bouldin_score(coords_scaled, df["KMeans_Geo_Cluster"])
        mlflow.log_param("algorithm", "KMeans")
        mlflow.log_param("n_clusters", 7)
        mlflow.log_metric("silhouette_score", sil_km)
        mlflow.log_metric("davies_bouldin_score", db_km)
        mlflow.log_metric("inertia", km.inertia_)
        print(f"    KMeans  → Silhouette: {sil_km:.4f} | Davies-Bouldin: {db_km:.4f}")

    # --- DBSCAN ---
    print("  Running DBSCAN ...")
    with mlflow.start_run(run_name="DBSCAN_Geo"):
        # use a sample for DBSCAN speed
        sample_idx = np.random.choice(len(coords_scaled), min(50000, len(coords_scaled)), replace=False)
        db_model = DBSCAN(eps=0.03, min_samples=50, n_jobs=-1)
        labels_db = db_model.fit_predict(coords_scaled[sample_idx])
        n_clusters_db = len(set(labels_db)) - (1 if -1 in labels_db else 0)
        noise_db = int(np.sum(labels_db == -1))

        # assign back to full df
        df["DBSCAN_Geo_Cluster"] = -1
        df.iloc[sample_idx, df.columns.get_loc("DBSCAN_Geo_Cluster")] = labels_db

        sil_db = -1
        mask = labels_db != -1
        if n_clusters_db > 1 and mask.sum() > 1:
            sil_db = silhouette_score(coords_scaled[sample_idx][mask], labels_db[mask])

        mlflow.log_param("algorithm", "DBSCAN")
        mlflow.log_param("eps", 0.03)
        mlflow.log_param("min_samples", 50)
        mlflow.log_metric("n_clusters_found", n_clusters_db)
        mlflow.log_metric("noise_points", noise_db)
        if sil_db != -1:
            mlflow.log_metric("silhouette_score", sil_db)
        print(f"    DBSCAN  → Clusters: {n_clusters_db} | Noise: {noise_db:,} | Silhouette: {sil_db:.4f}")

    # --- Hierarchical ---
    print("  Running Hierarchical Clustering (sample 10K) ...")
    with mlflow.start_run(run_name="Hierarchical_Geo_k7"):
        hc_idx = np.random.choice(len(coords_scaled), min(10000, len(coords_scaled)), replace=False)
        hc = AgglomerativeClustering(n_clusters=7, linkage="ward")
        hc_labels = hc.fit_predict(coords_scaled[hc_idx])
        sil_hc = silhouette_score(coords_scaled[hc_idx], hc_labels)
        db_hc = davies_bouldin_score(coords_scaled[hc_idx], hc_labels)

        df["HC_Geo_Cluster"] = -1
        df.iloc[hc_idx, df.columns.get_loc("HC_Geo_Cluster")] = hc_labels

        mlflow.log_param("algorithm", "AgglomerativeClustering")
        mlflow.log_param("n_clusters", 7)
        mlflow.log_param("linkage", "ward")
        mlflow.log_metric("silhouette_score", sil_hc)
        mlflow.log_metric("davies_bouldin_score", db_hc)
        print(f"    Hierarchical → Silhouette: {sil_hc:.4f} | Davies-Bouldin: {db_hc:.4f}")

    print("\n  ── Geo Clustering Summary ──")
    print(f"  KMeans      : Silhouette={sil_km:.4f}  DB={db_km:.4f}")
    print(f"  DBSCAN      : Silhouette={sil_db:.4f}  Clusters={n_clusters_db}")
    print(f"  Hierarchical: Silhouette={sil_hc:.4f}  DB={db_hc:.4f}")

    return df


# ─────────────────────────────────────────────
# STEP 5: TEMPORAL CLUSTERING
# ─────────────────────────────────────────────
def run_temporal_clustering(df):
    print("\n[STEP 5] Temporal Clustering ...")

    temp_cols = ["Hour", "Day_of_Week_Num", "Month", "Is_Weekend", "Crime_Severity_Score"]
    temp_cols = [c for c in temp_cols if c in df.columns]
    X_temp = df[temp_cols].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_temp)

    mlflow.set_experiment("Temporal_Clustering")

    with mlflow.start_run(run_name="KMeans_Temporal_k4"):
        km_temp = KMeans(n_clusters=4, random_state=RANDOM_STATE, n_init=10)
        df["Temporal_Cluster"] = km_temp.fit_predict(X_scaled)
        sil_t = silhouette_score(X_scaled, df["Temporal_Cluster"])
        db_t = davies_bouldin_score(X_scaled, df["Temporal_Cluster"])
        mlflow.log_param("algorithm", "KMeans")
        mlflow.log_param("n_clusters", 4)
        mlflow.log_param("feature_type", "temporal")
        mlflow.log_metric("silhouette_score", sil_t)
        mlflow.log_metric("davies_bouldin_score", db_t)
        print(f"  Temporal KMeans → Silhouette: {sil_t:.4f} | Davies-Bouldin: {db_t:.4f}")

    # Print cluster profiles
    profile = df.groupby("Temporal_Cluster").agg(
        Count=("Primary Type", "count"),
        Avg_Hour=("Hour", "mean"),
        Weekend_Pct=("Is_Weekend", "mean"),
        Top_Crime=("Primary Type", lambda x: x.value_counts().index[0])
    )
    print("\n  Temporal Cluster Profiles:")
    print(profile.to_string())

    return df


# ─────────────────────────────────────────────
# STEP 6: DIMENSIONALITY REDUCTION
# ─────────────────────────────────────────────
def run_dimensionality_reduction(df):
    print("\n[STEP 6] Dimensionality Reduction ...")

    features = ["Latitude", "Longitude", "Hour", "Day_of_Week_Num",
                "Month", "Is_Weekend", "Crime_Severity_Score",
                "Crime_Type_Encoded", "District", "Beat", "Arrest", "Domestic"]
    features = [f for f in features if f in df.columns]

    X = df[features].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    mlflow.set_experiment("Dimensionality_Reduction")

    # --- PCA ---
    print("  Running PCA ...")
    with mlflow.start_run(run_name="PCA_3_components"):
        pca = PCA(n_components=3, random_state=RANDOM_STATE)
        X_pca = pca.fit_transform(X_scaled)
        explained = pca.explained_variance_ratio_
        cumulative = float(np.cumsum(explained)[-1])
        for i, v in enumerate(explained):
            mlflow.log_metric(f"explained_variance_pc{i+1}", float(v))
        mlflow.log_metric("cumulative_variance", cumulative)
        mlflow.log_param("n_components", 3)
        mlflow.log_param("n_input_features", len(features))
        print(f"  PCA → PC1: {explained[0]*100:.1f}%  PC2: {explained[1]*100:.1f}%  PC3: {explained[2]*100:.1f}%  Total: {cumulative*100:.1f}%")

    df["PCA_1"] = X_pca[:, 0]
    df["PCA_2"] = X_pca[:, 1]
    df["PCA_3"] = X_pca[:, 2]

    # Feature loadings
    comp_df = pd.DataFrame(np.abs(pca.components_), columns=features,
                           index=["PC1", "PC2", "PC3"])
    print("\n  Top features per component:")
    for pc in ["PC1", "PC2", "PC3"]:
        top = comp_df.loc[pc].sort_values(ascending=False).head(3)
        print(f"    {pc}: {', '.join([f'{k}({v:.3f})' for k, v in top.items()])}")

    # Save loadings
    comp_df.to_csv(os.path.join(OUTPUT_DIR, "pca_loadings.csv"))

    # --- t-SNE ---
    print("\n  Running t-SNE (sample 5000) ...")
    tsne_n = min(5000, len(df))
    tsne_idx = np.random.choice(len(df), tsne_n, replace=False)
    X_tsne_input = X_pca[tsne_idx]  # use PCA output as input to t-SNE

    with mlflow.start_run(run_name="tSNE_2D"):
        tsne = TSNE(n_components=2, perplexity=30, random_state=RANDOM_STATE,
                    learning_rate="auto", init="pca", max_iter=500)
        X_tsne = tsne.fit_transform(X_tsne_input)
        mlflow.log_param("perplexity", 30)
        mlflow.log_param("sample_size", tsne_n)
        mlflow.log_metric("kl_divergence", float(tsne.kl_divergence_))
        print(f"  t-SNE → KL Divergence: {tsne.kl_divergence_:.4f}")

    # Save t-SNE result for app
    tsne_df = pd.DataFrame({
        "tSNE_1": X_tsne[:, 0],
        "tSNE_2": X_tsne[:, 1],
        "Primary_Type": df.iloc[tsne_idx]["Primary Type"].values,
        "Time_of_Day": df.iloc[tsne_idx]["Time_of_Day"].values,
        "Season": df.iloc[tsne_idx]["Season"].values,
        "Crime_Severity_Score": df.iloc[tsne_idx]["Crime_Severity_Score"].values,
    })
    tsne_df.to_parquet(os.path.join(OUTPUT_DIR, "tsne_result.parquet"), index=False)
    print("  t-SNE result saved.")

    return df


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  PatrolIQ - Data Pipeline")
    print("=" * 55)

    if not os.path.exists(CSV_FILE):
        print(f"\nERROR: '{CSV_FILE}' not found in current directory.")
        print("Download from: https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2")
        print("Rename it to 'chicago_crimes.csv' and place it here.")
        exit(1)

    df = load_and_sample()
    df = clean_data(df)
    df = engineer_features(df)
    df = run_geographic_clustering(df)
    df = run_temporal_clustering(df)
    df = run_dimensionality_reduction(df)

    out_path = os.path.join(OUTPUT_DIR, "processed.parquet")
    df.to_parquet(out_path, index=False)
    print(f"\n[DONE] Processed data saved to '{out_path}'")
    print(f"       Columns: {list(df.columns)}")
    print(f"       Shape  : {df.shape}")
    print("\nNow run:  streamlit run app.py")
    print("=" * 55)
