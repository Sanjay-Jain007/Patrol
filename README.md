# PatrolIQ - Smart Safety Analytics Platform
GUVI Capstone Project | Chicago Crime Analysis

---

## Project Structure

```
patroliq/
├── chicago_crimes.csv         ← PUT YOUR DOWNLOADED CSV HERE (rename to this)
├── data_pipeline.py           ← Run this FIRST — reads CSV, does all ML, saves outputs
├── app.py                     ← Streamlit home page
├── requirements.txt
├── pages/
│   ├── 1_EDA.py
│   ├── 2_Geographic_Clustering.py
│   ├── 3_Temporal_Clustering.py
│   ├── 4_Dimensionality_Reduction.py
│   └── 5_MLflow_Tracker.py
├── data/                      ← Auto-created by pipeline
│   ├── processed.parquet
│   ├── tsne_result.parquet
│   └── pca_loadings.csv
└── mlruns/                    ← MLflow experiment logs
```

---

## How to Run

### Step 1 — Download the dataset
Go to: https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2
Click Export → CSV. Rename the downloaded file to `chicago_crimes.csv` and place it
in the project folder.

### Step 2 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Run the data pipeline
```bash
python data_pipeline.py
```
This will:
- Read chicago_crimes.csv
- Randomly sample 500,000 records
- Clean and engineer features
- Run KMeans, DBSCAN, Hierarchical clustering
- Run PCA and t-SNE
- Log all metrics to MLflow
- Save processed.parquet, tsne_result.parquet, pca_loadings.csv

Takes about 5-10 minutes depending on your machine.

### Step 4 — Launch the Streamlit app
```bash
streamlit run app.py
```
Opens at http://localhost:8501

### Optional — View MLflow UI
```bash
mlflow ui --backend-store-uri ./mlruns
```
Opens at http://localhost:5000
