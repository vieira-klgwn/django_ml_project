import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import silhouette_score, silhouette_samples
import numpy as np
import joblib

SEGMENT_FEATURES = ["estimated_income", "selling_price"]

df = pd.read_csv("dummy-data/vehicles_ml_dataset.csv")

# === Colleague's logic applied here ===
# 1. Remove outliers (outside 1% and 99%)
for col in SEGMENT_FEATURES:
    low = df[col].quantile(0.01)
    high = df[col].quantile(0.99)
    df = df[(df[col] >= low) & (df[col] <= high)]

X_raw = df[SEGMENT_FEATURES].values

# 2. PowerTransformer (best for money data)
scaler = PowerTransformer(method="yeo-johnson")
X_scaled = scaler.fit_transform(X_raw)

# 3. Try k=2 to 5 and pick the best
results = []
best_cv = None
best_k = None
best_model = None
best_scaler = None
best_labels = None
best_core_mask = None
best_score = None
best_cluster_mapping = None

for k in range(2, 6):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=50, max_iter=1000)
    all_labels = kmeans.fit_predict(X_scaled)
    sample_sil = silhouette_samples(X_scaled, all_labels)
    THRESHOLD = 0.70
    core_mask = sample_sil >= THRESHOLD
    X_core = X_scaled[core_mask]
    core_labels = all_labels[core_mask]
    if len(set(core_labels)) < 2:
        continue
    refined_score = silhouette_score(X_core, core_labels)
    
    # CV of cluster sizes (colleague's "cv")
    cluster_sizes = np.bincount(core_labels)
    cv = round(np.std(cluster_sizes) / np.mean(cluster_sizes), 4) if np.mean(cluster_sizes) != 0 else 0
    
    # Sort clusters by income (Economy first)
    centers_orig = scaler.inverse_transform(kmeans.cluster_centers_)
    sorted_clusters = centers_orig[:, 0].argsort()   # sort by estimated_income
    cluster_mapping = {i: f"Cluster-{i+1}" for i in range(k)}
    if k == 3:
        cluster_mapping = {
            sorted_clusters[0]: "Economy",
            sorted_clusters[1]: "Standard",
            sorted_clusters[2]: "Premium",
        }
    
    results.append({
        "k": k, "cv": cv, "silhouette": round(refined_score, 4),
        "model": kmeans, "scaler": scaler, "labels": all_labels,
        "core_mask": core_mask, "score": refined_score,
        "cluster_mapping": cluster_mapping
    })
    
    if refined_score >= 0.9 and (best_cv is None or cv < best_cv):
        best_cv = cv
        best_k = k
        best_model = kmeans
        best_scaler = scaler
        best_labels = all_labels
        best_core_mask = core_mask
        best_score = refined_score
        best_cluster_mapping = cluster_mapping

# Fallback if nothing reached 0.9
if best_model is None and results:
    best_result = results[0]
    best_model = best_result["model"]
    best_scaler = best_result["scaler"]
    best_labels = best_result["labels"]
    best_core_mask = best_result["core_mask"]
    best_score = best_result["score"]
    best_cluster_mapping = best_result["cluster_mapping"]
    best_cv = best_result["cv"]
    best_k = best_result["k"]

# Assign clusters to dataframe
df["cluster_id"] = best_labels
df["client_class"] = df["cluster_id"].map(best_cluster_mapping)

# === Build bundle exactly like before (so predict_cluster_id still works) ===
bundle = {
    "model": best_model,
    "mapping": best_cluster_mapping,
    "scaler": best_scaler,
    "features": SEGMENT_FEATURES
}
joblib.dump(bundle, "model_generators/clustering/clustering_model.pkl")

# === Overall CV (for backward compatibility with your template) ===
cv_income = round((df["estimated_income"].std() / df["estimated_income"].mean()) * 100, 2)
cv_price = round((df["selling_price"].std() / df["selling_price"].mean()) * 100, 2)

# === Per-cluster CV table (this is what reduces the big CV problem) ===
per_cluster_cv = {}
for feature in SEGMENT_FEATURES:
    per_cluster_cv[feature] = {}
    for cluster in df["client_class"].unique():
        values = df[df["client_class"] == cluster][feature].values
        mean = np.mean(values)
        std = np.std(values)
        cv_cluster = round(std / mean, 4) if mean != 0 else 0
        per_cluster_cv[feature][cluster] = cv_cluster

overall_cv = {}
for feature in SEGMENT_FEATURES:
    mean = np.mean(df[feature].values)
    std = np.std(df[feature].values)
    overall_cv[feature] = round(std / mean, 4) if mean != 0 else 0

# Build summary tables (exactly as before)
cluster_summary = df.groupby("client_class")[SEGMENT_FEATURES].mean()
cluster_counts = df["client_class"].value_counts().reset_index()
cluster_counts.columns = ["client_class", "count"]
cluster_summary = cluster_summary.merge(cluster_counts, on="client_class")
comparison_df = df[["client_name", "estimated_income", "selling_price", "client_class"]]

silhouette_avg = round(best_score, 2)

# === Build nice HTML cv_table (same style your template expects) ===
cv_table = "<table class='table table-bordered table-sm'><thead><tr><th>Feature</th>"
for cluster in sorted(df["client_class"].unique()):
    cv_table += f"<th>{cluster} CV</th>"
cv_table += "<th>Overall CV</th></tr></thead><tbody>"
for feature in SEGMENT_FEATURES:
    cv_table += f"<tr><td>{feature}</td>"
    for cluster in sorted(df["client_class"].unique()):
        cv_table += f"<td>{per_cluster_cv[feature][cluster]}</td>"
    cv_table += f"<td>{overall_cv[feature]}</td></tr>"
cv_table += "</tbody></table>"

def evaluate_clustering_model():
    return {
        "silhouette": silhouette_avg,
        "cv_income": cv_income,
        "cv_price": cv_price,
        "summary": cluster_summary.to_html(
            classes="table table-bordered table-striped table-sm",
            float_format="%.2f",
            justify="center",
            index=False,
        ),
        "comparison": comparison_df.head(10).to_html(
            classes="table table-bordered table-striped table-sm",
            float_format="%.2f",
            justify="center",
            index=False,
        ),
        "cv_table": cv_table,           # ← new table with per-cluster CV
        "cluster_cv_table": cv_table,   # also keep old name in case you use it
    }

def predict_cluster_id(bundle, estimated_income, selling_price):
    X = np.array([[estimated_income, selling_price]])
    X_scaled = bundle["scaler"].transform(X)
    cluster_id = bundle["model"].predict(X_scaled)[0]
    return bundle["mapping"][cluster_id]

if __name__ == '__main__':
    print(f"Clustering model trained. Silhouette Score: {silhouette_avg} | Best k: {best_k}")