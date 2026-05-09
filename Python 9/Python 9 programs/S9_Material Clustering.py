# ============================================================
# IMPROVED MATERIAL CLUSTERING SYSTEM (FUSION-GRADE)
# ============================================================
print("SY-5, Kevin Victor, Roll No.-30")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold

# Show ALL columns
pd.set_option('display.max_columns', None)

# -----------------------------
# STEP 1: STRUCTURED DATASET (REAL CLUSTERS)
# -----------------------------

np.random.seed(42)
n = 150

# Cluster 1: Plasma-facing (high temp, low durability)
cluster1 = pd.DataFrame({
    "max_temp_tolerance": np.random.normal(3800, 150, n),
    "thermal_conductivity": np.random.normal(300, 30, n),
    "radiation_resistance": np.random.uniform(0.7, 1.0, n),
    "mechanical_strength": np.random.normal(400, 80, n),
    "erosion_resistance": np.random.uniform(0.6, 1.0, n),
    "fatigue_life": np.random.normal(2000, 500, n),
    "density": np.random.uniform(15, 20, n),
    "cost_index": np.random.uniform(7, 10, n)
})

# Cluster 2: Structural materials (moderate temp, high durability)
cluster2 = pd.DataFrame({
    "max_temp_tolerance": np.random.normal(2500, 150, n),
    "thermal_conductivity": np.random.normal(150, 30, n),
    "radiation_resistance": np.random.uniform(0.5, 0.8, n),
    "mechanical_strength": np.random.normal(900, 100, n),
    "erosion_resistance": np.random.uniform(0.4, 0.7, n),
    "fatigue_life": np.random.normal(8000, 1000, n),
    "density": np.random.uniform(5, 10, n),
    "cost_index": np.random.uniform(4, 7, n)
})

# Cluster 3: Experimental composites (balanced)
cluster3 = pd.DataFrame({
    "max_temp_tolerance": np.random.normal(3200, 200, n),
    "thermal_conductivity": np.random.normal(220, 40, n),
    "radiation_resistance": np.random.uniform(0.6, 0.9, n),
    "mechanical_strength": np.random.normal(700, 120, n),
    "erosion_resistance": np.random.uniform(0.5, 0.8, n),
    "fatigue_life": np.random.normal(5000, 800, n),
    "density": np.random.uniform(8, 14, n),
    "cost_index": np.random.uniform(5, 8, n)
})

data = pd.concat([cluster1, cluster2, cluster3], ignore_index=True)

# -----------------------------
# STEP 2: AGGRESSIVE FEATURE ENGINEERING
# -----------------------------

data["thermal_efficiency"] = data["thermal_conductivity"] * data["max_temp_tolerance"]
data["durability_index"] = data["fatigue_life"] * data["mechanical_strength"]
data["resilience_score"] = data["radiation_resistance"] * data["erosion_resistance"]

# NEW FEATURES
data["thermal_stress_factor"] = data["max_temp_tolerance"] / data["mechanical_strength"]
data["performance_cost_ratio"] = data["durability_index"] / (data["cost_index"] + 1)
data["stability_index"] = data["resilience_score"] * data["fatigue_life"]
data["heat_resistance_index"] = data["thermal_efficiency"] * data["radiation_resistance"]
data["degradation_rate"] = 1 / (data["fatigue_life"] + 1)

# -----------------------------
# STEP 3: PREPROCESSING
# -----------------------------

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Remove low-variance noise features
selector = VarianceThreshold(threshold=0.1)
filtered_data = selector.fit_transform(scaled_data)

# -----------------------------
# STEP 4: POOR INITIAL CLUSTERING
# -----------------------------

k_initial = 2
labels_init = KMeans(n_clusters=k_initial, random_state=42).fit_predict(filtered_data)
score_init = silhouette_score(filtered_data, labels_init)

print("\n===== INITIAL CLUSTERING =====")
print(f"K = {k_initial}, Silhouette Score = {score_init:.4f}")

# -----------------------------
# STEP 5: FIND BEST K
# -----------------------------

print("\n===== SILHOUETTE ANALYSIS =====")

scores = []
k_range = range(2, 9)

for k in k_range:
    labels = KMeans(n_clusters=k, random_state=42).fit_predict(filtered_data)
    score = silhouette_score(filtered_data, labels)
    scores.append(score)
    print(f"K = {k}, Silhouette Score = {score:.4f}")

# Plot
plt.figure()
plt.plot(k_range, scores, marker='o')
plt.xlabel("K")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Optimization")
plt.show()

best_k = k_range[np.argmax(scores)]
print(f"\nBest K: {best_k}")

# -----------------------------
# STEP 6: FINAL MODEL
# -----------------------------

kmeans = KMeans(n_clusters=best_k, random_state=42)
labels = kmeans.fit_predict(filtered_data)

data["cluster"] = labels

# -----------------------------
# STEP 7: PCA VISUALIZATION
# -----------------------------

pca = PCA(n_components=2)
reduced = pca.fit_transform(filtered_data)

plt.figure(figsize=(8,6))
scatter = plt.scatter(reduced[:,0], reduced[:,1], c=labels)
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("Material Clusters (PCA Projection)")
plt.colorbar(scatter)
plt.show()

# -----------------------------
# STEP 8: PCA INTERPRETATION
# -----------------------------

print("\n===== PCA INTERPRETATION =====")

explained = pca.explained_variance_ratio_
print(f"Variance captured by PCA1: {explained[0]:.2f}")
print(f"Variance captured by PCA2: {explained[1]:.2f}")

if explained.sum() > 0.7:
    print("→ PCA captures most structure → clusters are meaningful")
else:
    print("→ Information loss in PCA → clusters may overlap")

print("\nCluster separation insights:")

centroids = kmeans.cluster_centers_
centroids_2D = pca.transform(centroids)

for i, c in enumerate(centroids_2D):
    print(f"Cluster {i} center in PCA space: {c}")

print("\nInterpretation:")
print("- Well-separated clusters → distinct material categories")
print("- Overlapping clusters → similar material behavior")
print("- Dense grouping → stable material performance class")

# -----------------------------
# STEP 9: FULL CLUSTER SUMMARY
# -----------------------------

print("\n===== FULL CLUSTER SUMMARY =====")
print(data.groupby("cluster").mean())