# ============================================
# CUSTOMER CLUSTERING: NORMAL vs FESTIVAL DAYS
# WITH SILHOUETTE SCORE EVALUATION
# ============================================
print("SY-5, Kevin Victor, Roll No.-30")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# -----------------------------
# STEP 1: SIMULATED DATASET
# -----------------------------

np.random.seed(42)

n_customers = 300

normal_data = pd.DataFrame({
    "purchase_freq": np.random.randint(1, 10, n_customers),
    "avg_spend": np.random.normal(2000, 500, n_customers),
    "discount_sensitivity": np.random.uniform(0.2, 0.6, n_customers),
    "bulk_purchase": np.random.randint(0, 2, n_customers)
})

festival_data = pd.DataFrame({
    "purchase_freq": np.random.randint(5, 20, n_customers),
    "avg_spend": np.random.normal(5000, 1500, n_customers),
    "discount_sensitivity": np.random.uniform(0.5, 1.0, n_customers),
    "bulk_purchase": np.random.randint(0, 2, n_customers)
})

normal_data["season"] = 0
festival_data["season"] = 1

data = pd.concat([normal_data, festival_data], ignore_index=True)
data["avg_spend"] = data["avg_spend"].clip(lower=500)

# -----------------------------
# STEP 2: PREPROCESSING
# -----------------------------

features = ["purchase_freq", "avg_spend", "discount_sensitivity", "bulk_purchase"]

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data[features])

# -----------------------------
# STEP 3: K-MEANS CLUSTERING
# -----------------------------

kmeans = KMeans(n_clusters=4, random_state=42)
data["cluster"] = kmeans.fit_predict(scaled_data)

# -----------------------------
# STEP 4: VISUALIZATION
# -----------------------------

plt.figure(figsize=(10, 6))

scatter = plt.scatter(
    data["purchase_freq"],
    data["avg_spend"],
    c=data["cluster"],
)

plt.xlabel("Purchase Frequency")
plt.ylabel("Average Spend")
plt.title("Customer Clusters: Normal vs Festival Behavior")

plt.colorbar(scatter, label="Cluster")
plt.show()

# -----------------------------
# STEP 5: SILHOUETTE SCORE
# -----------------------------

score = silhouette_score(scaled_data, data["cluster"])

print("\n===== SILHOUETTE SCORE =====\n")
print(f"Score for K=4: {score:.4f}")

# -----------------------------
# STEP 6: FIND OPTIMAL K
# -----------------------------

print("\n===== SILHOUETTE ANALYSIS FOR DIFFERENT K =====\n")

k_range = range(2, 8)
scores = []

for k in k_range:
    kmeans_temp = KMeans(n_clusters=k, random_state=42)
    labels = kmeans_temp.fit_predict(scaled_data)
    s_score = silhouette_score(scaled_data, labels)
    scores.append(s_score)
    print(f"K = {k}, Silhouette Score = {s_score:.4f}")

# Plot K vs Score
plt.figure()
plt.plot(k_range, scores, marker='o')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("Optimal K using Silhouette Score")
plt.show()

# -----------------------------
# STEP 7: INTERPRETATION
# -----------------------------

print("\n===== MODEL PERFORMANCE INTERPRETATION =====\n")

if score > 0.7:
    print("Strong clustering structure detected. Clusters are well-separated.")
elif score > 0.5:
    print("Reasonable clustering. Some overlap exists, but usable for segmentation.")
elif score > 0.25:
    print("Weak clustering. Consider feature engineering or different K.")
else:
    print("Poor clustering. Model needs improvement.")

print("""
Backend Recommendations:

1. If score is low:
   - Add more features (customer age, category preference, time of purchase)
   - Improve data scaling or normalization
   - Remove noise/outliers

2. If clusters overlap:
   - Increase dimensionality (more behavioral features)
   - Try alternative algorithms (DBSCAN, Hierarchical Clustering)

3. If optimal K != 4:
   - Update production model with better K
   - Re-evaluate marketing segmentation logic

4. Festival vs Normal Insight:
   - Seasonal behavior may require separate clustering models
   - Consider training two independent models for higher accuracy
""")