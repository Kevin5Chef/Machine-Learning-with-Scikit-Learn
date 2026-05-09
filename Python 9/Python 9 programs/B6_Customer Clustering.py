# ============================================
# CUSTOMER CLUSTERING: NORMAL vs FESTIVAL DAYS
# ============================================
print("SY-5, Kevin Victor, Roll No.-30")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# -----------------------------
# STEP 1: SIMULATED DATASET
# -----------------------------

np.random.seed(42)

n_customers = 300

# NORMAL DAY DATA
normal_data = pd.DataFrame({
    "purchase_freq": np.random.randint(1, 10, n_customers),
    "avg_spend": np.random.normal(2000, 500, n_customers),
    "discount_sensitivity": np.random.uniform(0.2, 0.6, n_customers),
    "bulk_purchase": np.random.randint(0, 2, n_customers)
})

# FESTIVAL DAY DATA (behavior changes)
festival_data = pd.DataFrame({
    "purchase_freq": np.random.randint(5, 20, n_customers),
    "avg_spend": np.random.normal(5000, 1500, n_customers),
    "discount_sensitivity": np.random.uniform(0.5, 1.0, n_customers),
    "bulk_purchase": np.random.randint(0, 2, n_customers)
})

# Add label
normal_data["season"] = 0
festival_data["season"] = 1

# Combine datasets
data = pd.concat([normal_data, festival_data], ignore_index=True)

# Handle negatives (due to normal distribution)
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

# Scatter plot: Spend vs Frequency
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
# STEP 5: INSIGHT GENERATION
# -----------------------------

cluster_summary = data.groupby("cluster").mean()

pd.set_option('display.max_columns', None)

print("\n===== CLUSTER SUMMARY =====\n")
print(cluster_summary)

print("\n===== BUSINESS INSIGHTS =====\n")

for i, row in cluster_summary.iterrows():
    print(f"\nCluster {i}:")

    if row["avg_spend"] > 4000:
        print("- High spenders detected")
        print("Strategy: Premium bundles, exclusive deals, early access sales")

    if row["discount_sensitivity"] > 0.7:
        print("- Highly discount-driven customers")
        print("Strategy: Flash sales, coupons, limited-time offers")

    if row["purchase_freq"] > 10:
        print("- Frequent buyers")
        print("Strategy: Loyalty programs, subscription discounts")

    if row["bulk_purchase"] > 0.5:
        print("- Bulk buyers")
        print("Strategy: Combo offers, family packs, festival bundles")

    if row["avg_spend"] < 2500:
        print("- Low spenders")
        print("Strategy: Entry-level pricing, small packs, upselling")

# -----------------------------
# STEP 6: SEASON ANALYSIS
# -----------------------------

print("\n===== SEASONAL INSIGHTS =====\n")

season_summary = data.groupby("season")[features].mean()
print(season_summary)

print("\nINTERPRETATION:")
print("0 = Normal Days, 1 = Festival Days")

print("""
Festival days show:
- Higher spending
- More frequent purchases
- Higher discount sensitivity

Normal days show:
- Lower spending
- More stable buying patterns

Business strategy:
- Festival: Aggressive discounts and bundles
- Normal: Loyalty programs, retention strategies, and cross-selling
""")