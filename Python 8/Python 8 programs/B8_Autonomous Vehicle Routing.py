import numpy as np
import pandas as pd
print("SY-5, Kevin Victor, Roll No.-30")
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# -------------------------------
# STEP 1: Create Realistic Dataset
# -------------------------------

np.random.seed(42)
n_samples = 600

data = pd.DataFrame({
    # Environmental + Traffic
    "traffic_density": np.random.uniform(0.1, 1.0, n_samples),
    "weather_severity": np.random.uniform(0, 1, n_samples),
    "road_complexity": np.random.uniform(1, 10, n_samples),

    # Sensor + perception
    "obstacle_count": np.random.randint(0, 20, n_samples),
    "lidar_accuracy": np.random.uniform(0.8, 1.0, n_samples),
    "gps_signal_strength": np.random.uniform(0.5, 1.0, n_samples),

    # Vehicle dynamics
    "vehicle_speed": np.random.uniform(20, 100, n_samples),
    "battery_level": np.random.uniform(20, 100, n_samples),

    # Network / orchestrator layer
    "network_latency": np.random.uniform(10, 200, n_samples),
    "fleet_density": np.random.uniform(0.1, 1.0, n_samples),

    # Logistics
    "delivery_priority": np.random.randint(1, 5, n_samples)
})

# -------------------------------
# STEP 2: Feature Engineering
# -------------------------------

data["congestion_index"] = (
    data["traffic_density"] * data["obstacle_count"] * data["road_complexity"]
)

data["route_efficiency"] = (
    data["vehicle_speed"] * data["lidar_accuracy"] / (data["traffic_density"] + 0.1)
)

data["network_reliability"] = (
    data["gps_signal_strength"] / (data["network_latency"] + 1)
)

data["risk_factor"] = (
    data["weather_severity"] * data["road_complexity"] * (1 - data["lidar_accuracy"])
)

# -------------------------------
# STEP 3: Target Variable (Multi-class)
# -------------------------------
# 0 = Route A (fastest)
# 1 = Route B (balanced)
# 2 = Route C (safest)

conditions = [
    (data["route_efficiency"] > 200),
    (data["risk_factor"] < 1),
]

choices = [0, 1]
data["optimal_route"] = np.select(conditions, choices, default=2)

# -------------------------------
# STEP 4: Display All Columns
# -------------------------------

pd.set_option('display.max_columns', None)

# -------------------------------
# STEP 5: Introduce Missing Values (ONLY in features)
# -------------------------------

for col in data.columns:
    if col != "optimal_route":   # Protect target variable
        data.loc[data.sample(frac=0.05).index, col] = np.nan

# Safety check (just in case)
data = data.dropna(subset=["optimal_route"])

print("\nInitial Dataset:\n")
print(data.head())

# -------------------------------
# STEP 6: Separate Features & Target
# -------------------------------

X = data.drop("optimal_route", axis=1)
y = data["optimal_route"]

# -------------------------------
# STEP 7: Imputation
# -------------------------------

imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(X)

# -------------------------------
# STEP 8: Scaling
# -------------------------------

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# -------------------------------
# STEP 9: Train-Test Split
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# -------------------------------
# STEP 10: Train Models
# -------------------------------

# Logistic Regression (multi-class)
log_model = LogisticRegression(max_iter=500, multi_class='multinomial')
log_model.fit(X_train, y_train)

# KNN Model
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

# -------------------------------
# STEP 11: Predictions
# -------------------------------

log_preds = log_model.predict(X_test)
knn_preds = knn_model.predict(X_test)

# -------------------------------
# STEP 12: Evaluation
# -------------------------------

print("\nModel Comparison:\n")

print("Logistic Regression Accuracy:",
      accuracy_score(y_test, log_preds))

print("KNN Accuracy:",
      accuracy_score(y_test, knn_preds))

# -------------------------------
# STEP 13: Prediction Comparison
# -------------------------------

comparison = pd.DataFrame({
    "Actual": y_test.values,
    "Logistic_Pred": log_preds,
    "KNN_Pred": knn_preds
})

print("\nPrediction Comparison:\n")
print(comparison.head(10))