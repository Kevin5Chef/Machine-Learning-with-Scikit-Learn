import numpy as np
import pandas as pd
print("SY-5, Kevin Victor, Roll No.-30")
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# -------------------------------
# STEP 1: Simulate Realistic Dataset
# -------------------------------

np.random.seed(42)
n = 500

data = pd.DataFrame({
    "air_traffic_density": np.random.uniform(0.2, 1.0, n),
    "runway_occupancy": np.random.uniform(0.1, 1.0, n),
    "weather_severity": np.random.uniform(0, 1, n),
    "visibility": np.random.uniform(1, 10, n),
    "fuel_urgency": np.random.uniform(0, 1, n),
    "holding_aircraft": np.random.randint(0, 15, n),
    "ground_traffic": np.random.randint(0, 20, n),
    "departure_queue": np.random.randint(0, 20, n),
    "arrival_queue": np.random.randint(0, 20, n),
})

# -------------------------------
# STEP 2: Feature Engineering
# -------------------------------

data["congestion_index"] = (
    data["air_traffic_density"] * data["holding_aircraft"] * data["runway_occupancy"]
)

data["risk_score"] = (
    data["weather_severity"] * (1 / data["visibility"]) * data["air_traffic_density"]
)

data["priority_score"] = (
    data["fuel_urgency"] * 2 + data["arrival_queue"] * 0.5
)

data["cascade_risk"] = (
    data["departure_queue"] * data["ground_traffic"] * data["air_traffic_density"]
)

# -------------------------------
# STEP 3: Target Variable (Decision)
# -------------------------------
# 0 = Hold
# 1 = Allow Landing
# 2 = Allow Takeoff
# 3 = Emergency Priority

conditions = [
    (data["fuel_urgency"] > 0.8),
    (data["risk_score"] < 0.2) & (data["arrival_queue"] > data["departure_queue"]),
    (data["risk_score"] < 0.2)
]

choices = [3, 1, 2]
data["decision"] = np.select(conditions, choices, default=0)

# -------------------------------
# STEP 4: Introduce Missing Values (ONLY features)
# -------------------------------

for col in data.columns:
    if col != "decision":
        data.loc[data.sample(frac=0.05).index, col] = np.nan

pd.set_option('display.max_columns', None)

print("\nInitial Dataset:\n")
print(data.head())

# -------------------------------
# STEP 5: Preprocessing
# -------------------------------

X = data.drop("decision", axis=1)
y = data["decision"]

# Imputation
imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(X)

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# -------------------------------
# STEP 6: Train Model
# -------------------------------

model = LogisticRegression(max_iter=500, multi_class='multinomial')
model.fit(X_train, y_train)

# -------------------------------
# STEP 7: Predictions
# -------------------------------

preds = model.predict(X_test)

# -------------------------------
# STEP 8: Evaluation
# -------------------------------

print("\nModel Accuracy:", accuracy_score(y_test, preds))

# -------------------------------
# STEP 9: Decision + Explanation Engine
# -------------------------------

def explain_decision(row, prediction):
    reasons = []

    # Critical triggers
    if row["fuel_urgency"] > 0.8:
        reasons.append("High fuel urgency → Emergency landing priority")

    if row["risk_score"] > 0.5:
        reasons.append("High weather/visibility risk")

    if row["congestion_index"] > 5:
        reasons.append("Severe airspace congestion")

    if row["cascade_risk"] > 50:
        reasons.append("Potential cascade failure risk")

    # Map prediction to action
    if prediction == 1:
        action = "Allow Landing"
    elif prediction == 2:
        action = "Allow Takeoff"
    elif prediction == 3:
        action = "Emergency Priority Landing"
    else:
        action = "Hold Pattern"

    # ✅ NEW: Default reasoning if no major triggers
    if len(reasons) == 0:
        reasons.append("System stable: low congestion and acceptable risk levels")
        reasons.append("Following standard queue-based scheduling (arrival/departure order)")

    return action, reasons

# -------------------------------
# STEP 10: Fallback Mechanism
# -------------------------------

def fallback_check(row, action):
    if row["cascade_risk"] > 70 or row["risk_score"] > 0.7:
        return "⚠ Fallback Activated: Re-evaluate / Delay decision"
    return "Safe"

# -------------------------------
# STEP 11: Simulate High Load Scenarios
# -------------------------------

print("\nAATC Decision Reports (High Load Scenarios):\n")

sample_indices = np.random.choice(len(X_test), 5)

for i in sample_indices:
    original = X.iloc[y_test.index[i]]
    pred = preds[i]

    action, reasons = explain_decision(original, pred)
    fallback = fallback_check(original, action)

    print("---- Scenario ----")
    print("Decision:", action)
    print("Reasons:", reasons)
    print("Fallback Status:", fallback)
    print()