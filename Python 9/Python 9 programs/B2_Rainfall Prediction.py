# Rainfall Prediction using Logistic Regression + Confusion Matrix

import numpy as np
import pandas as pd
print("SY-5, Kevin Victor, Roll No.-30")
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# -------------------------------
# STEP 1: Create Realistic Dataset
# -------------------------------

np.random.seed(42)
n_samples = 500

data = pd.DataFrame({
    "temperature": np.random.uniform(15, 40, n_samples),            # °C
    "humidity": np.random.uniform(30, 100, n_samples),              # %
    "wind_speed": np.random.uniform(0, 25, n_samples),              # km/h
    "pressure": np.random.uniform(980, 1030, n_samples),            # hPa
    "cloud_cover": np.random.uniform(0, 100, n_samples),            # %
    "ocean_temp": np.random.uniform(20, 32, n_samples),             # °C
    "enso_index": np.random.uniform(-2, 2, n_samples),              # ENSO index
    "instability_index": np.random.uniform(0, 1, n_samples)         # local instability
})

# -------------------------------
# STEP 2: Define Rainfall Logic
# -------------------------------

# Strong realistic conditions for rainfall
data["rain"] = (
    (data["humidity"] > 65).astype(int) +
    (data["cloud_cover"] > 50).astype(int) +
    (data["pressure"] < 1005).astype(int) +
    (data["instability_index"] > 0.6).astype(int) +
    (data["enso_index"] > 0.5).astype(int)
)

# If 3 or more conditions satisfied → Rain
data["rain"] = (data["rain"] >= 3).astype(int)

# -------------------------------
# STEP 3: Features & Target
# -------------------------------

X = data.drop("rain", axis=1)
y = data["rain"]

# -------------------------------
# STEP 4: Train-Test Split
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# STEP 5: Feature Scaling
# -------------------------------

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------------
# STEP 6: Train Logistic Regression
# -------------------------------

model = LogisticRegression()
model.fit(X_train, y_train)

# -------------------------------
# STEP 7: Predictions
# -------------------------------

y_pred = model.predict(X_test)

# -------------------------------
# STEP 8: Evaluation Metrics
# -------------------------------

print("\nModel Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# -------------------------------
# STEP 9: Confusion Matrix
# -------------------------------

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:\n")
print(cm)

# -------------------------------
# STEP 10: Explanation of Confusion Matrix
# -------------------------------

tn, fp, fn, tp = cm.ravel()

print("\n--- Confusion Matrix Explanation ---\n")

print(f"True Negatives (TN): {tn}")
print("-> Model correctly predicted NO RAIN when there was actually no rain.\n")

print(f"False Positives (FP): {fp}")
print("-> Model predicted RAIN, but actually there was NO RAIN (False Alarm).\n")

print(f"False Negatives (FN): {fn}")
print("-> Model predicted NO RAIN, but actually it rained (Missed Rainfall).\n")

print(f"True Positives (TP): {tp}")
print("-> Model correctly predicted RAIN when it actually rained.\n")

print("Interpretation Summary:")
print("- High TP and TN → good model")
print("- High FP → too many false alarms")
print("- High FN → dangerous (missed rainfall prediction)")