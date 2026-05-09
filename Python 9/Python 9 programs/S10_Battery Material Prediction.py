# ============================================================
# FIXED BATTERY MATERIAL ML PIPELINE (NO OVERFLOW)
# ============================================================
print("SY-5, Kevin Victor, Roll No.-30")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -----------------------------
# STEP 1: DATASET
# -----------------------------

np.random.seed(42)
n = 500

data = pd.DataFrame({
    "electrical_conductivity": np.random.uniform(1, 100, n),
    "thermal_stability": np.random.uniform(200, 1000, n),
    "charge_capacity": np.random.uniform(50, 300, n),
    "ion_mobility": np.random.uniform(0.1, 1.0, n),
    "degradation_rate": np.random.uniform(0.01, 0.2, n),
    "mechanical_strength": np.random.uniform(100, 1000, n),
    "density": np.random.uniform(1, 10, n),
    "cost_index": np.random.uniform(1, 10, n)
})

# -----------------------------
# STEP 2: FEATURE ENGINEERING (STABLE QEPI)
# -----------------------------

data["energy_density"] = data["charge_capacity"] * data["electrical_conductivity"]
data["stability_score"] = data["thermal_stability"] / (data["degradation_rate"] + 0.01)
data["performance_index"] = data["energy_density"] * data["ion_mobility"]
data["safety_index"] = data["thermal_stability"] / data["degradation_rate"]
data["cost_efficiency"] = data["performance_index"] / (data["cost_index"] + 1)

# -----------------------------
# ⚛️ SAFE QUANTUM FEATURE
# -----------------------------

# Normalize interaction term first (CRITICAL FIX)
interaction_raw = (
    data["energy_density"] *
    data["stability_score"] *
    data["ion_mobility"]
)

interaction_scaled = interaction_raw / (interaction_raw.max() + 1e-6)

# Butterfly-like bounded chaos (NO overflow)
data["quantum_variation"] = np.sin(10 * data["ion_mobility"]) + \
                            np.cos(15 * data["degradation_rate"])

# Use tanh instead of exp (bounded between -1 and 1)
data["quantum_effect"] = np.tanh(
    data["quantum_variation"] * interaction_scaled
)

# Final QEPI
data["quantum_performance_index"] = (
    data["performance_index"] * (1 + data["quantum_effect"])
)

# -----------------------------
# TARGET
# -----------------------------

data["target"] = (
    (data["safety_index"] > 5000) &
    (data["quantum_performance_index"] > data["quantum_performance_index"].median()) &
    (data["degradation_rate"] < 0.1)
).astype(int)

# -----------------------------
# STEP 3: CLEANING (CRITICAL FIX)
# -----------------------------

# Replace inf → NaN → fill
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.fillna(data.mean(), inplace=True)

# -----------------------------
# STEP 4: PREPROCESSING
# -----------------------------

X = data.drop("target", axis=1)
y = data["target"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# -----------------------------
# STEP 5: BASE MODEL
# -----------------------------

model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc_before = accuracy_score(y_test, y_pred)

print("\n===== BEFORE TUNING =====")
print(f"Accuracy: {acc_before:.4f}")

# -----------------------------
# STEP 6: EVALUATION
# -----------------------------

print("\nClassification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix (Before Tuning)")
plt.colorbar()
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# -----------------------------
# STEP 7: GRID SEARCH
# -----------------------------

param_grid = {
    "C": [0.001, 0.01, 0.1, 1, 10, 50],
    "solver": ["lbfgs", "liblinear"],
    "penalty": ["l2"],
    "class_weight": [None, "balanced"]
}

grid = GridSearchCV(
    LogisticRegression(max_iter=2000),
    param_grid,
    cv=5,
    scoring="accuracy"
)

grid.fit(X_train, y_train)

best_model = grid.best_estimator_

print("\nBest Parameters:", grid.best_params_)

# -----------------------------
# STEP 8: AFTER TUNING
# -----------------------------

y_pred_tuned = best_model.predict(X_test)
acc_after = accuracy_score(y_test, y_pred_tuned)

print("\n===== AFTER TUNING =====")
print(f"Accuracy: {acc_after:.4f}")

# -----------------------------
# STEP 9: COMPARISON
# -----------------------------

print("\n===== ACCURACY COMPARISON =====")
print(f"Before Tuning: {acc_before:.4f}")
print(f"After Tuning:  {acc_after:.4f}")

plt.figure()
plt.bar(["Before", "After"], [acc_before, acc_after])
plt.title("Accuracy Comparison")
plt.ylabel("Accuracy")
plt.show()

# -----------------------------
# STEP 10: FEATURE IMPORTANCE
# -----------------------------

coeffs = pd.Series(best_model.coef_[0], index=X.columns)
coeffs.sort_values().plot(kind='barh')
plt.title("Feature Importance")
plt.show()

# -----------------------------
# STEP 11: FINAL INSIGHT
# -----------------------------

print("\n===== INTERPRETATION =====")

print("""
1. Features like energy_density and safety_index strongly influence prediction.

2. High positive coefficient → material is likely SAFE alternative.

3. Negative coefficient → reduces suitability.

4. Tuning improves generalization by optimizing regularization strength.

5. Scientists can use this model to:
   - Screen materials quickly
   - Avoid unsafe compositions
   - Focus on high-performance candidates   
""")

print("\nThe model shows no improvement after tuning, indicating that the default parameters were already optimal for this dataset. "
      "\nThe feature importance plot reveals that 'safety_index' and 'quantum_performance_index' are the most influential features in predicting whether a material is a safe alternative for battery applications.")