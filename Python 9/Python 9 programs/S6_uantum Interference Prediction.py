# ============================================================
# DEEP QUANTUM ANALYSIS (DQA) MODEL
# Quantum Double-Slit Experiment Simulation
# Logistic Regression + Hyperparameter Tuning
# ============================================================
print("SY-5, Kevin Victor, Roll No.-30")
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# STEP 1: SIMULATED QUANTUM DATASET
# -----------------------------

np.random.seed(42)
n_samples = 1200

data = pd.DataFrame({
    "wavelength": np.random.uniform(400, 700, n_samples),  # nm
    "slit_width": np.random.uniform(0.1, 1.0, n_samples),
    "slit_distance": np.random.uniform(0.5, 5.0, n_samples),
    "detector_active": np.random.randint(0, 2, n_samples),
    "phase_shift": np.random.uniform(0, 2*np.pi, n_samples),
    "coherence_level": np.random.uniform(0.3, 1.0, n_samples),
    "noise_level": np.random.uniform(0, 0.5, n_samples),
    "particle_intensity": np.random.uniform(0.1, 1.0, n_samples)
})

# -----------------------------
# TARGET VARIABLE
# -----------------------------
# Interference occurs if:
# - No detector OR high coherence
# - Low noise

data["interference"] = (
    ((data["detector_active"] == 0) | (data["coherence_level"] > 0.7)) &
    (data["noise_level"] < 0.3)
).astype(int)

# -----------------------------
# STEP 2: FEATURE ENGINEERING (DQA)
# -----------------------------

# Capture complex quantum interactions
poly = PolynomialFeatures(degree=2, include_bias=False)
features = poly.fit_transform(data.drop("interference", axis=1))

feature_names = poly.get_feature_names_out(data.drop("interference", axis=1).columns)
X = pd.DataFrame(features, columns=feature_names)
y = data["interference"]

# -----------------------------
# STEP 3: PREPROCESSING
# -----------------------------

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# -----------------------------
# STEP 4: BASE MODEL
# -----------------------------

base_model = LogisticRegression(max_iter=100)

base_model.fit(X_train, y_train)
y_pred_base = base_model.predict(X_test)

base_accuracy = accuracy_score(y_test, y_pred_base)

print("\n===== BASE MODEL PERFORMANCE =====\n")
print("Accuracy (Before Tuning):", base_accuracy)
print(classification_report(y_test, y_pred_base))

# -----------------------------
# STEP 5: GRID SEARCH TUNING
# -----------------------------

param_grid = {
    "C": [0.01, 0.1, 1, 10],
    "solver": ["lbfgs", "liblinear"],
    "penalty": ["l2"],
    "max_iter": [100, 300, 500]
}

grid = GridSearchCV(
    LogisticRegression(),
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid.fit(X_train, y_train)

print("\n===== BEST PARAMETERS =====\n")
print(grid.best_params_)

# -----------------------------
# STEP 6: TUNED MODEL
# -----------------------------

best_model = grid.best_estimator_

y_pred_tuned = best_model.predict(X_test)

tuned_accuracy = accuracy_score(y_test, y_pred_tuned)

print("\n===== TUNED MODEL PERFORMANCE =====\n")
print("Accuracy (After Tuning):", tuned_accuracy)
print(classification_report(y_test, y_pred_tuned))

# -----------------------------
# STEP 7: ACCURACY COMPARISON
# -----------------------------

print("\n===== ACCURACY COMPARISON =====\n")
print(f"Before Tuning: {base_accuracy:.4f}")
print(f"After Tuning:  {tuned_accuracy:.4f}")

if tuned_accuracy > base_accuracy:
    print("Model improved after tuning.")
else:
    print("No significant improvement. Consider better features or models.")

# -----------------------------
# STEP 8: BACKEND INSIGHTS
# -----------------------------

print("\n===== BACKEND INSIGHTS =====\n")

print("1. Why initial model underperforms:")
print("- Quantum systems are highly non-linear")
print("- Logistic regression struggles with complex interactions")

print("\n2. Role of Feature Engineering (DQA):")
print("- Polynomial features capture interference interactions")
print("- Helps approximate quantum relationships")

print("\n3. Why tuning improves performance:")
print("- Better regularization reduces overfitting")
print("- Solver selection impacts convergence")

print("\n4. Further Improvements:")
print("- Use non-linear models (Random Forest, XGBoost)")
print("- Use neural networks for quantum-like patterns")
print("- Train separate models for different regimes (coherent vs decoherent)")

print("\n5. Production Insight:")
print("- Monitor accuracy drift due to environmental noise")
print("- Retrain model periodically with updated experiment data")