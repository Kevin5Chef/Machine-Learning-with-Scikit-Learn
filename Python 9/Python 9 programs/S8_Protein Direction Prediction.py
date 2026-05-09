# ============================================================
# PROTEIN STRUCTURE (2D) BRANCH DIRECTION PREDICTION
# Logistic Regression + GridSearchCV
# ============================================================
print("SY-5, Kevin Victor, Roll No.-30")
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# STEP 1: SIMULATED DATASET
# -----------------------------

np.random.seed(42)
n_samples = 1500

data = pd.DataFrame({
    "density_intensity": np.random.uniform(0.2, 1.0, n_samples),
    "gradient_x": np.random.uniform(-1, 1, n_samples),
    "gradient_y": np.random.uniform(-1, 1, n_samples),
    "local_curvature": np.random.uniform(0, 2, n_samples),
    "signal_to_noise": np.random.uniform(0.1, 1.0, n_samples),
    "edge_strength": np.random.uniform(0, 1, n_samples),
    "neighbor_density": np.random.uniform(0.2, 1.0, n_samples),
    "angular_variation": np.random.uniform(0, 360, n_samples)
})

# -----------------------------
# TARGET VARIABLE (8 DIRECTIONS)
# -----------------------------
# 0 to 7 representing 8 cardinal directions

angles = (np.arctan2(data["gradient_y"], data["gradient_x"]) * 180 / np.pi) % 360
data["direction"] = (angles // 45).astype(int)

# -----------------------------
# STEP 2: ADVANCED FEATURE ENGINEERING (DQA++)
# -----------------------------

# Original features
base_features = data.drop("direction", axis=1)

# ---------------------------------
# Fourier Transform Features
# ---------------------------------
# Capture periodic patterns in protein structures

fft_features = np.fft.fft(base_features, axis=0)
fft_real = np.real(fft_features)
fft_imag = np.imag(fft_features)

fft_real_df = pd.DataFrame(fft_real, columns=[f"{col}_fft_real" for col in base_features.columns])
fft_imag_df = pd.DataFrame(fft_imag, columns=[f"{col}_fft_imag" for col in base_features.columns])

# ---------------------------------
# Laplacian Features (2nd derivative)
# ---------------------------------
# Capture curvature / structural changes

laplacian_df = base_features.diff().diff().fillna(0)
laplacian_df.columns = [f"{col}_laplacian" for col in base_features.columns]

# ---------------------------------
# Combine all features
# ---------------------------------

enhanced_features = pd.concat(
    [base_features, fft_real_df, fft_imag_df, laplacian_df],
    axis=1
)

# ---------------------------------
# Polynomial expansion (retain original intent)
# ---------------------------------

poly = PolynomialFeatures(degree=2, include_bias=False)
X = poly.fit_transform(enhanced_features)

y = data["direction"]

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

base_model = LogisticRegression(max_iter=100, multi_class='multinomial')

base_model.fit(X_train, y_train)
y_pred_base = base_model.predict(X_test)

base_acc = accuracy_score(y_test, y_pred_base)

print("\n===== BASE MODEL PERFORMANCE =====\n")
print("Accuracy (Before Tuning):", base_acc)
print(classification_report(y_test, y_pred_base))

# -----------------------------
# STEP 5: GRID SEARCH CV
# -----------------------------

param_grid = {
    "C": [0.01, 0.1, 1, 10],
    "solver": ["lbfgs", "saga"],
    "max_iter": [100, 300, 500]
}

grid = GridSearchCV(
    LogisticRegression(multi_class='multinomial'),
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
tuned_acc = accuracy_score(y_test, y_pred_tuned)

print("\n===== TUNED MODEL PERFORMANCE =====\n")
print("Accuracy (After Tuning):", tuned_acc)
print(classification_report(y_test, y_pred_tuned))

# -----------------------------
# STEP 7: ACCURACY COMPARISON
# -----------------------------

print("\n===== ACCURACY COMPARISON =====\n")
print(f"Before Tuning: {base_acc:.4f}")
print(f"After Tuning:  {tuned_acc:.4f}")

# -----------------------------
# STEP 8: WHY MODEL IMPROVED
# -----------------------------

print("\n===== WHY MODEL IMPROVED =====\n")

print("1. Regularization (C parameter):")
print("- Controls overfitting vs underfitting")
print("- Optimal C improves generalization on unseen protein structures")

print("\n2. Solver Optimization:")
print("- Different solvers handle multi-class optimization differently")
print("- 'lbfgs' and 'saga' improve convergence for high-dimensional data")

print("\n3. Increased Iterations (max_iter):")
print("- Ensures convergence in complex feature space")
print("- Important due to polynomial feature expansion")

print("\n4. Feature Engineering Impact:")
print("- Polynomial features capture non-linear protein geometry")
print("- Helps approximate structural relationships from EM data")

print("\n5. Noise Handling:")
print("- Proper scaling + tuning improves robustness to noisy EM signals")

print("\nBackend Recommendations:")

print("- Add spatial neighborhood features (graph-based modeling)")
print("- Use CNNs for direct EM image input instead of tabular features")
print("- Try non-linear models (Random Forest, XGBoost)")
print("- Consider sequence + structure hybrid models (like AlphaFold-style systems)")