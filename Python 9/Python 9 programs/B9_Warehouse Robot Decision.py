# ============================================================
# WAREHOUSE ROBOT DECISION SYSTEM USING LOGISTIC REGRESSION
# WITH GRIDSEARCHCV HYPERPARAMETER TUNING
# ============================================================
print("SY-5, Kevin Victor, Roll No.-30")
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# -----------------------------
# STEP 1: SIMULATED DATASET
# -----------------------------

np.random.seed(42)
n_samples = 1000

data = pd.DataFrame({
    "stock_level": np.random.randint(0, 500, n_samples),
    "daily_demand": np.random.randint(1, 50, n_samples),
    "lead_time_days": np.random.randint(1, 15, n_samples),
    "order_fulfillment_rate": np.random.uniform(0.5, 1.0, n_samples),
    "delay_probability": np.random.uniform(0, 1, n_samples),
    "traffic_congestion": np.random.randint(0, 10, n_samples),
    "equipment_available": np.random.randint(0, 2, n_samples),
    "handling_cost": np.random.uniform(10, 200, n_samples),
    "unit_price": np.random.uniform(100, 5000, n_samples),
    "demand_variability": np.random.uniform(0.1, 1.0, n_samples)
})

# -----------------------------
# TARGET VARIABLE (Robot Decision)
# -----------------------------
# 1 = High priority task
# 0 = Low priority

data["priority"] = (
    (data["stock_level"] < 100) &
    (data["daily_demand"] > 25) |
    (data["delay_probability"] > 0.7) |
    (data["order_fulfillment_rate"] < 0.7)
).astype(int)

# -----------------------------
# STEP 2: PREPROCESSING
# -----------------------------

X = data.drop("priority", axis=1)
y = data["priority"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# -----------------------------
# STEP 3: BASE MODEL
# -----------------------------

model = LogisticRegression()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\n===== BASE MODEL PERFORMANCE =====\n")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# -----------------------------
# STEP 4: GRID SEARCH CV
# -----------------------------

param_grid = {
    "C": [0.01, 0.1, 1, 10],
    "solver": ["liblinear", "lbfgs"],
    "penalty": ["l2"],
    "max_iter": [100, 200, 500]
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
# STEP 5: BEST MODEL
# -----------------------------

best_model = grid.best_estimator_

y_pred_best = best_model.predict(X_test)

print("\n===== OPTIMIZED MODEL PERFORMANCE =====\n")
print("Accuracy:", accuracy_score(y_test, y_pred_best))
print(classification_report(y_test, y_pred_best))

# -----------------------------
# STEP 6: FEATURE IMPORTANCE
# -----------------------------

coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Importance": best_model.coef_[0]
}).sort_values(by="Importance", ascending=False)

print("\n===== FEATURE IMPORTANCE =====\n")
print(coefficients)

# -----------------------------
# STEP 7: BACKEND INSIGHTS
# -----------------------------

print("\n===== BACKEND INSIGHTS =====\n")

print("1. Why GridSearchCV matters\n")

print("Without tuning:")
print("- Model may underfit or overfit")
print("- Poor decision-making leading to revenue loss\n")

print("With tuning:")
print("- Balanced bias-variance tradeoff")
print("- Better real-world decisions\n")

print("2. Feature importance insights\n")

print("From coefficients:")
print("- High positive values increase priority")
print("- Negative values reduce priority\n")

print("Example:")
print("- High demand leads to prioritization")
print("- High stock leads to deprioritization\n")

print("3. Performance improvement strategy\n")

print("If accuracy is low:")
print("- Add more features such as:")
print("  * Time of day")
print("  * SKU category")
print("  * Customer priority")
print("  * Seasonal demand\n")

print("Note: Real-world systems typically use very large and complex feature sets.")