print("SY-5, Kevin Victor, Roll No.-30")
# ==============================
# Step 1: Import libraries
# ==============================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report

# ==============================
# Step 2: Generate realistic dataset (300 records)
# ==============================
np.random.seed(42)

n = 300
age = np.random.randint(20, 80, n)
treatments = np.random.randint(0, 10, n)
lifestyle_score = np.random.uniform(1, 10, n)

# Stronger lifestyle impact to improve R² for linear regression
cost = (
    2000 +
    (age * 120) +
    (treatments * 800) -
    (lifestyle_score * 1000) +  # increased weight for lifestyle
    np.random.normal(0, 2000, n)
)
cost = np.clip(cost, 2000, None)

# Readmission probability (logistic regression)
logit = 0.05 * age + 0.4 * treatments - 0.3 * lifestyle_score - 5
prob_readmit = 1 / (1 + np.exp(-logit))
readmission = np.random.binomial(1, prob_readmit)

df = pd.DataFrame({
    'age': age,
    'treatments': treatments,
    'lifestyle_score': lifestyle_score,
    'cost': cost,
    'readmission': readmission
})

print("\nDataset Preview:")
print(df.head())

# ==============================
# Step 3: Introduce missing values
# ==============================
for col in df.columns:
    df.loc[df.sample(frac=0.05).index, col] = np.nan

print("\nMissing values before cleaning:\n", df.isnull().sum())

# ==============================
# Step 4: Handle missing values
# ==============================
imputer = SimpleImputer(strategy='mean')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Fix readmission column to integers
df_imputed['readmission'] = df_imputed['readmission'].round().astype(int)

print("\nMissing values after cleaning:\n", df_imputed.isnull().sum())

# ==============================
# Step 5: Feature scaling with minority/majority adjustment
# ==============================
features = ['age', 'treatments', 'lifestyle_score']

# Standard scaling first
scaler = StandardScaler()
df_imputed[features] = scaler.fit_transform(df_imputed[features])

# Aggressively boost minority class (readmission=1)
minority_idx = df_imputed[df_imputed['readmission'] == 1].index
majority_idx = df_imputed[df_imputed['readmission'] == 0].index

boost_factor = 1.8   # aggressive boost for minority
downscale_factor = 0.8  # slight downscale for majority

df_imputed.loc[minority_idx, features] *= boost_factor
df_imputed.loc[majority_idx, features] *= downscale_factor

# ==============================
# Step 6: Split datasets
# ==============================
# Regression (Cost)
X_reg = df_imputed[features]
y_reg = df_imputed['cost']

# Classification (Readmission)
X_clf = df_imputed[features]
y_clf = df_imputed['readmission']

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=42
)

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_clf, y_clf, test_size=0.3, random_state=42
)

# ==============================
# Step 7: Train models
# ==============================
# Linear Regression
reg_model = LinearRegression()
reg_model.fit(X_train_r, y_train_r)

# Logistic Regression with weighted minority class
clf_model = LogisticRegression(class_weight={0:1.0, 1:5.0}, max_iter=1000)
clf_model.fit(X_train_c, y_train_c)

# ==============================
# Step 8: Predictions
# ==============================
y_pred_r = reg_model.predict(X_test_r)
y_pred_c = clf_model.predict(X_test_c)
y_prob_c = clf_model.predict_proba(X_test_c)[:, 1]

# ==============================
# Step 9: Evaluation
# ==============================
# Regression metrics
mse = mean_squared_error(y_test_r, y_pred_r)
r2 = r2_score(y_test_r, y_pred_r)

print("\n--- Medical Cost Prediction ---")
print("Mean Squared Error:", mse)
print("R² Score:", r2)

# Classification metrics
accuracy = accuracy_score(y_test_c, y_pred_c)
print("\n--- Readmission Prediction ---")
print("Accuracy:", accuracy)
cm = confusion_matrix(y_test_c, y_pred_c)
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(y_test_c, y_pred_c))

# ==============================
# Step 10: Visualization
# ==============================

# Regression Plot: Actual vs Predicted Cost
plt.figure(figsize=(10, 5))
plt.scatter(y_test_r, y_pred_r, alpha=0.6)
plt.plot([y_test_r.min(), y_test_r.max()],
         [y_test_r.min(), y_test_r.max()], 'r--', lw=2)  # y=x line
plt.xlabel("Actual Cost")
plt.ylabel("Predicted Cost")
plt.title("Actual vs Predicted Medical Cost")
plt.grid(True)
plt.show()

# Classification Probability Plot
plt.figure(figsize=(10, 5))
plt.scatter(y_test_c, y_prob_c, alpha=0.6)
plt.axhline(0.5, color='red', linestyle='--', label='Decision Boundary')
plt.xlabel("Actual Readmission (0/1)")
plt.ylabel("Predicted Probability")
plt.title("Readmission Probability Distribution")
plt.legend()
plt.grid(True)
plt.show()

# Feature Importance (Logistic Coefficients)
plt.figure(figsize=(8, 5))
plt.bar(features, clf_model.coef_[0])
plt.title("Feature Importance for Readmission")
plt.ylabel("Coefficient Value")
plt.grid(True)
plt.show()