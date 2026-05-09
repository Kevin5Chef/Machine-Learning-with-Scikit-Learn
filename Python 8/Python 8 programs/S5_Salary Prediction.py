import numpy as np
import pandas as pd
print("SY-5, Kevin Victor, Roll No.-30")
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score   # <-- added

# -------------------------------
# STEP 1: Create Enhanced Dataset
# -------------------------------

np.random.seed(42)
n = 300

data = pd.DataFrame({
    "years_experience": np.random.uniform(0, 15, n),
    "skill_level": np.random.uniform(1, 10, n),
    "skill_index": np.random.randint(1, 15, n),

    "productivity_score": np.random.uniform(40, 100, n),
    "teamwork_score": np.random.uniform(1, 10, n)
})

# -------------------------------
# STEP 2: Feature Engineering
# -------------------------------

data["efficiency"] = data["productivity_score"] / (data["years_experience"] + 1)
data["skill_synergy"] = data["skill_level"] * data["skill_index"]

# -------------------------------
# STEP 3: Create Target (Salary)
# -------------------------------

data["salary"] = (
    30000 +
    data["years_experience"] * 3500 +
    data["skill_level"] * 5000 +
    data["skill_index"] * 2000 +
    data["productivity_score"] * 300 +
    data["teamwork_score"] * 2500 +
    data["efficiency"] * 1500 +
    data["skill_synergy"] * 100 +
    np.random.normal(0, 5000, n)
)

# -------------------------------
# STEP 4: Introduce Missing Values
# -------------------------------

for col in data.columns:
    if col != "salary":
        data.loc[data.sample(frac=0.05).index, col] = np.nan

pd.set_option('display.max_columns', None)

print("\nInitial Dataset:\n")
print(data.head())

# -------------------------------
# STEP 5: Preprocessing
# -------------------------------

X = data.drop("salary", axis=1)
y = data["salary"]

imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# -------------------------------
# STEP 6: Train-Test Split
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# -------------------------------
# STEP 7: Train Model
# -------------------------------

model = LinearRegression()
model.fit(X_train, y_train)

# -------------------------------
# STEP 8: Predictions
# -------------------------------

predictions = model.predict(X_test)

# -------------------------------
# STEP 9: Evaluation
# -------------------------------

mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)   # <-- added

print("\nMean Squared Error:", mse)
print("R² Score:", r2)               # <-- added (just beneath MSE)

# -------------------------------
# STEP 10: Results Comparison
# -------------------------------

results = pd.DataFrame({
    "Actual Salary": y_test.values,
    "Predicted Salary": predictions
})

print("\nSample Predictions:\n")
print(results.head(10))

# -------------------------------
# STEP 11: Model Interpretation
# -------------------------------

print("\nModel Coefficients (Feature Importance):")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef:.2f}")

print("\nIntercept:", model.intercept_)