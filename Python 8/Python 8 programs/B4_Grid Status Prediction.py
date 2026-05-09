print("SY-5, Kevin Victor, Roll No.-30")

# ==============================
# Step 1: Import libraries
# ==============================
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

pd.set_option('display.max_columns', None)

# ==============================
# Step 2: Generate diversified dataset
# ==============================
np.random.seed(42)

n = 300

temperature = np.random.uniform(15, 45, n)
humidity = np.random.uniform(20, 90, n)
wind_speed = np.random.uniform(0, 25, n)

season = np.random.choice([0, 1, 2], n)

voltage = np.random.uniform(210, 240, n)
grid_frequency = np.random.uniform(49.5, 50.5, n)

solar_output = np.random.uniform(50, 500, n) * (1 + 0.3 * (season == 1))
wind_output = np.random.uniform(20, 300, n) * (1 + 0.4 * (season == 2))

industrial_load = np.random.uniform(200, 1000, n)
residential_load = np.random.uniform(300, 1500, n) + (temperature * 10)

demand = (
        400 +
        (temperature * 40) +
        (residential_load * 0.5) +
        (industrial_load * 0.7) -
        (solar_output * 0.25) -
        (wind_output * 0.2) +
        np.random.normal(0, 150, n)
)

df = pd.DataFrame({
    'temperature': temperature,
    'humidity': humidity,
    'wind_speed': wind_speed,
    'season': season,
    'voltage': voltage,
    'grid_frequency': grid_frequency,
    'solar_output': solar_output,
    'wind_output': wind_output,
    'industrial_load': industrial_load,
    'residential_load': residential_load,
    'demand': demand
})

print("\nDataset Preview:\n", df.head())

# ==============================
# Step 3: Missing values
# ==============================
for col in df.columns:
    df.loc[df.sample(frac=0.05).index, col] = np.nan

print("\nMissing values before cleaning:\n", df.isnull().sum())

# ==============================
# Step 4: Imputation
# ==============================
imputer = SimpleImputer(strategy='mean')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

print("\nMissing values after cleaning:\n", df_imputed.isnull().sum())

# ==============================
# Step 5: Feature Engineering
# ==============================
df_imputed['load_ratio'] = (
                                   df_imputed['residential_load'] + df_imputed['industrial_load']
                           ) / (df_imputed['solar_output'] + df_imputed['wind_output'] + 1)

df_imputed['renewable_ratio'] = (
                                        df_imputed['solar_output'] + df_imputed['wind_output']
                                ) / (df_imputed['demand'] + 1)

df_imputed['stability_index'] = (
        abs(df_imputed['voltage'] - 230) +
        abs(df_imputed['grid_frequency'] - 50)
)

df_imputed['peak_stress'] = df_imputed['temperature'] * df_imputed['demand']

# ==============================
# Step 6: Target
# ==============================
df_imputed['grid_status'] = np.where(
    (df_imputed['load_ratio'] > 4.5) |
    (df_imputed['stability_index'] > 8) |
    (df_imputed['peak_stress'] > np.percentile(df_imputed['peak_stress'], 75)),
    1, 0
)

print("\nGrid Status Distribution:\n", df_imputed['grid_status'].value_counts())

# ==============================
# Step 7: Scaling
# ==============================
features = [
    'temperature', 'humidity', 'wind_speed', 'season',
    'voltage', 'grid_frequency',
    'solar_output', 'wind_output',
    'industrial_load', 'residential_load',
    'load_ratio', 'renewable_ratio',
    'stability_index', 'peak_stress'
]

scaler = StandardScaler()
df_imputed[features] = scaler.fit_transform(df_imputed[features])

# ==============================
# Step 8: Split
# ==============================
X = df_imputed[features]
y = df_imputed['grid_status']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ==============================
# Step 9: Train model
# ==============================
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ==============================
# Step 10: Predictions
# ==============================
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# ==============================
# Step 11: Evaluation
# ==============================
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ==============================
# Step 12: Transparent Insights (UPDATED)
# ==============================
print("\nSample Predictions and System Insights (with dominant factors):\n")

coefficients = model.coef_[0]

for i in range(10):
    status = "CRITICAL" if y_pred[i] == 1 else "STABLE"

    print(f"Case {i + 1}:")
    print(f"Probability of Critical State: {y_prob[i]:.2f}")
    print(f"Predicted Grid Status: {status}")

    # Contribution calculation
    sample = X_test.iloc[i].values
    contributions = sample * coefficients

    feature_contrib = list(zip(features, contributions))
    feature_contrib.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Dominant Factors:")
    for f, c in feature_contrib[:3]:
        impact = "increasing risk" if c > 0 else "stabilizing"
        print(f"  {f}: {c:.3f} ({impact})")

    if status == "CRITICAL":
        print("Insight: High stress detected. Consider load redistribution or boosting renewables.")
    else:
        print("Insight: Grid stable with balanced load and conditions.")

    print("-" * 50)

# ==============================
# Step 13: Feature Importance
# ==============================
print("\nFeature Importance (Logistic Coefficients):\n")

for feature, coef in zip(features, model.coef_[0]):
    print(f"{feature}: {coef:.4f}")