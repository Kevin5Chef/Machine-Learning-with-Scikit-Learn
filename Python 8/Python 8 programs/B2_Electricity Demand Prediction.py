print("SY-5, Kevin Victor, Roll No.-30")

# ==============================
# Step 1: Import libraries
# ==============================
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression

# Display all columns
pd.set_option('display.max_columns', None)

# ==============================
# Step 2: Generate realistic dataset
# ==============================
np.random.seed(42)

n = 300

# Environmental conditions (Indian summer peak)
temperature = np.random.uniform(30, 48, n)
temperature = np.clip(temperature, None, 45)   # Cap at 45°C

humidity = np.random.uniform(20, 80, n)
wind_speed = np.random.uniform(0, 20, n)

# Grid parameters
voltage = np.random.uniform(210, 240, n)
grid_frequency = np.random.uniform(49.5, 50.5, n)

# Renewable inputs
solar_output = np.random.uniform(50, 500, n)
wind_output = np.random.uniform(20, 300, n)

# Demand-related engineered features
industrial_load = np.random.uniform(200, 1000, n)
residential_load = np.random.uniform(300, 1500, n)

# Target: Total electricity demand (MW)
demand = (
    500 +
    (temperature * 50) +
    (residential_load * 0.6) +
    (industrial_load * 0.8) -
    (solar_output * 0.3) -
    (wind_output * 0.2) +
    np.random.normal(0, 100, n)
)

df = pd.DataFrame({
    'temperature': temperature,
    'humidity': humidity,
    'wind_speed': wind_speed,
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
# Step 3: Introduce missing values
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
# Step 5: Feature scaling
# ==============================
features = df_imputed.columns[:-1]

scaler = StandardScaler()
df_imputed[features] = scaler.fit_transform(df_imputed[features])

# ==============================
# Step 6: Train-test split
# ==============================
X = df_imputed[features]
y = df_imputed['demand']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ==============================
# Step 7: Train Linear Regression model
# ==============================
model = LinearRegression()
model.fit(X_train, y_train)

# ==============================
# Step 8: Display slope & intercept
# ==============================
print("\nIntercept:", model.intercept_)

print("\nSlope (coefficients):")
for feature, coef in zip(features, model.coef_):
    print(f"{feature}: {coef:.4f}")

# ==============================
# Step 9: Explanation
# ==============================
print("\nExplanation (Slope & Intercept Meaning)")

print("\nIntercept:")
print("This is the baseline electricity demand when all features are zero (after scaling). It represents the base load of the city grid.")

print("\nSlope (coefficients):")
print("Each value shows how much demand changes when that feature increases by 1 unit.")
print("Positive slope (e.g., temperature): higher temperature → higher demand (AC usage)")
print("Negative slope (e.g., solar_output): more renewable energy → reduces grid demand")

print("\nIn a smart grid, these values help engineers understand system sensitivity and optimize load balancing in real time.")