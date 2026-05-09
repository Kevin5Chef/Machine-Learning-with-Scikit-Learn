import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
print("SY-5, Kevin Victor, Roll No.-30")
# -------------------------------
# DISPLAY SETTINGS
# -------------------------------
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print("Student Height-Weight Dataset Normalization\n")

# -------------------------------
# STEP 1: CREATE REALISTIC DATASET
# -------------------------------
# Heights: 5'4" (162 cm) to 6'4" (193 cm)
# Weights: 55 kg to 95 kg

np.random.seed(42)
n = 100

heights = np.random.normal(loc=172, scale=7, size=n)   # avg ~172 cm
weights = np.random.normal(loc=70, scale=10, size=n)   # avg ~70 kg

# Clip values to realistic bounds
heights = np.clip(heights, 162, 193)
weights = np.clip(weights, 55, 95)

# Create DataFrame
df = pd.DataFrame({
    "Height (cm)": heights,
    "Weight (kg)": weights
})

print("STEP 1: Original Dataset (First 5 rows)\n")
print(df.head())

print("\nSummary Statistics BEFORE Scaling:\n")
print(df.describe())


# -------------------------------
# STEP 2: APPLY STANDARD SCALER
# -------------------------------
# StandardScaler transforms data to:
# mean = 0, standard deviation = 1

scaler = StandardScaler()

scaled_data = scaler.fit_transform(df)

# Convert back to DataFrame for readability
df_scaled = pd.DataFrame(scaled_data, columns=df.columns)

print("\nSTEP 2: Scaled Dataset (First 5 rows)\n")
print(df_scaled.head())

print("\nSummary Statistics AFTER Scaling:\n")
print(df_scaled.describe())


# -------------------------------
# STEP 3: EXPLANATION
# -------------------------------
print("\nExplanation:")

print("""
1. Original data has different scales:
   - Height is in centimeters (~160–190)
   - Weight is in kilograms (~55–95)

2. Without scaling:
   - Height may dominate ML models due to larger magnitude

3. StandardScaler performs:
   z = (x - mean) / standard deviation

4. After scaling:
   - Both features have mean ≈ 0
   - Standard deviation ≈ 1
   - Ensures equal contribution in ML models

5. This is essential for:
   - Distance-based algorithms (KNN, K-Means)
   - Gradient-based models
""")