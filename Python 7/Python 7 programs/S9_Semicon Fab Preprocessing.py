import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
print("SY-5, Kevin Victor, Roll No.-30")
# -------------------------------
# DISPLAY SETTINGS
# -------------------------------
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1400)

print("Semiconductor Fab Robot Dataset - Preprocessing Pipeline\n")

# -------------------------------
# STEP 1: CREATE REALISTIC DATASET
# -------------------------------
# Simulates wafer fabrication environment with nanometer precision

np.random.seed(42)
n = 200

data = pd.DataFrame({

    # Wafer quality parameters
    "Wafer_Size_mm": np.random.normal(300, 5, n),
    "Wafer_Thickness_nm": np.random.normal(775000, 5000, n),
    "Layer_Count": np.random.randint(50, 120, n),

    # Defects
    "Minor_Defects": np.random.randint(0, 10, n),
    "Major_Defects": np.random.randint(0, 3, n),

    # Lithography precision (nm scale)
    "Lithography_Deviation_nm": np.random.normal(2, 0.5, n),
    "Etching_Precision_nm": np.random.normal(1.5, 0.3, n),

    # Environmental conditions (critical)
    "Cleanroom_Particles_per_m3": np.random.normal(50, 10, n),
    "Temperature_C": np.random.normal(22, 1, n),
    "Pressure_Pa": np.random.normal(101325, 500, n),
    "Humidity_%": np.random.normal(45, 5, n),

    # Resource utilization
    "Power_Consumption_kWh": np.random.normal(500, 50, n),
    "Water_Usage_L": np.random.normal(2000, 200, n),
    "Airflow_m3_per_min": np.random.normal(300, 30, n),

    # Process timings
    "UV_Activation_Time_min": np.random.normal(20, 5, n),
    "Cleaning_Time_min": np.random.normal(15, 3, n),

    # Logistics / tools
    "Machines_Used": np.random.randint(5, 15, n),
    "Operator_Interventions": np.random.randint(0, 5, n)
})

# Ensure no negative values
data = data.clip(lower=0)

print("STEP 1: Initial Dataset (First 5 rows)\n")
print(data.head())


# -------------------------------
# STEP 2: FEATURE ENGINEERING
# -------------------------------
# Create derived features for performance and quality

data["Total_Defects"] = data["Minor_Defects"] + data["Major_Defects"]
data["Process_Efficiency"] = data["Layer_Count"] / (data["UV_Activation_Time_min"] + data["Cleaning_Time_min"])
data["Resource_Load"] = data["Power_Consumption_kWh"] + data["Water_Usage_L"]
data["Environmental_Stability"] = (
    data["Temperature_C"] * data["Humidity_%"] / (data["Pressure_Pa"] + 1)
)
data["Precision_Index"] = (
    data["Lithography_Deviation_nm"] + data["Etching_Precision_nm"]
)

print("\nSTEP 2: After Feature Engineering\n")
print(data.head())


# -------------------------------
# STEP 3: INTRODUCE MISSING VALUES
# -------------------------------
data_missing = data.copy()

for col in data_missing.columns:
    num_missing = np.random.randint(10, 20)
    indices = np.random.choice(data_missing.index, num_missing, replace=False)
    data_missing.loc[indices, col] = np.nan

print("\nSTEP 3: Missing Values BEFORE Imputation\n")
print(data_missing.isnull().sum())


# -------------------------------
# STEP 4: IMPUTE MISSING VALUES (FIRST)
# -------------------------------
# Correct first step: handle missing values

imputer = SimpleImputer(strategy='mean')

data_imputed = pd.DataFrame(
    imputer.fit_transform(data_missing),
    columns=data_missing.columns
)

print("\nSTEP 4: Missing Values AFTER Imputation\n")
print(data_imputed.isnull().sum())


# -------------------------------
# STEP 5: FEATURE SCALING (SECOND)
# -------------------------------
# Now apply scaling AFTER imputation

scaler = StandardScaler()

data_scaled = scaler.fit_transform(data_imputed)

df_scaled = pd.DataFrame(data_scaled, columns=data_imputed.columns)

print("\nSTEP 5: Scaled Dataset (First 5 rows)\n")
print(df_scaled.head())


# -------------------------------
# STEP 6: EXPLANATION
# -------------------------------
print("\nExplanation:")

print("""
Correct Preprocessing Order:
1. First Impute Missing Values
2. Then Apply Scaling

Why this order is important:

1. StandardScaler cannot handle missing values:
   - If NaN values are present, scaling will fail or produce invalid results.

2. Mean/standard deviation calculation:
   - Scaling depends on mean and standard deviation.
   - Missing values distort these calculations.

3. Data consistency:
   - Imputation ensures a complete dataset before transformation.

4. If scaling is done BEFORE imputation:
   - NaN values remain unchanged
   - Statistical properties become incorrect
   - Model performance degrades

5. Real-world implication (semiconductor fab):
   - Incorrect preprocessing may misinterpret nanometer-level deviations
   - This can lead to faulty wafers and massive production loss

Conclusion:
Always handle missing data first, then normalize/scale features.
""")