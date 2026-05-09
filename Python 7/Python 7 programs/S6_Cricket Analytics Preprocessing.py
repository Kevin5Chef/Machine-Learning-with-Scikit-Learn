import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
print("SY-5, Kevin Victor, Roll No.-30")
# -------------------------------
# DISPLAY SETTINGS
# -------------------------------
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1200)

print("Cricket Sports Analytics Dataset - StandardScaler Application\n")

# -------------------------------
# STEP 1: CREATE REALISTIC DATASET
# -------------------------------
np.random.seed(42)
n = 100

data = pd.DataFrame({

    # Core performance metrics
    "Runs": np.random.randint(500, 5000, n).astype(int),  # Ensured integer type
    "Strike_Rate": np.random.normal(130, 20, n),
    "Batting_Avg_Global": np.random.normal(38, 8, n),
    "Batting_Avg_Recent": np.random.normal(42, 10, n),

    # Match condition splits
    "Home_Avg": np.random.normal(40, 8, n),
    "Away_Avg": np.random.normal(35, 9, n),
    "Home_SR": np.random.normal(135, 15, n),
    "Away_SR": np.random.normal(125, 20, n),

    # Performance vs bowling type
    "SR_vs_Pace": np.random.normal(135, 15, n),
    "SR_vs_Spin": np.random.normal(125, 15, n),

    # Shot distribution (% contribution)
    "Leg_Side_%": np.random.uniform(30, 60, n),
    "Off_Side_%": np.random.uniform(30, 60, n),
    "V_Region_%": np.random.uniform(10, 30, n),
    "Square_%": np.random.uniform(10, 40, n),

    # Consistency metrics
    "Boundary_%": np.random.uniform(40, 70, n),
    "Dot_Ball_%": np.random.uniform(20, 50, n)
})

# Categorical features
batsman_types = ["Aggressive", "Defensive", "Balanced"]
dismissal_types = ["Bowled", "Caught", "LBW", "Run Out"]
signature_shots = ["Cover Drive", "Pull Shot", "Cut Shot", "Flick"]
arch_rivals = ["Starc", "Bumrah", "Rashid Khan", "Rabada"]

data["Batsman_Type"] = np.random.choice(batsman_types, n)
data["Dismissal_Type"] = np.random.choice(dismissal_types, n)
data["Signature_Shot"] = np.random.choice(signature_shots, n)
data["Arch_Rival_Bowler"] = np.random.choice(arch_rivals, n)

print("STEP 1: Original Dataset (First 5 rows)\n")
print(data.head())


# -------------------------------
# STEP 2: ENCODE CATEGORICAL DATA
# -------------------------------
label_cols = ["Batsman_Type", "Dismissal_Type", "Signature_Shot", "Arch_Rival_Bowler"]

encoders = {}
for col in label_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    encoders[col] = le

print("\nSTEP 2: After Encoding Categorical Features\n")
print(data.head())


# -------------------------------
# STEP 3: APPLY STANDARD SCALER
# -------------------------------
scaler = StandardScaler()

scaled_data = scaler.fit_transform(data)

df_scaled = pd.DataFrame(scaled_data, columns=data.columns)

print("\nSTEP 3: Scaled Dataset (First 5 rows)\n")
print(df_scaled.head())


# -------------------------------
# STEP 4: COMPARISON
# -------------------------------
print("\nSummary BEFORE Scaling:\n")
print(data.describe())

print("\nSummary AFTER Scaling:\n")
print(df_scaled.describe())


# -------------------------------
# STEP 5: EXPLANATION
# -------------------------------
print("\nExplanation:")

print("""
1. Features like Runs (~500–5000) and Strike Rate (~100–200) are on very different scales.

2. Without scaling:
   - 'Runs' dominates ML models due to larger magnitude.
   - Smaller-scale features like strike rate contribute less.

3. StandardScaler transforms each feature using:
   z = (x - mean) / standard deviation

4. After scaling:
   - All features have mean ≈ 0
   - Standard deviation ≈ 1
   - Features contribute equally to model learning

5. Impact in cricket analytics:
   - Fair comparison between consistency (average) and aggression (strike rate)
   - Better clustering of batsmen types
   - Improved model performance in predictions

6. This is especially important for:
   - K-Means clustering (player grouping)
   - KNN (player similarity)
   - Regression models (performance prediction)
""")