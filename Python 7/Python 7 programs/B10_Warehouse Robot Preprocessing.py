import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
print("SY-5, Kevin Victor, Roll No.-30")
# -------------------------------
# DISPLAY SETTINGS (show all columns)
# -------------------------------
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print("Warehouse Robot Dataset - Complete Preprocessing Pipeline\n")

# -------------------------------
# STEP 1: CREATE REALISTIC DATASET
# -------------------------------
# This dataset simulates a warehouse robot handling:
# - Inventory management
# - Order processing
# - Delivery preparation
# - Sensor-based navigation

np.random.seed(42)
n = 200

data = pd.DataFrame({

    # Sensor inputs (robot environment)
    "Battery_Level (%)": np.random.normal(70, 15, n),
    "Temperature (C)": np.random.normal(25, 4, n),
    "Humidity (%)": np.random.normal(55, 10, n),
    "Obstacle_Distance (m)": np.random.normal(2, 1, n),

    # Inventory related features
    "Perishable_Stock": np.random.randint(50, 200, n),
    "NonPerishable_Stock": np.random.randint(100, 500, n),
    "Reorder_Level": np.random.randint(60, 150, n),

    # Order & demand features
    "Orders_Received": np.random.randint(20, 100, n),
    "Priority_Orders": np.random.randint(5, 30, n),

    # Delivery preparation
    "Packages_Prepared": np.random.randint(10, 80, n),
    "Drone_Requests": np.random.randint(5, 40, n),

    # Robot operation metrics
    "Picking_Time (min)": np.random.normal(15, 5, n),
    "Sorting_Time (min)": np.random.normal(10, 3, n),

    # Target-like feature (efficiency score)
    "Efficiency_Score": np.random.normal(75, 10, n)
})

# Ensure no negative values
data = data.clip(lower=0)

print("STEP 1: Initial Dataset (First 5 rows)\n")
print(data.head())


# -------------------------------
# STEP 2: FEATURE ENGINEERING
# -------------------------------
# Create derived features to improve understanding

data["Total_Stock"] = data["Perishable_Stock"] + data["NonPerishable_Stock"]
data["Order_Fulfillment_Ratio"] = data["Packages_Prepared"] / (data["Orders_Received"] + 1)
data["Robot_Load"] = data["Orders_Received"] + data["Drone_Requests"]
data["Stock_Gap"] = data["Reorder_Level"] - data["Perishable_Stock"]

print("\nSTEP 2: After Feature Engineering\n")
print(data.head())


# -------------------------------
# STEP 3: INTRODUCE MISSING VALUES
# -------------------------------
# Simulate real-world incomplete sensor and system data

data_missing = data.copy()

for col in data_missing.columns:
    num_missing = np.random.randint(10, 20)
    indices = np.random.choice(data_missing.index, num_missing, replace=False)
    data_missing.loc[indices, col] = np.nan

print("\nSTEP 3: Missing Values Count BEFORE Imputation\n")
print(data_missing.isnull().sum())


# -------------------------------
# STEP 4: HANDLE MISSING VALUES
# -------------------------------
# Using mean strategy for numerical data

imputer = SimpleImputer(strategy='mean')

data_imputed = pd.DataFrame(
    imputer.fit_transform(data_missing),
    columns=data_missing.columns
)

print("\nSTEP 4: Missing Values Count AFTER Imputation\n")
print(data_imputed.isnull().sum())


# -------------------------------
# STEP 5: SPLIT DATASET
# -------------------------------
# Separate features and target

X = data_imputed.drop("Efficiency_Score", axis=1)
y = data_imputed["Efficiency_Score"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nSTEP 5: Train-Test Split")
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# -------------------------------
# STEP 6: FEATURE SCALING
# -------------------------------
# Standardize features for better ML performance

scaler = StandardScaler()

# Fit ONLY on training data
X_train_scaled = scaler.fit_transform(X_train)

# Transform test data using same scaler
X_test_scaled = scaler.transform(X_test)

# Convert to DataFrame for readability
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns)

print("\nSTEP 6: Scaled Training Data (First 5 rows)\n")
print(X_train_scaled_df.head())

print("\nSTEP 6: Scaled Test Data (First 5 rows)\n")
print(X_test_scaled_df.head())


# -------------------------------
# FINAL NOTE (IMPORTANT CONCEPT)
# -------------------------------
print("\nIMPORTANT NOTES:")

print("""
1. Dataset simulates real warehouse robot operations including:
   - Sensors (battery, temperature, obstacles)
   - Inventory tracking
   - Order handling and delivery preparation

2. Feature engineering improves model understanding by adding:
   - Total stock
   - Load on robot
   - Fulfillment efficiency

3. Missing values are handled using mean imputation.

4. Scaling is applied ONLY on training data to avoid data leakage.

5. The same scaler is used on test data to ensure consistency.

6. This pipeline represents a complete real-world preprocessing workflow.
""")