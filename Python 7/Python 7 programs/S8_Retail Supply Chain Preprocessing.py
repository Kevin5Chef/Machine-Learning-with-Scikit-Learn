import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
print("SY-5, Kevin Victor, Roll No.-30")
# -------------------------------
# DISPLAY SETTINGS
# -------------------------------
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1200)

print("Retail Supply Chain Dataset - Missing Value Handling\n")

# -------------------------------
# STEP 1: CREATE REALISTIC DATASET
# -------------------------------
np.random.seed(42)
n = 200

data = pd.DataFrame({

    # Customer purchase data
    "Purchase_Amount": np.random.normal(1500, 500, n),
    "Items_Per_Order": np.random.randint(1, 15, n),
    "Most_Frequent_Item_Price": np.random.normal(200, 50, n),

    # Inventory data
    "Perishable_Stock": np.random.randint(50, 300, n),
    "NonPerishable_Stock": np.random.randint(100, 800, n),
    "Shelf_Utilization (%)": np.random.uniform(50, 100, n),

    # Supply-demand features
    "Daily_Demand": np.random.randint(50, 300, n),
    "Supply_Arrival": np.random.randint(40, 250, n),
    "Reorder_Level": np.random.randint(60, 200, n),

    # Delivery pipeline
    "Orders_for_Delivery": np.random.randint(20, 150, n),
    "Avg_Delivery_Time (min)": np.random.normal(40, 10, n),
    "Warehouse_to_Hub_Time (min)": np.random.normal(20, 5, n),

    # Profit-related
    "Profit_per_Order": np.random.normal(300, 100, n)
})

# Ensure no negative values
data = data.clip(lower=0)

print("STEP 1: Initial Dataset (First 5 rows)\n")
print(data.head())


# -------------------------------
# STEP 2: FEATURE ENGINEERING
# -------------------------------
# Create meaningful derived features

data["Total_Stock"] = data["Perishable_Stock"] + data["NonPerishable_Stock"]
data["Demand_Supply_Gap"] = data["Daily_Demand"] - data["Supply_Arrival"]
data["Avg_Item_Value"] = data["Purchase_Amount"] / (data["Items_Per_Order"] + 1)
data["Logistics_Load"] = data["Orders_for_Delivery"] + data["Daily_Demand"]
data["Profit_Margin_Estimate"] = data["Profit_per_Order"] / (data["Purchase_Amount"] + 1)

print("\nSTEP 2: After Feature Engineering\n")
print(data.head())


# -------------------------------
# STEP 3: INTRODUCE MISSING VALUES
# -------------------------------
# Focus especially on Purchase_Amount

data_missing = data.copy()

for col in data_missing.columns:
    num_missing = np.random.randint(10, 20)
    indices = np.random.choice(data_missing.index, num_missing, replace=False)
    data_missing.loc[indices, col] = np.nan

print("\nSTEP 3: Missing Values BEFORE Imputation\n")
print(data_missing.isnull().sum())


# -------------------------------
# STEP 4: HANDLE MISSING VALUES
# -------------------------------
# Use mean strategy (common for numeric retail data)

imputer = SimpleImputer(strategy='mean')

data_imputed = pd.DataFrame(
    imputer.fit_transform(data_missing),
    columns=data_missing.columns
)

print("\nSTEP 4: Missing Values AFTER Imputation\n")
print(data_imputed.isnull().sum())


# -------------------------------
# STEP 5: TRAIN-TEST SPLIT
# -------------------------------
# Target variable: Profit_per_Order

X = data_imputed.drop("Profit_per_Order", axis=1)
y = data_imputed["Profit_per_Order"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nSTEP 5: Train-Test Split")
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# -------------------------------
# FINAL OUTPUT SAMPLE
# -------------------------------
print("\nSample Training Data (First 5 rows):\n")
print(X_train.head())

print("\nSample Testing Data (First 5 rows):\n")
print(X_test.head())


# -------------------------------
# EXPLANATION
# -------------------------------
print("\nExplanation:")

print("""
1. The dataset simulates a real retail and supply-chain system including:
   - Customer purchases
   - Inventory management
   - Supply-demand balance
   - Delivery logistics
   - Profit estimation

2. Feature engineering enhances understanding by adding:
   - Total stock
   - Demand-supply gap
   - Customer buying trends
   - Logistics load

3. Missing values were introduced artificially to simulate real-world data issues.

4. SimpleImputer (mean strategy) replaces missing values with the average of each column.

5. After imputation:
   - No missing values remain
   - Dataset becomes suitable for machine learning

6. Dataset is split into training and testing sets for model development.

7. Target variable selected: Profit_per_Order
""")