import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# -------------------------------
# Pandas display fix (ONLY ADDITION)
# -------------------------------
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print("SY-5, Kevin Victor, Roll No.-30")

# -------------------------------
# STEP 1: Create a realistic dataset
# -------------------------------
np.random.seed(42)
n = 200  # number of samples

data = pd.DataFrame({
    # Environmental features
    "Temperature (C)": np.random.normal(26, 3, n),
    "Humidity (%)": np.random.normal(60, 10, n),
    "Light Intensity (lux)": np.random.normal(300, 80, n),

    # Occupancy (0 = not occupied, 1 = occupied)
    "Occupancy": np.random.choice([0, 1], n, p=[0.3, 0.7]),

    # Appliance usage (minutes per day)
    "TV Usage (min)": np.random.normal(120, 40, n),
    "Refrigerator Usage (min)": np.random.normal(1440, 1, n),  # nearly constant
    "Washing Machine (min)": np.random.normal(30, 20, n),
    "Dishwasher (min)": np.random.normal(25, 15, n),
    "Microwave (min)": np.random.normal(15, 10, n),

    # Target-like feature
    "Energy Consumption (kWh)": np.random.normal(8, 2, n)
})

# Ensure no negative values
data = data.clip(lower=0)

print("Original Dataset (First 5 rows):\n")
print(data.head())


# -------------------------------
# STEP 2: Train-Test Split
# -------------------------------
X = data.drop("Energy Consumption (kWh)", axis=1)
y = data["Energy Consumption (kWh)"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTrain-Test Split:")
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# -------------------------------
# STEP 3: Apply StandardScaler
# -------------------------------
scaler = StandardScaler()

# Fit only on training data
X_train_scaled = scaler.fit_transform(X_train)

# Use the same scaler to transform test data
X_test_scaled = scaler.transform(X_test)

print("\nScaling completed.")
print("Training data scaled using fit_transform().")
print("Test data scaled using transform() only.")


# -------------------------------
# STEP 4: Display sample results
# -------------------------------
print("\nSample Scaled Training Data:\n")
print(X_train_scaled[:5])

print("\nSample Scaled Test Data:\n")
print(X_test_scaled[:5])


# -------------------------------
# STEP 5: Explanation
# -------------------------------
print("\nExplanation:")

print("""
1. The StandardScaler computes mean and standard deviation from the training data only.

2. These statistics are then used to transform both training and test data.

3. The scaler must NOT be fitted on test data because:

   - It introduces data leakage: information from test data influences the model.
   - It leads to unrealistic evaluation since test data represents unseen future data.
   - It artificially improves model performance, giving misleading results.

4. Correct workflow:
   - Fit scaler on training data.
   - Apply the same transformation to test data.

5. Analogy:
   Training data represents known historical data used to build the model.
   Test data represents unseen future data and must remain independent.
""")