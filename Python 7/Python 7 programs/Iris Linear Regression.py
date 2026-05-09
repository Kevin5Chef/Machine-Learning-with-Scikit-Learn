# Step 1: Import required libraries
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Step 2: Load the dataset
iris = load_iris()
X = iris.data
y = iris.data[:, 0]  # Predicting sepal length (column 0)

# Remove sepal length from features to avoid leakage
X = np.delete(X, 0, axis=1)

print("Original shape of X:", X.shape)

# Step 3: Introduce artificial missing values (for demonstration)
np.random.seed(42)
missing_mask = np.random.rand(*X.shape) < 0.1  # 10% missing
X[missing_mask] = np.nan

print("\nNumber of missing values:", np.isnan(X).sum())

# Step 4: Handle missing values using mean imputation
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

print("\nMissing values after imputation:", np.isnan(X_imputed).sum())

# Step 5: Feature Scaling (important for regression)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Step 6: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

# Step 7: Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 8: Predictions
y_pred = model.predict(X_test)

# Step 9: Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Model Evaluation ---")
print("Mean Squared Error:", mse)
print("R² Score:", r2)

# Step 10: Compare actual vs predicted values
print("\nSample Predictions:")
for i in range(10):
    print(f"Actual: {y_test[i]:.2f}, Predicted: {y_pred[i]:.2f}")