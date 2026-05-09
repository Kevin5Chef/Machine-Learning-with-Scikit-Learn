# Step 1: Import required libraries
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 2: Load dataset
iris = load_iris()
X = iris.data
y = iris.target  # Classification target

print("Target Classes:", iris.target_names)

# Step 3: Introduce artificial missing values (for practice)
np.random.seed(42)
missing_mask = np.random.rand(*X.shape) < 0.1
X[missing_mask] = np.nan

print("\nMissing values before cleaning:", np.isnan(X).sum())

# Step 4: Handle missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

print("Missing values after cleaning:", np.isnan(X_imputed).sum())

# Step 5: Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Step 6: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

# Step 7: Train Logistic Regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Step 8: Predictions
y_pred = model.predict(X_test)

# Step 9: Evaluation
accuracy = accuracy_score(y_test, y_pred)

print("\n--- Classification Evaluation ---")
print("Accuracy:", accuracy)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Step 10: Show sample predictions
print("\nSample Predictions:")
for i in range(10):
    print(f"Actual: {iris.target_names[y_test[i]]}, Predicted: {iris.target_names[y_pred[i]]}")