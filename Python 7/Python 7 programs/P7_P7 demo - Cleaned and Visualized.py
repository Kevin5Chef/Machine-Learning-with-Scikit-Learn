print("SY-5, Kevin Victor, Roll No.-30")
# ==============================
# Step 1: Import libraries
# ==============================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ==============================
# Step 2: Generate dataset
# ==============================
np.random.seed(42)

n = 300

study_hours = np.random.uniform(0, 10, n)

# Linear relationship + noise
scores = 5 * study_hours + 50 + np.random.normal(0, 10, n)

scores = np.clip(scores, 0, 100)

df = pd.DataFrame({
    'study_hours': study_hours,
    'score': scores
})

# ==============================
# Step 3: Introduce missing values
# ==============================
for col in df.columns:
    df.loc[df.sample(frac=0.05).index, col] = np.nan

print("\nMissing values before cleaning:\n", df.isnull().sum())

# ==============================
# Step 4: Handle missing values
# ==============================
df.fillna(df.mean(), inplace=True)

print("\nMissing values after cleaning:\n", df.isnull().sum())

# ==============================
# Step 5: Remove outliers (IQR)
# ==============================
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

df_clean = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

print("\nDataset size after outlier removal:", df_clean.shape)

# ==============================
# Step 6: Split data
# ==============================
X = df_clean[['study_hours']]
y = df_clean['score']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ==============================
# Step 7: Train model
# ==============================
model = LinearRegression()
model.fit(X_train, y_train)

# ==============================
# Step 8: Predictions
# ==============================
y_pred = model.predict(X_test)

# ==============================
# Display Actual vs Predicted
# ==============================
comparison_df = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_pred
})

print("\nActual vs Predicted Values:\n")
print(comparison_df.head(10))

# ==============================
# Step 9: Evaluation
# ==============================
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Custom "accuracy" (within ±10 marks)
tolerance = 10
accuracy_custom = np.mean(np.abs(y_test - y_pred) <= tolerance)

print("\nMean Squared Error:", mse)
print("R² Score:", r2)
print(f"Custom Accuracy (within ±{tolerance} marks): {accuracy_custom:.2f}")

# ==============================
# Step 10: Visualization
# ==============================
plt.figure(figsize=(10, 6))

# Scatter actual data
plt.scatter(X_test, y_test, color='blue', label='Actual Scores')

# Sort for smooth line
sorted_indices = np.argsort(X_test.values.flatten())
X_sorted = X_test.values.flatten()[sorted_indices]
y_sorted = y_pred[sorted_indices]

# Regression line
plt.plot(X_sorted, y_sorted, color='red', linewidth=2, label='Regression Line')

plt.xlabel("Study Hours")
plt.ylabel("Score")
plt.title("Study Hours vs Student Scores (Linear Regression)")
plt.legend()
plt.grid(True)

plt.show()