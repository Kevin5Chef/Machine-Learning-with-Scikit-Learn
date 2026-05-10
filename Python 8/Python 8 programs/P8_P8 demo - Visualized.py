print("SY-5, Kevin Victor, Roll No.-30")
# ==============================
# Step 1: Import libraries
# ==============================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ==============================
# Step 2: Generate improved dataset
# ==============================
np.random.seed(42)

n = 300

study_hours = np.random.uniform(0, 10, n)

# Softer logistic curve (less steep)
logit = 0.8 * (study_hours - 5)

# Add noise
noise = np.random.normal(0, 1, n)

# Final probability
probability = 1 / (1 + np.exp(-(logit + noise)))

# Sample pass/fail (IMPORTANT change)
pass_status = np.random.binomial(1, probability)

df = pd.DataFrame({
    'study_hours': study_hours,
    'pass': pass_status
})

print("\nDataset Preview:")
print(df.head())

# ==============================
# Step 3: Train-test split
# ==============================
X = df[['study_hours']]
y = df['pass']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ==============================
# Step 4: Train model
# ==============================
model = LogisticRegression()
model.fit(X_train, y_train)

# ==============================
# Step 5: Predictions
# ==============================
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# ==============================
# Actual vs Predicted Comparison
# ==============================
comparison_df = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_pred
})

print("\nActual vs Predicted Values:\n")
print(comparison_df.head(10))

# ==============================
# Step 6: Evaluation
# ==============================
print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ==============================
# Step 7: Sample predictions
# ==============================
print("\nSample Predictions:")
for i in range(10):
    print(f"Study Hours: {X_test.iloc[i,0]:.2f}, "
          f"Probability: {y_prob[i]:.2f}, "
          f"Predicted: {'Pass' if y_pred[i]==1 else 'Fail'}")

# ==============================
# Step 8: Visualization
# ==============================

# Create smooth curve
x_range = np.linspace(0, 10, 100)
logit_curve = 0.8 * (x_range - 5)
prob_curve = 1 / (1 + np.exp(-logit_curve))

plt.figure(figsize=(10, 6))

# Scatter actual data
plt.scatter(X, y, alpha=0.5, label="Actual Data")

# Plot probability curve
plt.plot(x_range, prob_curve, color='red', label="Ideal Probability Curve")

# FIX APPLIED HERE: Convert to DataFrame with proper column name
x_range_df = pd.DataFrame(x_range, columns=['study_hours'])
model_probs = model.predict_proba(x_range_df)[:, 1]

# Plot model prediction curve
plt.plot(x_range, model_probs, color='green', linestyle='--', label="Model Curve")

plt.axhline(0.5, color='black', linestyle=':', label='Decision Boundary')

plt.xlabel("Study Hours")
plt.ylabel("Probability of Passing")
plt.title("Improved Study Hours vs Pass Probability")
plt.legend()
plt.grid(True)

plt.show()