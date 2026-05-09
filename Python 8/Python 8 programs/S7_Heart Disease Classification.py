import numpy as np
import pandas as pd
print("SY-5, Kevin Victor, Roll No.-30")
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# -------------------------------
# STEP 1: Create Realistic Dataset
# -------------------------------

np.random.seed(42)
n = 400

data = pd.DataFrame({
    # Demographics
    "age": np.random.randint(30, 85, n),
    "gender": np.random.choice(["Male", "Female"], n),

    # Clinical Features
    "blood_pressure": np.random.randint(90, 180, n),
    "cholesterol": np.random.randint(150, 300, n),
    "max_heart_rate": np.random.randint(70, 200, n),

    # Symptoms (0-1 scale)
    "chest_pain": np.random.uniform(0, 1, n),
    "shortness_breath": np.random.uniform(0, 1, n),
    "fatigue": np.random.uniform(0, 1, n),
    "dizziness": np.random.uniform(0, 1, n),
    "palpitations": np.random.uniform(0, 1, n),

    # Lifestyle
    "smoking": np.random.choice([0, 1], n),
    "diabetes": np.random.choice([0, 1], n),
    "obesity": np.random.choice([0, 1], n),

    "lifestyle_index": np.random.uniform(1, 10, n)
})

# -------------------------------
# STEP 2: Feature Engineering
# -------------------------------

data["risk_score"] = (
    data["age"] * 0.3 +
    data["blood_pressure"] * 0.2 +
    data["cholesterol"] * 0.2 +
    data["smoking"] * 15 +
    data["diabetes"] * 10 +
    data["obesity"] * 8
)

data["symptom_severity"] = (
    data["chest_pain"] +
    data["shortness_breath"] +
    data["fatigue"] +
    data["dizziness"] +
    data["palpitations"]
)

# -------------------------------
# STEP 3: Create Target (3 Classes)
# -------------------------------

conditions = []

for i in range(n):
    if data["risk_score"][i] > 180 and data["chest_pain"][i] > 0.6:
        conditions.append("CAD")
    elif data["symptom_severity"][i] > 2.5 and data["fatigue"][i] > 0.7:
        conditions.append("Heart Failure")
    else:
        conditions.append("Arrhythmia")

data["disease"] = conditions

# -------------------------------
# STEP 4: Introduce Missing Values
# -------------------------------

for col in data.columns:
    if col != "disease":
        data.loc[data.sample(frac=0.05).index, col] = np.nan

pd.set_option('display.max_columns', None)

print("\nInitial Dataset:\n")
print(data.head())

# -------------------------------
# STEP 5: Preprocessing
# -------------------------------

# Encode categorical
le = LabelEncoder()
data["gender"] = le.fit_transform(data["gender"])

X = data.drop("disease", axis=1)
y = data["disease"]

# Impute missing values
imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(X)

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Encode target
y = LabelEncoder().fit_transform(y)

# -------------------------------
# STEP 6: Train-Test Split
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# -------------------------------
# STEP 7: Train KNN Model
# -------------------------------

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# -------------------------------
# STEP 8: Predictions
# -------------------------------

predictions = knn.predict(X_test)

# -------------------------------
# STEP 9: Evaluation
# -------------------------------

accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:", accuracy)

print("\nClassification Report:\n")
print(classification_report(y_test, predictions))

# -------------------------------
# STEP 10: Sample Predictions
# -------------------------------

results = pd.DataFrame({
    "Actual": y_test,
    "Predicted": predictions
})

print("\nSample Predictions:\n")
print(results.head(10))

print("CAD never got predicted because it had the least samples and had very strict conditions to be classified as CAD. \nThe model likely struggled to learn the patterns for CAD due to its rarity and the specific feature thresholds required for classification.")
