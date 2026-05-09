import numpy as np
import pandas as pd
print("SY-5, Kevin Victor, Roll No.-30")
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

np.random.seed(42)

# -------------------------------
# STEP 1: Define Fruit Profiles (DISTINCT FEATURES)
# -------------------------------

fruit_profiles = {
    "Apple":        {"weight": (150, 250), "size": (6, 9), "shape": 1, "curvature": 0, "sweet": (6,8), "texture": 7, "rarity": 0.2},
    "Banana":       {"weight": (100, 200), "size": (15, 25), "shape": 2, "curvature": 9, "sweet": (7,9), "texture": 5, "rarity": 0.1},
    "Orange":       {"weight": (120, 250), "size": (6, 10), "shape": 1, "curvature": 0, "sweet": (5,7), "texture": 6, "rarity": 0.2},
    "Mango":        {"weight": (200, 500), "size": (8, 15), "shape": 2, "curvature": 2, "sweet": (8,10), "texture": 6, "rarity": 0.3},
    "Pineapple":    {"weight": (800, 2000), "size": (15, 30), "shape": 3, "curvature": 0, "sweet": (6,8), "texture": 9, "rarity": 0.4},
    "Watermelon":   {"weight": (2000, 8000), "size": (20, 40), "shape": 1, "curvature": 0, "sweet": (7,9), "texture": 5, "rarity": 0.3},
    "Papaya":       {"weight": (500, 1500), "size": (15, 25), "shape": 2, "curvature": 1, "sweet": (6,8), "texture": 4, "rarity": 0.3},
    "Grapes":       {"weight": (5, 10), "size": (1, 2), "shape": 1, "curvature": 0, "sweet": (6,8), "texture": 3, "rarity": 0.2},
    "Strawberry":   {"weight": (10, 20), "size": (2, 4), "shape": 4, "curvature": 1, "sweet": (5,7), "texture": 4, "rarity": 0.3},
    "Blueberry":    {"weight": (1, 5), "size": (1, 2), "shape": 1, "curvature": 0, "sweet": (6,8), "texture": 3, "rarity": 0.4},
    "Kiwi":         {"weight": (50, 100), "size": (4, 7), "shape": 2, "curvature": 0, "sweet": (5,7), "texture": 8, "rarity": 0.4},
    "Dragon Fruit": {"weight": (300, 800), "size": (10, 15), "shape": 2, "curvature": 0, "sweet": (6,8), "texture": 9, "rarity": 0.9},
    "Lychee":       {"weight": (10, 25), "size": (2, 4), "shape": 1, "curvature": 0, "sweet": (7,9), "texture": 6, "rarity": 0.5},
    "Guava":        {"weight": (100, 300), "size": (5, 10), "shape": 1, "curvature": 0, "sweet": (5,7), "texture": 6, "rarity": 0.3},
    "Peach":        {"weight": (100, 200), "size": (5, 8), "shape": 1, "curvature": 0, "sweet": (6,8), "texture": 7, "rarity": 0.4},
    "Pear":         {"weight": (150, 300), "size": (7, 12), "shape": 4, "curvature": 1, "sweet": (6,8), "texture": 6, "rarity": 0.3},
    "Plum":         {"weight": (50, 100), "size": (3, 6), "shape": 1, "curvature": 0, "sweet": (5,7), "texture": 5, "rarity": 0.4},
    "Cherry":       {"weight": (5, 10), "size": (1, 2), "shape": 1, "curvature": 0, "sweet": (6,8), "texture": 4, "rarity": 0.5},
    "Apricot":      {"weight": (30, 70), "size": (3, 6), "shape": 1, "curvature": 0, "sweet": (6,8), "texture": 5, "rarity": 0.4},
    "Fig":          {"weight": (40, 100), "size": (4, 7), "shape": 2, "curvature": 0, "sweet": (7,9), "texture": 7, "rarity": 0.6},
    "Pomegranate":  {"weight": (200, 500), "size": (7, 12), "shape": 1, "curvature": 0, "sweet": (5,7), "texture": 8, "rarity": 0.5},
    "Avocado":      {"weight": (150, 300), "size": (7, 12), "shape": 2, "curvature": 0, "sweet": (3,5), "texture": 9, "rarity": 0.6},
    "Coconut":      {"weight": (1000, 3000), "size": (15, 25), "shape": 1, "curvature": 0, "sweet": (4,6), "texture": 10, "rarity": 0.5},
    "Honeydew Melon":{"weight": (1500, 4000), "size": (20, 35), "shape": 1, "curvature": 0, "sweet": (7,9), "texture": 5, "rarity": 0.4},
    "Passion Fruit":{"weight": (50, 150), "size": (4, 7), "shape": 1, "curvature": 0, "sweet": (6,8), "texture": 6, "rarity": 0.8},
}

# -------------------------------
# STEP 2: Generate Dataset
# -------------------------------

rows = []

for fruit, props in fruit_profiles.items():
    for _ in range(25):  # samples per fruit
        rows.append({
            "fruit": fruit,
            "weight": np.random.uniform(*props["weight"]),
            "size": np.random.uniform(*props["size"]),
            "shape_index": props["shape"],
            "curvature": props["curvature"],
            "sweetness": np.random.uniform(*props["sweet"]),
            "texture": props["texture"],
            "rarity": props["rarity"]
        })

data = pd.DataFrame(rows)

pd.set_option('display.max_columns', None)
print("\nDataset:\n")
print(data.head())

# -------------------------------
# STEP 3: Encode Labels
# -------------------------------

le = LabelEncoder()
data["label"] = le.fit_transform(data["fruit"])

X = data.drop(["fruit", "label"], axis=1)
y = data["label"]

# -------------------------------
# STEP 4: Scaling
# -------------------------------

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# STEP 5: Train-Test Split
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# -------------------------------
# STEP 6: Train KNN
# -------------------------------

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# -------------------------------
# STEP 7: Predictions
# -------------------------------

preds = knn.predict(X_test)

# -------------------------------
# STEP 8: Accuracy
# -------------------------------

print("\nModel Accuracy:", accuracy_score(y_test, preds))

# -------------------------------
# STEP 9: Sample Predictions
# -------------------------------

sample = pd.DataFrame({
    "Actual": le.inverse_transform(y_test[:10]),
    "Predicted": le.inverse_transform(preds[:10])
})

print("\nSample Predictions:\n")
print(sample)