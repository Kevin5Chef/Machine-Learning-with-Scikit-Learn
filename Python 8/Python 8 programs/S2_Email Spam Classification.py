import numpy as np
import pandas as pd
print("SY-5, Kevin Victor, Roll No.-30")
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

pd.set_option('display.max_colwidth', None)   # Show full text (no truncation)
# -------------------------------
# STEP 1: Create Dataset
# -------------------------------

np.random.seed(42)
n = 400

data = pd.DataFrame({

    # Basic spam indicators
    "contains_links": np.random.choice([0, 1], n),
    "num_exclamations": np.random.randint(0, 10, n),
    "contains_promo_words": np.random.choice([0, 1], n),

    # Advanced features
    "sender_reputation": np.random.uniform(0, 1, n),   # low → suspicious
    "historical_spam_score": np.random.uniform(0, 1, n),
    "unusual_email_id": np.random.choice([0, 1], n),

    # Trojan-style deception features
    "formal_tone_score": np.random.uniform(0, 1, n),
    "hidden_links": np.random.choice([0, 1], n),
    "payment_keywords": np.random.choice([0, 1], n),
    "urgent_tone": np.random.choice([0, 1], n),

    # User interaction
    "user_unsubscribed": np.random.choice([0, 1], n),
    "previous_interaction": np.random.choice([0, 1], n)
})

# -------------------------------
# STEP 2: Feature Engineering
# -------------------------------

data["deception_score"] = (
    data["formal_tone_score"] * 0.5 +
    data["hidden_links"] * 0.8 +
    data["urgent_tone"] * 0.6
)

data["risk_score"] = (
    (1 - data["sender_reputation"]) * 2 +
    data["historical_spam_score"] * 2 +
    data["unusual_email_id"] * 1.5 +
    data["payment_keywords"] * 1.2 +
    data["contains_links"] * 1
)

# -------------------------------
# STEP 3: Create Target (Balanced)
# -------------------------------

labels = []

for i in range(n):

    if data["risk_score"][i] > 3.5:
        labels.append(1)

    elif data["deception_score"][i] > 1.2 and data["formal_tone_score"][i] > 0.7:
        labels.append(1)

    elif data["contains_promo_words"][i] == 1 and data["user_unsubscribed"][i] == 1:
        labels.append(1)

    else:
        labels.append(0)

data["spam"] = labels

print("\nClass Distribution BEFORE balancing:\n")
print(data["spam"].value_counts())

# -------------------------------
# STEP 3.5: FORCE CLASS BALANCE
# -------------------------------

spam_data = data[data["spam"] == 1]
not_spam_data = data[data["spam"] == 0]

# Force exact numbers
spam_data = spam_data.sample(220, random_state=42, replace=True)
not_spam_data = not_spam_data.sample(180, random_state=42, replace=True)

# Combine and shuffle
data = pd.concat([spam_data, not_spam_data])
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

print("\nClass Distribution AFTER balancing:\n")
print(data["spam"].value_counts())

# -------------------------------
# STEP 4: Preprocessing
# -------------------------------

X = data.drop("spam", axis=1)
y = data["spam"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# STEP 5: Train-Test Split
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# -------------------------------
# STEP 6: Train Logistic Regression
# -------------------------------

model = LogisticRegression()
model.fit(X_train, y_train)

# -------------------------------
# STEP 7: Predictions
# -------------------------------

predictions = model.predict(X_test)

# -------------------------------
# STEP 8: Evaluation
# -------------------------------

accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:", accuracy)

print("\nClassification Report:\n")
print(classification_report(y_test, predictions))

# -------------------------------
# STEP 9: Explain Predictions (REASON ENGINE)
# -------------------------------

def generate_reason(row):
    reasons = []

    if row["sender_reputation"] < 0.3:
        reasons.append("Low sender reputation")

    if row["historical_spam_score"] > 0.6:
        reasons.append("Sender has spam history")

    if row["hidden_links"] == 1:
        reasons.append("Contains hidden/masked links")

    if row["payment_keywords"] == 1:
        reasons.append("Contains payment-related keywords")

    if row["urgent_tone"] == 1:
        reasons.append("Creates urgency (clickbait risk)")

    if row["formal_tone_score"] > 0.7 and row["hidden_links"] == 1:
        reasons.append("Trojan-style formal deception")

    if row["contains_promo_words"] == 1 and row["user_unsubscribed"] == 1:
        reasons.append("Unwanted promotional email")

    if row["unusual_email_id"] == 1:
        reasons.append("Suspicious email ID format")

    if len(reasons) == 0:
        reasons.append("Legitimate email behavior detected")

    return reasons

# -------------------------------
# STEP 10: Show Sample Predictions + Improved Reasons
# -------------------------------

X_test_df = pd.DataFrame(scaler.inverse_transform(X_test), columns=X.columns)

results = []

for i in range(10):
    row = X_test_df.iloc[i]

    if predictions[i] == 1:
        # Spam → detailed reasons
        reasons = generate_reason(row)
        reason_text = " | ".join(reasons)

    else:
        # Not Spam → clear safe explanation
        reason_text = (
            "Email classified as SAFE: follows normal communication patterns | "
            "Sender appears trustworthy | No strong spam indicators detected | "
            "Any links or payment references are likely legitimate and expected"
        )

    results.append({
        "Predicted": predictions[i],
        "Reasons": reason_text
    })

results_df = pd.DataFrame(results)

print("\nSample Predictions with Reasons:\n")
print(results_df.to_string(index=False))