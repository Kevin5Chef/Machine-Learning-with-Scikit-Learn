import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
print("SY-5, Kevin Victor, Roll No.-30")
# Set seed for reproducibility
np.random.seed(42)

num_students = 100

# Generate marks (same realistic trend)
os_marks = np.random.normal(loc=60, scale=10, size=num_students)
cn_marks = np.random.normal(loc=58, scale=12, size=num_students)
ai_marks = np.random.normal(loc=75, scale=8, size=num_students)
python_marks = np.random.normal(loc=78, scale=7, size=num_students)

# Clip to 0–100 and convert to integers
os_marks = np.clip(os_marks, 0, 100).astype(int)
cn_marks = np.clip(cn_marks, 0, 100).astype(int)
ai_marks = np.clip(ai_marks, 0, 100).astype(int)
python_marks = np.clip(python_marks, 0, 100).astype(int)

# Create grades based on average marks
avg_marks = (os_marks + cn_marks + ai_marks + python_marks) / 4

grades = []
for avg in avg_marks:
    if avg >= 75:
        grades.append("A")
    elif avg >= 60:
        grades.append("B")
    else:
        grades.append("C")

# Create DataFrame
df = pd.DataFrame({
    "Operating Systems": os_marks,
    "Computer Networks": cn_marks,
    "Artificial Intelligence": ai_marks,
    "Python": python_marks,
    "Grade": grades
})

# -------------------------------
# Introduce missing values
# -------------------------------
df_missing = df.copy()

for col in df_missing.columns:
    num_missing = np.random.randint(10, 21)  # 10 to 20 missing values
    missing_indices = np.random.choice(df_missing.index, num_missing, replace=False)
    df_missing.loc[missing_indices, col] = np.nan

# -------------------------------
# Count missing values BEFORE
# -------------------------------
print("Missing values BEFORE imputation:\n")
print(df_missing.isnull().sum())

# -------------------------------
# Apply SimpleImputer (most_frequent)
# -------------------------------
imputer = SimpleImputer(strategy='most_frequent')

df_imputed = pd.DataFrame(
    imputer.fit_transform(df_missing),
    columns=df_missing.columns
)

# Convert numeric columns back to int
numeric_cols = ["Operating Systems", "Computer Networks", "Artificial Intelligence", "Python"]
df_imputed[numeric_cols] = df_imputed[numeric_cols].astype(int)

# -------------------------------
# Count missing values AFTER
# -------------------------------
print("\nMissing values AFTER imputation:\n")
print(df_imputed.isnull().sum())

# -------------------------------
# Display corrected dataset
# -------------------------------
print("\nCorrected DataFrame (First 10 rows):\n")
print(df_imputed.head(10))