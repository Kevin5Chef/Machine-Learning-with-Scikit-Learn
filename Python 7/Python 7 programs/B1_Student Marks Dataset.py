import pandas as pd
import numpy as np
print("SY-5, Kevin Victor, Roll No.-30")
# Set seed for reproducibility (optional)
np.random.seed(42)

# Number of students
num_students = 100

# Generate marks with realistic trends
# OS & CN -> tougher subjects (lower mean)
os_marks = np.random.normal(loc=60, scale=10, size=num_students)
cn_marks = np.random.normal(loc=58, scale=12, size=num_students)

# AI & Python -> easier scoring (higher mean)
ai_marks = np.random.normal(loc=75, scale=8, size=num_students)
python_marks = np.random.normal(loc=78, scale=7, size=num_students)

# Clip values to stay within 0–100
os_marks = np.clip(os_marks, 0, 100)
cn_marks = np.clip(cn_marks, 0, 100)
ai_marks = np.clip(ai_marks, 0, 100)
python_marks = np.clip(python_marks, 0, 100)

# Create DataFrame
df = pd.DataFrame({
    "Operating Systems": os_marks.astype(int),
    "Computer Networks": cn_marks.astype(int),
    "Artificial Intelligence": ai_marks.astype(int),
    "Python": python_marks.astype(int)
})

# Display first 5 rows
print("First 5 rows of dataset:\n")
print(df.head())

# Print column names
print("\nColumn Names:\n")
print(df.columns)

# Display basic structure
print("\nDataset Structure:\n")
print(df.info())