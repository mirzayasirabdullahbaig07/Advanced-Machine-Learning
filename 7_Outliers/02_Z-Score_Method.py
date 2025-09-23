# ============================================================
#                Z-SCORE METHOD FOR OUTLIERS
# ============================================================

# -------------------------------
# What is Z-Score?
# -------------------------------
# Z-Score = (X - Mean) / Standard Deviation
# It represents how many standard deviations a data point (X)
# is away from the mean.
#
# Example:
#   If Z = 0 → value = mean
#   If Z = +2 → value is 2 std above mean
#   If Z = -3 → value is 3 std below mean
#
# Common Threshold:
#   If |Z| > 3 → data is considered an outlier.

# -------------------------------
# What is Normal Distribution?
# -------------------------------
# - Bell-shaped curve.
# - Mean = Median = Mode.
# - 68% of data lies within 1 std, 
#   95% within 2 std, 
#   99.7% within 3 std (Empirical Rule).

# -------------------------------
# Treatment of Outliers
# -------------------------------
# 1. Trimming:
#    - Remove rows with outliers.
#    - Disadvantage: reduces dataset size.
#
# 2. Capping:
#    - Replace outliers with boundary values.
#    - Keeps dataset size intact.


# ============================================================
#                  IMPLEMENTATION IN PYTHON
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('placement.csv')
print("Shape of dataset:", df.shape)
print(df.sample(5))

# -------------------------------
# Distribution Plots
# -------------------------------
plt.figure(figsize=(16,5))

plt.subplot(1,2,1)
sns.distplot(df['cgpa'])
plt.title("Distribution of CGPA")

plt.subplot(1,2,2)
sns.distplot(df['placement_exam_marks'])
plt.title("Distribution of Placement Exam Marks")

plt.show()

# -------------------------------
# Summary Statistics
# -------------------------------
print("Skewness of placement_exam_marks:", df['placement_exam_marks'].skew())
print("Mean value of cgpa:", df['cgpa'].mean())
print("Std value of cgpa:", df['cgpa'].std())
print("Min value of cgpa:", df['cgpa'].min())
print("Max value of cgpa:", df['cgpa'].max())

# -------------------------------
# Finding Boundary Values
# -------------------------------
upper_limit = df['cgpa'].mean() + 3*df['cgpa'].std()
lower_limit = df['cgpa'].mean() - 3*df['cgpa'].std()
print("Highest allowed:", upper_limit)
print("Lowest allowed:", lower_limit)

# -------------------------------
# Finding Outliers
# -------------------------------
outliers = df[(df['cgpa'] > upper_limit) | (df['cgpa'] < lower_limit)]
print("Outliers detected:\n", outliers)

# -------------------------------
# Trimming Method
# -------------------------------
new_df = df[(df['cgpa'] < upper_limit) & (df['cgpa'] > lower_limit)]
print("After trimming shape:", new_df.shape)

# -------------------------------
# Z-Score Calculation
# -------------------------------
df['cgpa_zscore'] = (df['cgpa'] - df['cgpa'].mean()) / df['cgpa'].std()
print(df.head())

# Outliers by Z-Score
print(df[df['cgpa_zscore'] > 3])
print(df[df['cgpa_zscore'] < -3])

# Trimming using Z-Score
new_df = df[(df['cgpa_zscore'] < 3) & (df['cgpa_zscore'] > -3)]
print("After Z-score trimming shape:", new_df.shape)

# -------------------------------
# Capping Method
# -------------------------------
df['cgpa'] = np.where(
    df['cgpa'] > upper_limit, upper_limit,
    np.where(
        df['cgpa'] < lower_limit, lower_limit,
        df['cgpa']
    )
)

print("Shape after capping:", df.shape)
print(df['cgpa'].describe())


