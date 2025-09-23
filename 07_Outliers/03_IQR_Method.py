# ============================================================
#                IQR METHOD FOR OUTLIERS
# ============================================================

# -------------------------------
# When to Use IQR Method?
# -------------------------------
# - IQR method is very effective when data is skewed 
#   (not normally distributed).
# - It relies on percentiles rather than mean & std.

# -------------------------------
# What is a Box Plot?
# -------------------------------
# - A box plot (or whisker plot) visualizes the 
#   distribution of data using:
#   Q1 (25th percentile), Median, Q3 (75th percentile).
# - The "whiskers" extend to 1.5 * IQR.
# - Points outside whiskers are considered outliers.

# -------------------------------
# What is Percentile?
# -------------------------------
# - Percentile is a value below which a given 
#   percentage of observations fall.
#   Example: 25th percentile (Q1) = value below 
#   which 25% of data lies.

# -------------------------------
# What is IQR?
# -------------------------------
# - IQR (Interquartile Range) = Q3 - Q1
# - Outlier Boundaries:
#   Lower Bound = Q1 - 1.5 * IQR
#   Upper Bound = Q3 + 1.5 * IQR
# - Any value outside this range is considered an outlier.

# ============================================================
#                  IMPLEMENTATION IN PYTHON
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('placement.csv')
print(df.head())

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
# Descriptive Statistics
# -------------------------------
print(df['placement_exam_marks'].describe())

# -------------------------------
# Boxplot for Outlier Visualization
# -------------------------------
sns.boxplot(df['placement_exam_marks'])
plt.title("Boxplot of Placement Exam Marks")
plt.show()

# -------------------------------
# Finding the IQR
# -------------------------------
Q1 = df['placement_exam_marks'].quantile(0.25)
Q3 = df['placement_exam_marks'].quantile(0.75)
IQR = Q3 - Q1

print("Q1 (25th percentile):", Q1)
print("Q3 (75th percentile):", Q3)
print("IQR:", IQR)

# Boundary values
upper_limit = Q3 + 1.5 * IQR
lower_limit = Q1 - 1.5 * IQR
print("Upper limit:", upper_limit)
print("Lower limit:", lower_limit)

# -------------------------------
# Finding Outliers
# -------------------------------
print("Outliers above upper limit:\n", df[df['placement_exam_marks'] > upper_limit])
print("Outliers below lower limit:\n", df[df['placement_exam_marks'] < lower_limit])

# -------------------------------
# Trimming Method
# -------------------------------
new_df = df[(df['placement_exam_marks'] < upper_limit) & 
            (df['placement_exam_marks'] > lower_limit)]
print("After trimming shape:", new_df.shape)

# Comparing Before vs After Trimming
plt.figure(figsize=(16,8))

plt.subplot(2,2,1)
sns.distplot(df['placement_exam_marks'])
plt.title("Original Distribution")

plt.subplot(2,2,2)
sns.boxplot(df['placement_exam_marks'])
plt.title("Original Boxplot")

plt.subplot(2,2,3)
sns.distplot(new_df['placement_exam_marks'])
plt.title("After Trimming Distribution")

plt.subplot(2,2,4)
sns.boxplot(new_df['placement_exam_marks'])
plt.title("After Trimming Boxplot")

plt.show()

# -------------------------------
# Capping Method
# -------------------------------
new_df_cap = df.copy()

new_df_cap['placement_exam_marks'] = np.where(
    new_df_cap['placement_exam_marks'] > upper_limit,
    upper_limit,
    np.where(
        new_df_cap['placement_exam_marks'] < lower_limit,
        lower_limit,
        new_df_cap['placement_exam_marks']
    )
)

print("After capping shape:", new_df_cap.shape)

# Comparing Before vs After Capping
plt.figure(figsize=(16,8))

plt.subplot(2,2,1)
sns.distplot(df['placement_exam_marks'])
plt.title("Original Distribution")

plt.subplot(2,2,2)
sns.boxplot(df['placement_exam_marks'])
plt.title("Original Boxplot")

plt.subplot(2,2,3)
sns.distplot(new_df_cap['placement_exam_marks'])
plt.title("After Capping Distribution")

plt.subplot(2,2,4)
sns.boxplot(new_df_cap['placement_exam_marks'])
plt.title("After Capping Boxplot")
plt.show()

