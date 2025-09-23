# ============================================================
#           PERCENTILE METHOD (WINSORIZATION)
# ============================================================

# -------------------------------
# What is Winsorization?
# -------------------------------
# - Winsorization is a technique where extreme values
#   (outliers) are capped to a certain percentile range.
# - Example:
#   Replace values above the 99th percentile with the 99th percentile.
#   Replace values below the 1st percentile with the 1st percentile.

# -------------------------------
# Benefits of Winsorization
# -------------------------------
# 1. Keeps dataset size intact (no row removal).
# 2. Reduces the effect of extreme values on mean and std.
# 3. Useful for highly skewed distributions.
# 4. Works well for models sensitive to extreme values 
#    (Linear Regression, Logistic Regression, Deep Learning).

# -------------------------------
# When to Use This Technique?
# -------------------------------
# - When you don’t want to lose data (unlike trimming).
# - When the dataset is large, and trimming may cause 
#   information loss.
# - When outliers are due to rare but valid observations.

# -------------------------------
# What is Percentile?
# -------------------------------
# - A percentile indicates the value below which a given 
#   percentage of data falls.
#   Example: 
#   - 25th percentile (Q1): 25% of data lies below it.
#   - 99th percentile: 99% of data lies below it.
#
# - For Winsorization:
#   - We usually cap values at the 1st and 99th percentile.

# ============================================================
#                IMPLEMENTATION IN PYTHON
# ============================================================

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('weight-height.csv')
print(df.head())
print("Shape of dataset:", df.shape)

# -------------------------------
# Initial Summary
# -------------------------------
print(df['Height'].describe())

# Distribution and Boxplot
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
sns.distplot(df['Height'])
plt.title("Original Height Distribution")

plt.subplot(1,2,2)
sns.boxplot(df['Height'])
plt.title("Original Height Boxplot")

plt.show()

# -------------------------------
# Finding Percentile Limits
# -------------------------------
upper_limit = df['Height'].quantile(0.99)   # 99th percentile
lower_limit = df['Height'].quantile(0.01)   # 1st percentile

print("Upper limit (99th percentile):", upper_limit)
print("Lower limit (1st percentile):", lower_limit)

# -------------------------------
# Trimming Approach
# -------------------------------
new_df = df[(df['Height'] <= upper_limit) & (df['Height'] >= lower_limit)]
print("After trimming shape:", new_df.shape)
print(new_df['Height'].describe())

# Visualizing After Trimming
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
sns.distplot(new_df['Height'])
plt.title("After Trimming Distribution")

plt.subplot(1,2,2)
sns.boxplot(new_df['Height'])
plt.title("After Trimming Boxplot")

plt.show()

# -------------------------------
# Winsorization (Capping Approach)
# -------------------------------
df['Height'] = np.where(
    df['Height'] >= upper_limit, upper_limit,
    np.where(
        df['Height'] <= lower_limit, lower_limit,
        df['Height']
    )
)

print("After Winsorization shape:", df.shape)
print(df['Height'].describe())

# Visualizing After Winsorization
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
sns.distplot(df['Height'])
plt.title("After Winsorization Distribution")

plt.subplot(1,2,2)
sns.boxplot(df['Height'])
plt.title("After Winsorization Boxplot")

plt.show()


