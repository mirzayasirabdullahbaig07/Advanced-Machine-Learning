# ===============================
# Feature Scaling – Normalization
# ===============================

# What is normalization?
# Normalization is a data preprocessing technique used to rescale the values of numerical 
# features into a specific range (commonly [0,1]).
# Goal: Change the scale of features without distorting differences between them.

# Why we use normalization?
# - Machine Learning models like KNN, K-Means, Neural Networks, Gradient Descent based models
#   are sensitive to the scale of features.
# - Helps features contribute equally to the model.

# -----------------------
# Types of Normalization:
# -----------------------
# 1. Min-Max Scaling
# 2. Mean Normalization
# 3. Max Absolute Scaling
# 4. Robust Scaling

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------
# Example Dataset (Wine)
# -----------------------
df = pd.read_csv('wine_data.csv', header=None, usecols=[0,1,2])
df.columns = ['Class label', 'Alcohol', 'Malic acid']

# Plot KDE before scaling
sns.kdeplot(df['Alcohol'], label="Alcohol")
sns.kdeplot(df['Malic acid'], label="Malic acid")
plt.legend()
plt.title("Distribution Before Scaling")
plt.show()

# Scatterplot before scaling
color_dict = {1: 'red', 2: 'blue', 3: 'green'}
sns.scatterplot(x=df['Alcohol'], y=df['Malic acid'], hue=df['Class label'], palette=color_dict)
plt.title("Before Scaling (Scatter)")
plt.show()


# -----------------------
# Train-Test Split
# -----------------------
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    df.drop('Class label', axis=1), 
    df['Class label'], 
    test_size=0.3, 
    random_state=0
)


# =====================
# Min-Max Scaling
# =====================
# Formula: 
#     X_scaled = (X - X_min) / (X_max - X_min)
# - Maps data into the range [0, 1].
# - Sensitive to outliers.
# - Common in Neural Networks, KNN, K-Means.

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(X_train)   # learn min & max from training set

X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)


# Compare distributions before & after scaling
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))

# Before scaling
ax1.scatter(X_train['Alcohol'], X_train['Malic acid'], c=y_train)
ax1.set_title("Before Scaling")

# After scaling
ax2.scatter(X_train_scaled['Alcohol'], X_train_scaled['Malic acid'], c=y_train)
ax2.set_title("After Min-Max Scaling")

plt.show()


# KDE comparison before vs after scaling
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))
ax1.set_title('Before Scaling')
sns.kdeplot(X_train['Alcohol'], ax=ax1)
sns.kdeplot(X_train['Malic acid'], ax=ax1)

ax2.set_title('After Min-Max Scaling')
sns.kdeplot(X_train_scaled['Alcohol'], ax=ax2)
sns.kdeplot(X_train_scaled['Malic acid'], ax=ax2)
plt.show()


# =========================
# Other Normalization Types
# =========================

# 1. Mean Normalization:
# Formula: (X - mean(X)) / (X_max - X_min)
# - Centers values around 0, but still depends on min & max.
# - Rarely used.

# 2. MaxAbs Scaling:
# Formula: X_scaled = X / max(abs(X))
# - Scales values to [-1, 1].
# - Useful when data is already centered at 0.
# - Rarely used.

# 3. Robust Scaling:
# Formula: (X - Median) / IQR
# - Uses Median and Interquartile Range instead of mean & std.
# - Not affected by outliers.
# - Often used in real-world datasets with outliers.


# =========================
# Standardization vs Normalization
# =========================
# Standardization (Z-score Normalization):
# Formula: (X - mean) / std
# - Mean = 0, Std = 1
# - Useful when data follows Gaussian distribution
# - Models: Logistic Regression, Linear Regression, PCA, SVM, Gradient Descent

# Normalization (Min-Max Scaling):
# Formula: (X - min) / (max - min)
# - Data scaled into [0,1]
# - Useful when no assumption about distribution
# - Models: KNN, K-Means, Neural Networks


# =========================
# When to Use?
# =========================
# - Use Standardization: When algorithm assumes normal distribution or distance-based (Logistic Regression, PCA, SVM, Linear Models).
# - Use Normalization: When features have very different ranges & for distance-based models (KNN, K-Means, Neural Networks).
# - Use Robust Scaling: When dataset contains outliers.

