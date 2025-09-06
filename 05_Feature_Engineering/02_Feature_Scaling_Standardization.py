# ================================================
# Feature Scaling – Standardization 
# ================================================

# What is Feature Scaling?
# ------------------------
# Feature scaling is a technique to standardize or normalize 
# the independent features of data into a fixed range.
# It ensures that features with larger magnitudes do not dominate those with smaller magnitudes.

# Why do we need Feature Scaling?
# -------------------------------
# - Many ML algorithms (KNN, K-Means, PCA, Logistic Regression, Neural Networks, Gradient Descent)
#   are sensitive to the scale of features.
# - Without scaling, models may give biased results.

# Types of Feature Scaling:
# -------------------------
# 1. Standardization (Z-score Normalization)
# 2. Normalization (Min-Max Scaler, Robust Scaler)

# Formula for Standardization:
# ----------------------------
#   z = (x - mean) / standard_deviation

# ------------------------------------------------
# Example: Age & Salary dataset (Social_Network_Ads.csv)
# ------------------------------------------------

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('Social_Network_Ads.csv')
df = df.iloc[:, 2:]  # take only Age, EstimatedSalary, Purchased

print(df.sample(5))

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    df.drop('Purchased', axis=1), 
    df['Purchased'], 
    test_size=0.3, 
    random_state=0
)

print(X_train.shape, X_test.shape)

# StandardScaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# Fit on training data
scaler.fit(X_train)

# Transform both train and test
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame for easy visualization
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

print("Scaler mean values:", scaler.mean_)
print(X_train.head())
print(X_train_scaled.head())

# ------------------------------------------------
# Effect of Scaling (Scatter Plot Before vs After)
# ------------------------------------------------
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))

# Before scaling
ax1.scatter(X_train['Age'], X_train['EstimatedSalary'])
ax1.set_title("Before Scaling")
ax1.set_xlabel("Age")
ax1.set_ylabel("Estimated Salary")

# After scaling
ax2.scatter(X_train_scaled['Age'], X_train_scaled['EstimatedSalary'], color='red')
ax2.set_title("After Scaling")
ax2.set_xlabel("Age (scaled)")
ax2.set_ylabel("Estimated Salary (scaled)")

plt.show()

# ------------------------------------------------
# Distribution Comparison Before vs After Scaling
# ------------------------------------------------

# Age distribution
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))
ax1.set_title('Age Distribution Before Scaling')
sns.kdeplot(X_train['Age'], ax=ax1)

ax2.set_title('Age Distribution After Scaling')
sns.kdeplot(X_train_scaled['Age'], ax=ax2)
plt.show()

# Salary distribution
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))
ax1.set_title('Salary Distribution Before Scaling')
sns.kdeplot(X_train['EstimatedSalary'], ax=ax1)

ax2.set_title('Salary Distribution After Scaling')
sns.kdeplot(X_train_scaled['EstimatedSalary'], ax=ax2)
plt.show()

# ------------------------------------------------
# Why Scaling is Important? (Model Comparison)
# ------------------------------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Logistic Regression without scaling
model1 = LogisticRegression()
model1.fit(X_train, y_train)
y_pred1 = model1.predict(X_test)
print("Accuracy without Scaling:", accuracy_score(y_test, y_pred1))

# Logistic Regression with scaling
model2 = LogisticRegression()
model2.fit(X_train_scaled, y_train)
y_pred2 = model2.predict(X_test_scaled)
print("Accuracy with Scaling:", accuracy_score(y_test, y_pred2))

# ------------------------------------------------
# Effect of Outliers
# ------------------------------------------------
# Adding outliers (extreme Age and Salary values)
outlier_data = pd.DataFrame({
    "Age": [4, 90, 96],
    "EstimatedSalary": [20000, 2500000, 4000000]
})

X_outlier = pd.concat([X_train, outlier_data], ignore_index=True)

# Apply scaling again
X_outlier_scaled = pd.DataFrame(scaler.fit_transform(X_outlier), columns=X_outlier.columns)

# Compare distribution after adding outliers
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))
ax1.set_title("Age with Outliers (Before Scaling)")
sns.kdeplot(X_outlier['Age'], ax=ax1)

ax2.set_title("Age with Outliers (After Scaling)")
sns.kdeplot(X_outlier_scaled['Age'], ax=ax2)
plt.show()

# ------------------------------------------------
# When to use Standardization?
# ------------------------------------------------
# Algorithms that assume Gaussian distribution:
# - K-Means Clustering
# - KNN (K-Nearest Neighbors)
# - PCA (Principal Component Analysis)
# - Logistic Regression, Linear Regression
# - Neural Networks
# - Gradient Descent based models
