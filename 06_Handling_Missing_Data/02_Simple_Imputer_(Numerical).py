# ==========================================================
# NUMERICAL DATA MISSING VALUE HANDLING
# ==========================================================

# Missing values in numerical features can be handled in multiple ways.
# Two broad categories are:
# 1. Univariate Imputation  -> handle each variable independently
# 2. Multivariate Imputation -> use other variables to estimate missing ones
#
# Common techniques:
# - Mean/Median Imputation
# - Arbitrary Value Imputation
# - End of Distribution Imputation
# - Random Sample Imputation
#
# Below we implement and compare different techniques.
# ==========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# ==========================================================
# LOAD DATA
# ==========================================================
df = pd.read_csv("titanic_toy.csv")

print(df.info())
print(df.isnull().mean())

# Features / Target split
X = df.drop(columns=["Survived"])
y = df["Survived"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2
)

# ==========================================================
# 1. MEAN & MEDIAN IMPUTATION
# ==========================================================
#  Best for: quick, simple fixes in small projects
#  Disadvantages: distorts variance, weakens correlation, ignores outliers

# Calculate mean and median
mean_age = X_train["Age"].mean()
median_age = X_train["Age"].median()
mean_fare = X_train["Fare"].mean()
median_fare = X_train["Fare"].median()

# Apply imputation manually
X_train["Age_median"] = X_train["Age"].fillna(median_age)
X_train["Age_mean"] = X_train["Age"].fillna(mean_age)
X_train["Fare_median"] = X_train["Fare"].fillna(median_fare)
X_train["Fare_mean"] = X_train["Fare"].fillna(mean_fare)

# Variance check
print("Original Age variance: ", X_train["Age"].var())
print("After Median Age: ", X_train["Age_median"].var())
print("After Mean Age: ", X_train["Age_mean"].var())
print("Original Fare variance: ", X_train["Fare"].var())
print("After Median Fare: ", X_train["Fare_median"].var())
print("After Mean Fare: ", X_train["Fare_mean"].var())

# KDE plots for Age
fig = plt.figure()
ax = fig.add_subplot(111)
X_train["Age"].plot(kind="kde", ax=ax)
X_train["Age_median"].plot(kind="kde", ax=ax, color="red")
X_train["Age_mean"].plot(kind="kde", ax=ax, color="green")
ax.legend(["Original", "Median", "Mean"])

# KDE plots for Fare
fig = plt.figure()
ax = fig.add_subplot(111)
X_train["Fare"].plot(kind="kde", ax=ax)
X_train["Fare_median"].plot(kind="kde", ax=ax, color="red")
X_train["Fare_mean"].plot(kind="kde", ax=ax, color="green")
ax.legend(["Original", "Median", "Mean"])

# Boxplots
X_train[["Age", "Age_median", "Age_mean"]].boxplot()
X_train[["Fare", "Fare_median", "Fare_mean"]].boxplot()

# Sklearn implementation
imputer1 = SimpleImputer(strategy="median")
imputer2 = SimpleImputer(strategy="mean")

trf = ColumnTransformer(
    [
        ("imputer1", imputer1, ["Age"]),
        ("imputer2", imputer2, ["Fare"]),
    ],
    remainder="passthrough",
)
trf.fit(X_train)
print("Median Age used:", trf.named_transformers_["imputer1"].statistics_)
print("Mean Fare used:", trf.named_transformers_["imputer2"].statistics_)

X_train = trf.transform(X_train)
X_test = trf.transform(X_test)

# ==========================================================
# 2. ARBITRARY VALUE IMPUTATION
# ==========================================================
#  Useful when we want to flag missingness with extreme/unrealistic values
#  Helps ML models detect "was missing" pattern
#  Changes variance drastically, may create outliers

# Apply arbitrary values manually
X = df.drop(columns=["Survived"])
y = df["Survived"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2
)

X_train["Age_99"] = X_train["Age"].fillna(99)
X_train["Age_minus1"] = X_train["Age"].fillna(-1)
X_train["Fare_999"] = X_train["Fare"].fillna(999)
X_train["Fare_minus1"] = X_train["Fare"].fillna(-1)

# Variance check
print("Original Age variance: ", X_train["Age"].var())
print("After Age=99: ", X_train["Age_99"].var())
print("After Age=-1: ", X_train["Age_minus1"].var())
print("Original Fare variance: ", X_train["Fare"].var())
print("After Fare=999: ", X_train["Fare_999"].var())
print("After Fare=-1: ", X_train["Fare_minus1"].var())

# KDE plots
fig = plt.figure()
ax = fig.add_subplot(111)
X_train["Age"].plot(kind="kde", ax=ax)
X_train["Age_99"].plot(kind="kde", ax=ax, color="red")
X_train["Age_minus1"].plot(kind="kde", ax=ax, color="green")
ax.legend(["Original", "Age=99", "Age=-1"])

fig = plt.figure()
ax = fig.add_subplot(111)
X_train["Fare"].plot(kind="kde", ax=ax)
X_train["Fare_999"].plot(kind="kde", ax=ax, color="red")
X_train["Fare_minus1"].plot(kind="kde", ax=ax, color="green")
ax.legend(["Original", "Fare=999", "Fare=-1"])

# Sklearn implementation
imputer1 = SimpleImputer(strategy="constant", fill_value=99)
imputer2 = SimpleImputer(strategy="constant", fill_value=999)

trf = ColumnTransformer(
    [
        ("imputer1", imputer1, ["Age"]),
        ("imputer2", imputer2, ["Fare"]),
    ],
    remainder="passthrough",
)
trf.fit(X_train)
print("Constant Age value:", trf.named_transformers_["imputer1"].statistics_)
print("Constant Fare value:", trf.named_transformers_["imputer2"].statistics_)

X_train = trf.transform(X_train)
X_test = trf.transform(X_test)

# ==========================================================
# 3. END OF DISTRIBUTION IMPUTATION
# ==========================================================
# Replace missing values with extreme values at the tail of the distribution
#  Preserves distribution better than arbitrary constant
#  Still allows ML models to detect missingness
#  Can still distort variance slightly

# Example:
# Age -> fill with (mean + 3*std)
# Fare -> fill with (mean + 3*std)

# ==========================================================
# 4. RANDOM SAMPLE IMPUTATION
# ==========================================================
# Replace missing values with a random sample from the existing values
#  Keeps original distribution intact
#  Variance preserved
#  Randomness means reproducibility issues unless random_state is fixed

# ==========================================================
# SUMMARY
# - Mean/Median: Simple, quick, but variance/correlation distortion
# - Arbitrary: Useful for missingness flagging, but creates outliers
# - End of Distribution: Good for highlighting missingness while staying realistic
# - Random Sample: Best for keeping original distribution shape
# ==========================================================
