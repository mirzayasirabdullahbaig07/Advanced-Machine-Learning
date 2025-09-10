# ==============================================================
# Mathematical Transformers & FunctionTransformer in Python
# ==============================================================

# Mathematical transformations are used to make skewed data
# closer to a normal distribution (bell curve).
# Why normal distribution? Many ML models (like Linear Regression,
# Logistic Regression) assume that the data follows normal distribution
# for better performance and valid statistical inference.

# --------------------------------------------------------------
# Featured Mathematical Transformers:
# --------------------------------------------------------------
# 1. Log Transform        -> reduces right skew, stabilizes variance
# 2. Reciprocal Transform -> flips values (1/x), useful for large outliers
# 3. Power Transform      -> raise values to a power (e.g., square, sqrt)
# 4. Box-Cox Transform    -> automatic power transformation (only positive values)
# 5. Yeo-Johnson Transform-> similar to Box-Cox but works with negative values too
#
# All these are used to reduce skewness and make data closer to normal.

# --------------------------------------------------------------
# How to check if data is normal?
# --------------------------------------------------------------
# - sns.displot()        -> visualize probability density
# - df.skew()            -> check skewness value
# - stats.probplot()     -> QQ plot (more reliable for normality check)
#
# What is QQ plot? -> Compares distribution of data to a normal distribution.
# If points lie close to the 45° line → data is normally distributed.

# --------------------------------------------------------------
# Import Libraries
# --------------------------------------------------------------
import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer

# --------------------------------------------------------------
# Load Dataset
# --------------------------------------------------------------
df = pd.read_csv('train.csv', usecols=['Age', 'Fare', 'Survived'])

# Fill missing Age values with mean
df['Age'].fillna(df['Age'].mean(), inplace=True)

# Features (X) and Target (y)
X = df[['Age', 'Fare']]
y = df['Survived']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------------------------------------------
# Check Normality with Distribution & QQ Plots
# --------------------------------------------------------------
plt.figure(figsize=(14, 4))

# Distribution of Age
sns.displot(X_test['Age'])
plt.title('Age Distribution (PDF)')

# QQ Plot for Age
stats.probplot(X_train['Age'], dist='norm', plot=plt)
plt.title('Age QQ Plot')
plt.show()

plt.figure(figsize=(14, 4))

# Distribution of Fare
sns.displot(X_test['Fare'])
plt.title('Fare Distribution (PDF)')

# QQ Plot for Fare
stats.probplot(X_train['Fare'], dist='norm', plot=plt)
plt.title('Fare QQ Plot')
plt.show()

# --------------------------------------------------------------
# Train models WITHOUT transformation
# --------------------------------------------------------------
clf = LogisticRegression()
clf2 = DecisionTreeClassifier()

clf.fit(X_train, y_train)
clf2.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_pred1 = clf2.predict(X_test)

print("Accuracy Logistic Regression (Raw Data):", accuracy_score(y_test, y_pred))
print("Accuracy Decision Tree (Raw Data):", accuracy_score(y_test, y_pred1))

# --------------------------------------------------------------
# Apply Log Transformation using FunctionTransformer
# --------------------------------------------------------------
trf = FunctionTransformer(func=np.log1p)  # log1p = log(1+x), safe for 0 values

# Transform train and test
X_train_transformed = trf.fit_transform(X_train)
X_test_transformed = trf.transform(X_test)

# Train again with transformed data
clf = LogisticRegression()
clf2 = DecisionTreeClassifier()

clf.fit(X_train_transformed, y_train)
clf2.fit(X_train_transformed, y_train)

y_pred = clf.predict(X_test_transformed)
y_pred1 = clf2.predict(X_test_transformed)

print("Accuracy Logistic Regression (Log Transform):", accuracy_score(y_test, y_pred))
print("Accuracy Decision Tree (Log Transform):", accuracy_score(y_test, y_pred1))

# --------------------------------------------------------------
# Cross-validation with transformed data
# --------------------------------------------------------------
X_transformed = trf.fit_transform(X)

print("CV Accuracy Logistic Regression:", np.mean(cross_val_score(clf, X_transformed, y, cv=10)))
print("CV Accuracy Decision Tree:", np.mean(cross_val_score(clf2, X_transformed, y, cv=10)))

# --------------------------------------------------------------
# Compare BEFORE vs AFTER Transformation (QQ Plot for Fare)
# --------------------------------------------------------------
plt.figure(figsize=(14, 4))

# Before log
stats.probplot(X_train['Fare'], dist='norm', plot=plt)
plt.title('Fare BEFORE log transform')

# After log
stats.probplot(X_train_transformed[:, 1], dist='norm', plot=plt)
plt.title('Fare AFTER log transform')
plt.show()

# --------------------------------------------------------------
# Apply log transformation ONLY to 'Fare' column
# --------------------------------------------------------------
trf2 = ColumnTransformer(
    [('log', FunctionTransformer(np.log1p), ['Fare'])],
    remainder='passthrough'
)

X_train_transformed2 = trf2.fit_transform(X_train)
X_test_transformed2 = trf2.transform(X_test)

clf = LogisticRegression()
clf2 = DecisionTreeClassifier()

clf.fit(X_train_transformed2, y_train)
clf2.fit(X_train_transformed2, y_train)

y_pred = clf.predict(X_test_transformed2)
y_pred1 = clf2.predict(X_test_transformed2)

print("Accuracy Logistic Regression (Fare Log Only):", accuracy_score(y_test, y_pred))
print("Accuracy Decision Tree (Fare Log Only):", accuracy_score(y_test, y_pred1))

# --------------------------------------------------------------
# General Function to Apply Any Transformation
# --------------------------------------------------------------
def apply_transform(transform):
    # Take features & target
    X = df[['Age', 'Fare']]
    y = df['Survived']

    # Apply transformation only on Fare
    trf = ColumnTransformer(
        [('transform', FunctionTransformer(transform), ['Fare'])],
        remainder='passthrough'
    )

    # Transform data
    X_trans = trf.fit_transform(X)

    # Logistic Regression with CV
    clf = LogisticRegression()
    acc = np.mean(cross_val_score(clf, X_trans, y, scoring='accuracy', cv=10))
    print(f"Accuracy with {transform.__name__ if hasattr(transform, '__name__') else 'Custom'}:", acc)

    # QQ Plot before and after
    plt.figure(figsize=(14, 4))
    sm.ProbPlot(X['Fare'], dist="norm").qqplot(line='45', ax=plt.gca())
    plt.title('Fare BEFORE Transform')

    plt.figure(figsize=(14, 4))
    sm.ProbPlot(X_trans[:, 0], dist="norm").qqplot(line='45', ax=plt.gca())
    plt.title('Fare AFTER Transform')
    plt.show()

# --------------------------------------------------------------
# Test Different Transformations
# --------------------------------------------------------------
apply_transform(lambda x: x)       # No transform
apply_transform(np.square)         # Square transform
apply_transform(np.sqrt)           # Square root transform
apply_transform(np.reciprocal)     # Reciprocal transform
apply_transform(np.sin)            # Example of non-standard transform
