# ======================================================
# Missing Value Handling Techniques in Python
# ======================================================
# We will cover 3 key imputation methods:
# 1. Random Sample Imputation
# 2. Missing Indicator
# 3. Automatic Imputation (SimpleImputer)
# Plus: Full pipeline with GridSearch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn import set_config


# =====================================================
# 🔹 1. Random Sample Imputation
# =====================================================
# Definition:
# Replace missing values by randomly sampling from the existing (non-missing)
# values of the same variable.
#
# Why we use it:
# - Preserves the original distribution of the variable.
#
# Disadvantages:
# - Adds randomness (different results each run).
# - Risk of data leakage if test set is imputed using its own distribution.
#
# Example (Titanic):
# - Missing Age values → randomly pick ages from known ages (24, 32, 45).
#
# Benefit:
# - Keeps variability in the data (unlike mean/median which flattens it).

print("\n=== Random Sample Imputation ===")

df = pd.read_csv('train.csv', usecols=['Age', 'Fare', 'Survived'])
X = df.drop(columns=['Survived'])
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Create copy columns
X_train['Age_imputed'] = X_train['Age']
X_test['Age_imputed'] = X_test['Age']

# Impute missing values with random samples from training set
X_train.loc[X_train['Age_imputed'].isnull(), 'Age_imputed'] = (
    X_train['Age'].dropna().sample(X_train['Age'].isnull().sum()).values
)
X_test.loc[X_test['Age_imputed'].isnull(), 'Age_imputed'] = (
    X_train['Age'].dropna().sample(X_test['Age'].isnull().sum()).values
)

# Compare distributions
sns.distplot(X_train['Age'], label='Original', hist=False)
sns.distplot(X_train['Age_imputed'], label='Imputed', hist=False)
plt.legend()
plt.show()

print("Original variance:", X_train['Age'].var())
print("Variance after imputation:", X_train['Age_imputed'].var())


# =====================================================
# 🔹 2. Missing Indicator Technique
# =====================================================
# Definition:
# Create a new binary variable (0/1) to indicate whether the value was missing.
#
# Why we use it:
# - Sometimes the fact that a value is missing carries information.
#
# Disadvantages:
# - Doesn’t fill missing values directly (needs another imputer).
# - May increase dimensionality a lot if many features are missing.
#
# Example:
# - Add new column Age_NA = 1 (if missing), 0 (if not).
#
# Benefit:
# - Preserves "missingness pattern" which can improve model performance.

print("\n=== Missing Indicator Technique ===")

df = pd.read_csv('train.csv', usecols=['Age', 'Fare', 'Survived'])
X = df.drop(columns=['Survived'])
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Missing indicator
mi = MissingIndicator()
mi.fit(X_train)

X_train_missing = mi.transform(X_train)
X_test_missing = mi.transform(X_test)

# Add indicator to dataset
X_train['Age_NA'] = X_train_missing
X_test['Age_NA'] = X_test_missing

# Simple imputation + Logistic Regression
si = SimpleImputer()
X_train_trf = si.fit_transform(X_train)
X_test_trf = si.transform(X_test)

clf = LogisticRegression()
clf.fit(X_train_trf, y_train)
y_pred = clf.predict(X_test_trf)
print("Accuracy with Missing Indicator:", accuracy_score(y_test, y_pred))


# =====================================================
# 🔹 3. Automatic Imputation (SimpleImputer in Sklearn)
# =====================================================
# Definition:
# Automatically fills missing values using a pre-defined strategy
# (mean, median, most_frequent, constant).
#
# Why we use it:
# - Simple, fast, pipeline-friendly.
#
# Disadvantages:
# - Can distort variance (e.g., mean reduces spread).
# - Ignores correlation with other features.
#
# Example:
# - Fill Age with mean, Embarked with most frequent.
#
# Benefit:
# - Fast, reproducible, works with pipelines and grid search.

print("\n=== SimpleImputer with Indicator Column ===")

si = SimpleImputer(add_indicator=True)
X_train_trf2 = si.fit_transform(X_train)
X_test_trf2 = si.transform(X_test)

clf = LogisticRegression()
clf.fit(X_train_trf2, y_train)
y_pred = clf.predict(X_test_trf2)
print("Accuracy with SimpleImputer(add_indicator=True):", accuracy_score(y_test, y_pred))


# =====================================================
# 🔹 4. Full Pipeline with GridSearch
# =====================================================
# Here we build a complete pipeline with:
# - Median/Mean imputation for numeric data
# - Most frequent/constant imputation for categorical data
# - Scaling + OneHotEncoding
# - Logistic Regression as classifier
# - GridSearchCV for hyperparameter tuning

print("\n=== Full Pipeline with GridSearch ===")

df = pd.read_csv('train.csv')
df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)

X = df.drop(columns=['Survived'])
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Define transformers
numerical_features = ['Age', 'Fare']
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_features = ['Embarked', 'Sex']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(handle_unknown='ignore'))
])

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Pipeline with classifier
clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])

# GridSearch for best params
param_grid = {
    'preprocessor__num__imputer__strategy': ['mean', 'median'],
    'preprocessor__cat__imputer__strategy': ['most_frequent', 'constant'],
    'classifier__C': [0.1, 1.0, 10, 100]
}

grid_search = GridSearchCV(clf, param_grid, cv=10)
grid_search.fit(X_train, y_train)

print("Best params:", grid_search.best_params_)
print(f"Internal CV score: {grid_search.best_score_:.3f}")

# Show CV results
cv_results = pd.DataFrame(grid_search.cv_results_)
cv_results = cv_results.sort_values("mean_test_score", ascending=False)
print(cv_results[['param_classifier__C',
                  'param_preprocessor__cat__imputer__strategy',
                  'param_preprocessor__num__imputer__strategy',
                  'mean_test_score']].head())
