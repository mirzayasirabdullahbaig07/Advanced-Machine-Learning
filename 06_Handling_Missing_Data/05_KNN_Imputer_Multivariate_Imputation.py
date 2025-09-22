# ----------------------------------------------------------
# KNN Imputer & Multivariate Imputation - Complete Example
# Dataset: Titanic (Age, Pclass, Fare, Survived)
# ----------------------------------------------------------

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ----------------------------------------------------------
# Load dataset (we use only 4 columns for simplicity)
# Age and Fare may contain missing values
# Pclass is categorical (1st, 2nd, 3rd class)
# Survived is our target variable
# ----------------------------------------------------------
df = pd.read_csv('train.csv')[['Age', 'Pclass', 'Fare', 'Survived']]
print(df.head())

# Check percentage of missing values
print(df.isnull().mean() * 100)

# Split into features (X) and target (y)
X = df.drop(columns=['Survived'])
y = df['Survived']

# Train-test split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2
)

# ----------------------------------------------------------
# 1. KNN Imputer
# ----------------------------------------------------------
# KNNImputer works by finding 'k' nearest rows (neighbors)
# based on feature similarity and imputing missing values
# from them. Here, we set k=3 neighbors and use distance weights.
# This is an example of MULTIVARIATE IMPUTATION because it 
# uses multiple features (Age, Pclass, Fare) to impute missing values.
# ----------------------------------------------------------
knn = KNNImputer(n_neighbors=3, weights='distance')

# Fit & transform training set, transform test set
X_train_trf = knn.fit_transform(X_train)
X_test_trf = knn.transform(X_test)

# Train Logistic Regression model on imputed data
lr = LogisticRegression(max_iter=200)
lr.fit(X_train_trf, y_train)

# Predict on test set
y_pred = lr.predict(X_test_trf)

# Evaluate accuracy
acc_knn = accuracy_score(y_test, y_pred)
print("Accuracy with KNN Imputer:", acc_knn)

# ----------------------------------------------------------
# 2. Simple Imputer (Mean Strategy)
# ----------------------------------------------------------
# SimpleImputer fills missing values using column statistics.
# Here we use the default strategy = 'mean'.
# This is a UNIVARIATE IMPUTATION method, because it imputes
# each column independently, ignoring relationships between features.
# ----------------------------------------------------------
si = SimpleImputer()

# Fit & transform training set, transform test set
X_train_trf2 = si.fit_transform(X_train)
X_test_trf2 = si.transform(X_test)

# Train Logistic Regression model
lr = LogisticRegression(max_iter=200)
lr.fit(X_train_trf2, y_train)

# Predict on test set
y_pred2 = lr.predict(X_test_trf2)

# Evaluate accuracy
acc_simple = accuracy_score(y_test, y_pred2)
print("Accuracy with Simple Imputer (Mean):", acc_simple)

# ----------------------------------------------------------
# Comparison & Explanation
# ----------------------------------------------------------
# KNN Imputer:
# - Considers feature similarity when imputing values
# - More accurate imputation (uses relationships between Age, Fare, Pclass)
# - Example: If Age is missing, it will use similar passengers (same class, fare range) 
#   to estimate a realistic value.
#
# Simple Imputer (Mean):
# - Fills missing values with global mean
# - Ignores relationships between features
# - Example: If Age is missing, it just replaces with mean Age of all passengers
#
# Advantages of KNN Imputer:
# + Captures feature correlations (multivariate)
# + Produces more realistic values
# + Often improves ML model performance
#
# Disadvantages of KNN Imputer:
# - Computationally expensive (distance calculation)
# - Sensitive to irrelevant/noisy features
# - Requires feature scaling for best results
#
# Real-world Example:
# - Healthcare: Filling missing "blood pressure" using patients with 
#   similar "age, weight, and cholesterol levels"
# - Titanic dataset: Filling missing "Age" using passengers with 
#   similar "Pclass" and "Fare"
# ----------------------------------------------------------
