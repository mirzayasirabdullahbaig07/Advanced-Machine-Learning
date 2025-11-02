# ==========================================================
# CONFUSION MATRIX, ACCURACY, TYPE I & TYPE II ERRORS
# ==========================================================

# Accuracy is one of the simplest evaluation metrics.
# It measures the proportion of correct predictions made by the model.
# 
# Formula:
#     Accuracy = (True Positives + True Negatives) / (Total Predictions)
#
# Example: If a model correctly predicts 90 out of 100 samples,
# Accuracy = 90 / 100 = 0.90 (or 90%)

# ----------------------------------------------------------
# Accuracy in Multi-Class Classification
# ----------------------------------------------------------
# For multi-class problems, accuracy is calculated similarly:
#     Accuracy = (Number of Correct Predictions) / (Total Predictions)
#
# For example:
# If a model correctly classifies 910 images out of 1000,
# the accuracy is 91%.

# ----------------------------------------------------------
# When is Accuracy Good or Bad?
# ----------------------------------------------------------
# Accuracy depends on the problem you're solving.
# - It’s useful when your dataset is balanced.
# - It’s misleading when your dataset is imbalanced.
#
# Example:
# In medical diagnosis, if 99% of people are healthy,
# a model that always predicts “healthy” will have 99% accuracy
# but 0% usefulness — it never detects the disease.

# ----------------------------------------------------------
# Problem with Accuracy
# ----------------------------------------------------------
# Accuracy only tells you *how many* errors occurred,
# not *what type* of errors they were.
#
# Example:
# In a student placement model:
# The model shows 10 errors but doesn't specify if the error
# was predicting a placed student as unplaced or vice versa.
#
# To understand the *type of errors*, we use the **Confusion Matrix**.

# ----------------------------------------------------------
# Confusion Matrix
# ----------------------------------------------------------
# The confusion matrix shows the count of correct and incorrect predictions
# broken down by each class.
#
# Binary Classification Example:
#                 Predicted
#               |  1  |  0  |
#           ----------------
#   Actual  1  | TP | FN |
#           0  | FP | TN |
#
# Where:
# - TP (True Positive): Correctly predicted positive class (1 → 1)
# - TN (True Negative): Correctly predicted negative class (0 → 0)
# - FP (False Positive): Incorrectly predicted positive (0 → 1)
#   → Type I Error
# - FN (False Negative): Incorrectly predicted negative (1 → 0)
#   → Type II Error

# ----------------------------------------------------------
# Type I & Type II Errors
# ----------------------------------------------------------
# Type I Error (False Positive):
# - Model predicts positive, but it’s actually negative.
# - Example: Predicting a person has a disease when they don’t.
#
# Type II Error (False Negative):
# - Model predicts negative, but it’s actually positive.
# - Example: Predicting a person is healthy when they’re sick.

# ----------------------------------------------------------
# Confusion Matrix for Multi-Class Classification
# ----------------------------------------------------------
# For multi-class problems (e.g., digits 0–9),
# the confusion matrix becomes a larger square matrix (e.g., 10x10),
# where each row and column corresponds to a class.
#
# Each cell [i, j] represents how many samples of class i
# were predicted as class j.
#
# Diagonal elements → Correct predictions.
# Off-diagonal elements → Misclassifications.

# ----------------------------------------------------------
# Example 1: Binary Classification (Heart Disease Dataset)
# ----------------------------------------------------------

# Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

# Load dataset
df = pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')

# Split data
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-1], df.iloc[:, -1], test_size=0.2, random_state=2)

# Train models
clf1 = LogisticRegression()
clf2 = DecisionTreeClassifier()
clf1.fit(X_train, y_train)
clf2.fit(X_train, y_train)

# Predictions
y_pred1 = clf1.predict(X_test)
y_pred2 = clf2.predict(X_test)

# Accuracy
print("Accuracy of Logistic Regression:", accuracy_score(y_test, y_pred1))
print("Accuracy of Decision Tree:", accuracy_score(y_test, y_pred2))

# Confusion Matrices
print("\nLogistic Regression Confusion Matrix:\n", pd.DataFrame(confusion_matrix(y_test, y_pred1)))
print("\nDecision Tree Confusion Matrix:\n", pd.DataFrame(confusion_matrix(y_test, y_pred2)))

# Precision, Recall, F1-score
print("\nFor Logistic Regression Model")
print("-" * 50)
print("Precision:", precision_score(y_test, y_pred1))
print("Recall:", recall_score(y_test, y_pred1))
print("F1 Score:", f1_score(y_test, y_pred1))

print("\nFor Decision Tree Model")
print("-" * 50)
print("Precision:", precision_score(y_test, y_pred2))
print("Recall:", recall_score(y_test, y_pred2))
print("F1 Score:", f1_score(y_test, y_pred2))

# ----------------------------------------------------------
# Example 2: Multi-Class Classification (Iris Dataset)
# ----------------------------------------------------------

from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('/kaggle/input/iris/Iris.csv')
encoder = LabelEncoder()
df['Species'] = encoder.fit_transform(df['Species'])

X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, 1:-1], df.iloc[:, -1], test_size=0.2, random_state=1)

clf1 = LogisticRegression()
clf2 = DecisionTreeClassifier()
clf1.fit(X_train, y_train)
clf2.fit(X_train, y_train)

y_pred1 = clf1.predict(X_test)
y_pred2 = clf2.predict(X_test)

print("\nAccuracy of Logistic Regression:", accuracy_score(y_test, y_pred1))
print("Accuracy of Decision Tree:", accuracy_score(y_test, y_pred2))

print("\nLogistic Regression Confusion Matrix:\n", pd.DataFrame(confusion_matrix(y_test, y_pred1)))
print("\nDecision Tree Confusion Matrix:\n", pd.DataFrame(confusion_matrix(y_test, y_pred2)))

print("\nPrecision per class (Logistic Regression):", precision_score(y_test, y_pred1, average=None))
print("Recall per class (Logistic Regression):", recall_score(y_test, y_pred1, average=None))

# ----------------------------------------------------------
# Example 3: Multi-Class Classification (Digit Recognizer)
# ----------------------------------------------------------

df = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, 1:], df.iloc[:, 0], test_size=0.2, random_state=2)

clf1 = LogisticRegression()
clf2 = DecisionTreeClassifier()
clf1.fit(X_train, y_train)
clf2.fit(X_train, y_train)

y_pred1 = clf1.predict(X_test)
y_pred2 = clf2.predict(X_test)

print("\nAccuracy of Logistic Regression:", accuracy_score(y_test, y_pred1))
print("Accuracy of Decision Tree:", accuracy_score(y_test, y_pred2))

print("\nLogistic Regression Confusion Matrix:\n", pd.DataFrame(confusion_matrix(y_test, y_pred1)))
print("\nDecision Tree Confusion Matrix:\n", pd.DataFrame(confusion_matrix(y_test, y_pred2)))

from sklearn.metrics import classification_report

print("\nClassification Report (Logistic Regression):")
print(classification_report(y_test, y_pred1))

# ----------------------------------------------------------
# When Accuracy is Misleading
# ----------------------------------------------------------
# Accuracy is not always the best metric — especially for **imbalanced datasets**.
# For example:
# - In fraud detection or disease diagnosis, the minority class is more important.
# - High accuracy may hide the fact that the model fails to detect the minority class.
#
# In such cases, use:
# - Confusion Matrix
# - Precision, Recall, and F1 Score
# - ROC-AUC Curve
# - Matthews Correlation Coefficient (MCC)

# ----------------------------------------------------------
# Summary
# ----------------------------------------------------------
# - Accuracy gives an overall idea of model performance.
# - Confusion Matrix explains *what kind* of errors occurred.
# - Type I Error (False Positive): Model predicts positive but it’s negative.
# - Type II Error (False Negative): Model predicts negative but it’s positive.
# - For balanced datasets, accuracy works well.
# - For imbalanced datasets, prefer Precision, Recall, and F1-score.
