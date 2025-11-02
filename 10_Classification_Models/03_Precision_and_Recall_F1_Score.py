# ============================================================
#              PRECISION, RECALL, AND F1-SCORE
# ============================================================

# ------------------------------------------------------------
# WHAT IS PRECISION?
# ------------------------------------------------------------
# Precision measures the accuracy of positive predictions.
# It answers the question:
# "Out of all predicted positives, how many were actually positive?"

# Formula:
# Precision = True Positives / (True Positives + False Positives)

# Example:
# Suppose a model predicts 10 patients have a disease.
# Out of these 10, only 7 actually have the disease.
# Precision = 7 / (7 + 3) = 0.7 (70%)
# → This means 70% of the predicted positives were correct.


# ------------------------------------------------------------
# WHAT IS RECALL?
# ------------------------------------------------------------
# Recall (Sensitivity or True Positive Rate) measures how well the model 
# identifies all actual positive cases.
# It answers the question:
# "Out of all actual positives, how many did the model correctly identify?"

# Formula:
# Recall = True Positives / (True Positives + False Negatives)

# Example:
# There are 20 patients who actually have the disease.
# The model correctly predicts 15 of them.
# Recall = 15 / (15 + 5) = 0.75 (75%)
# → The model detects 75% of all actual positive cases.


# ------------------------------------------------------------
# WHAT IS F1-SCORE?
# ------------------------------------------------------------
# The F1-score is the harmonic mean of Precision and Recall.
# It balances both metrics, especially when the data is imbalanced.

# Formula:
# F1 = 2 * (Precision * Recall) / (Precision + Recall)

# Example:
# Precision = 0.7, Recall = 0.75
# F1 = 2 * (0.7 * 0.75) / (0.7 + 0.75) = 0.724
# → The F1-score gives a balanced measure of accuracy between Precision and Recall.


# ============================================================
#                  PRACTICAL IMPLEMENTATION
# ============================================================

# Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_score,
    recall_score, f1_score, classification_report
)

# ------------------------------------------------------------
# Example 1: Binary Classification (Heart Disease Dataset)
# ------------------------------------------------------------

# Load dataset
df = pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')

# Split features and target
X_train, X_test, y_train, y_test = train_test_split(
    df.iloc[:, :-1], df.iloc[:, -1],
    test_size=0.2, random_state=2
)

# Train Logistic Regression and Decision Tree models
clf1 = LogisticRegression()
clf2 = DecisionTreeClassifier()
clf1.fit(X_train, y_train)
clf2.fit(X_train, y_train)

# Predictions
y_pred1 = clf1.predict(X_test)
y_pred2 = clf2.predict(X_test)

# Accuracy comparison
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


# ------------------------------------------------------------
# Example 2: Multi-Class Classification (Iris Dataset)
# ------------------------------------------------------------

from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv('/kaggle/input/iris/Iris.csv')

# Encode labels (convert text labels to numbers)
encoder = LabelEncoder()
df['Species'] = encoder.fit_transform(df['Species'])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    df.iloc[:, 1:-1], df.iloc[:, -1],
    test_size=0.2, random_state=1
)

# Train models
clf1 = LogisticRegression()
clf2 = DecisionTreeClassifier()
clf1.fit(X_train, y_train)
clf2.fit(X_train, y_train)

# Predictions
y_pred1 = clf1.predict(X_test)
y_pred2 = clf2.predict(X_test)

print("\nAccuracy of Logistic Regression:", accuracy_score(y_test, y_pred1))
print("Accuracy of Decision Tree:", accuracy_score(y_test, y_pred2))

print("\nLogistic Regression Confusion Matrix:\n", pd.DataFrame(confusion_matrix(y_test, y_pred1)))
print("\nDecision Tree Confusion Matrix:\n", pd.DataFrame(confusion_matrix(y_test, y_pred2)))

# Precision and Recall for each class
print("\nPrecision per class (Logistic Regression):", precision_score(y_test, y_pred1, average=None))
print("Recall per class (Logistic Regression):", recall_score(y_test, y_pred1, average=None))

# ------------------------------------------------------------
# Example 3: Multi-Class Classification (Digit Recognizer)
# ------------------------------------------------------------

df = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df.iloc[:, 1:], df.iloc[:, 0],
    test_size=0.2, random_state=2
)

# Train models
clf1 = LogisticRegression(max_iter=1000)
clf2 = DecisionTreeClassifier()
clf1.fit(X_train, y_train)
clf2.fit(X_train, y_train)

# Predictions
y_pred1 = clf1.predict(X_test)
y_pred2 = clf2.predict(X_test)

print("\nAccuracy of Logistic Regression:", accuracy_score(y_test, y_pred1))
print("Accuracy of Decision Tree:", accuracy_score(y_test, y_pred2))

print("\nLogistic Regression Confusion Matrix:\n", pd.DataFrame(confusion_matrix(y_test, y_pred1)))
print("\nDecision Tree Confusion Matrix:\n", pd.DataFrame(confusion_matrix(y_test, y_pred2)))

# Classification Report for Logistic Regression
print("\nClassification Report (Logistic Regression):")
print(classification_report(y_test, y_pred1))


# ============================================================
# Multi-class Precision, Recall, and F1 — Cat, Dog, Rabbit Example
# ============================================================

# Suppose we have three classes: Cat, Dog, Rabbit
# Confusion Matrix Example:
#            Predicted
#            Cat  Dog  Rabbit
# Actual Cat   5    2     0
# Actual Dog   1    7     2
# Actual Rabbit 0   1     6

# We can calculate metrics per class:
# Precision (Cat) = 5 / (5 + 1) = 0.833
# Recall (Cat)    = 5 / (5 + 2) = 0.714
# F1 (Cat)        = 2 * (0.833 * 0.714) / (0.833 + 0.714) = 0.769

# ------------------------------------------------------------
# MACRO vs WEIGHTED Precision (Multi-class Averaging)
# ------------------------------------------------------------
# average='macro'   → Calculates Precision, Recall, F1 for each class,
#                     then takes the unweighted mean (treats all classes equally).
# average='weighted'→ Calculates Precision, Recall, F1 for each class,
#                     weighted by the number of samples per class
#                     (good when classes are imbalanced).

# Example:
# precision_score(y_test, y_pred, average='macro')
# precision_score(y_test, y_pred, average='weighted')

# ------------------------------------------------------------
# CALCULATE ALL SCORES
# ------------------------------------------------------------
print("\nMacro Precision (Digits Dataset):", precision_score(y_test, y_pred1, average='macro'))
print("Weighted Precision (Digits Dataset):", precision_score(y_test, y_pred1, average='weighted'))

print("\nMacro Recall (Digits Dataset):", recall_score(y_test, y_pred1, average='macro'))
print("Weighted Recall (Digits Dataset):", recall_score(y_test, y_pred1, average='weighted'))

print("\nMacro F1-Score (Digits Dataset):", f1_score(y_test, y_pred1, average='macro'))
print("Weighted F1-Score (Digits Dataset):", f1_score(y_test, y_pred1, average='weighted'))
