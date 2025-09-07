# Column Transformer in Sklearn for Machine Learning & Data Science
# ---------------------------------------------------------------
# When doing feature engineering, you often need different preprocessing
# steps for different column types (numeric, categorical, ordinal, etc.)
# Without ColumnTransformer, you’d manually process each column and 
# concatenate arrays. That’s tedious.
# ColumnTransformer solves this by applying multiple transformations
# in a clean pipeline.

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer

# Load dataset
df = pd.read_csv("covid_toy.csv")
print(df.head())

# Target and features
X = df.drop(columns=["has_covid"])
y = df["has_covid"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Missing values:\n", df.isnull().sum())

# ---------------------------
# Manual Transformation (not recommended in practice, but for understanding)
# ---------------------------

# SimpleImputer for Fever column
si = SimpleImputer(strategy="mean")
X_train_fever = si.fit_transform(X_train[["Fever"]])
X_test_fever = si.transform(X_test[["Fever"]])

# Ordinal Encoding for Cough column
oe = OrdinalEncoder(categories=[["Mild", "Strong"]])
X_train_cough = oe.fit_transform(X_train[["Cough"]])
X_test_cough = oe.transform(X_test[["Cough"]])

# OneHot Encoding for Gender & City
ohe = OneHotEncoder(drop="first", sparse=False)
X_train_gender_city = ohe.fit_transform(X_train[["Gender", "City"]])
X_test_gender_city = ohe.transform(X_test[["Gender", "City"]])

# Remaining numeric columns (e.g., Age)
X_train_age = X_train.drop(columns=["Gender", "Fever", "Cough", "City"]).values
X_test_age = X_test.drop(columns=["Gender", "Fever", "Cough", "City"]).values

# Combine everything
X_train_manual = np.concatenate(
    [X_train_fever, X_train_cough, X_train_gender_city, X_train_age], axis=1
)
X_test_manual = np.concatenate(
    [X_test_fever, X_test_cough, X_test_gender_city, X_test_age], axis=1
)

print("Manual transformed train shape:", X_train_manual.shape)

# ---------------------------
# Best Practice: ColumnTransformer
# ---------------------------

transformer = ColumnTransformer(
    transformers=[
        ("tnf1", SimpleImputer(strategy="mean"), ["Fever"]),  # impute Fever
        ("tnf2", OrdinalEncoder(categories=[["Mild", "Strong"]]), ["Cough"]),  # ordinal encode Cough
        ("tnf3", OneHotEncoder(drop="first", sparse=False), ["Gender", "City"]),  # one-hot encode Gender & City
    ],
    remainder="passthrough"  # keep other columns (like Age)
)

# Fit on train data and transform
X_train_transformed = transformer.fit_transform(X_train)
X_test_transformed = transformer.transform(X_test)

print("ColumnTransformer train shape:", X_train_transformed.shape)
print("ColumnTransformer test shape:", X_test_transformed.shape)
