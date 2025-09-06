# =============================================
# Feature Engineering in Machine Learning 
# =============================================

# What is Feature Engineering?
# ----------------------------
# Feature engineering is the process of using domain knowledge
# to extract or create features (input variables) from raw data.
# These features can improve the performance of machine learning algorithms.

# Why, Where, When, How?
# ----------------------
# WHY  -> To improve model accuracy and efficiency.
# WHERE -> Any dataset: text, numbers, images, audio, etc.
# WHEN -> Before training ML models (data preprocessing stage).
# HOW  -> By applying transformations, constructing new features,
#         selecting important ones, and extracting from unstructured data.

# --------------------------------------------------
# 1. FEATURE TRANSFORMATION 
# --------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

data = {
    "Name": ["Ali", "Sara", "John", "Mary", "Tom"],
    "Age": [25, np.nan, 30, 22, np.nan],
    "Gender": ["Male", "Female", "Male", "Female", "Male"],
    "Salary": [30000, 35000, 40000, 1000000, 45000],
    "Height_cm": [160, 170, 180, 175, 165],
    "Weight_kg": [50, 80, 100, 70, 60],
    "Date_of_Birth": pd.to_datetime(
        ["2000-01-01", "1998-05-15", "1995-07-23", "2002-09-10", "1999-12-30"]
    )
}

df = pd.DataFrame(data)

# a) Missing Values Handling
# Fill missing Age values with the mean of the column
df["Age"].fillna(df["Age"].mean(), inplace=True)

# b) Handling Categorical Features
# Label Encoding -> Gender column
label_encoder = LabelEncoder()
df["Gender_Label"] = label_encoder.fit_transform(df["Gender"])

# One-Hot Encoding -> Gender column
df = pd.get_dummies(df, columns=["Gender"], drop_first=True)

# c) Outlier Detection & Handling
# Example: Cap Salary values greater than 100000
df["Salary_Capped"] = np.where(df["Salary"] > 100000, 100000, df["Salary"])

# d) Feature Scaling
# Normalization (0-1 range) for Height
scaler_minmax = MinMaxScaler()
df["Height_Scaled"] = scaler_minmax.fit_transform(df[["Height_cm"]])

# Standardization (mean=0, std=1) for Weight
scaler_standard = StandardScaler()
df["Weight_Standardized"] = scaler_standard.fit_transform(df[["Weight_kg"]])

# --------------------------------------------------
# 2. FEATURE CONSTRUCTION 
# --------------------------------------------------
# Creating new features from existing data
current_year = pd.Timestamp.now().year
df["Age_from_DOB"] = current_year - df["Date_of_Birth"].dt.year

# --------------------------------------------------
# 3. FEATURE SELECTION 
# --------------------------------------------------
# Removing irrelevant/unnecessary columns
df_selected = df.drop(columns=["Name"])

# --------------------------------------------------
# 4. FEATURE EXTRACTION 
# --------------------------------------------------
# Extracting new features from raw/unstructured data
# Example: Length of Name string
df_selected["Name_Length"] = df["Name"].apply(len)

# --------------------------------------------------
# Summary:
# - Transformation = clean/prepare data (missing values, scaling, encoding)
# - Construction   = create new features (Age from DOB, Speed from Distance/Time)
# - Selection      = keep only useful features (drop irrelevant ones)
# - Extraction     = pull info from unstructured data (text, images, audio)
# --------------------------------------------------

print(df_selected.head())
