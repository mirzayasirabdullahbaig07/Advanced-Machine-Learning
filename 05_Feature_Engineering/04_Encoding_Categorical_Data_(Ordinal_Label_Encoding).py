# ================================================
# Encoding Categorical Data (Ordinal & Label Encoding)
# ================================================
# Feature engineering is the process of transforming raw data 
# into features that improve the performance of machine learning models.
# One of the most important parts of feature engineering is **Feature Transformation**.
#
# A type of feature transformation is **Feature Scaling**,
# and another important one is **Encoding Categorical Data**.
#
# -------------------------------
# Types of Data in Machine Learning
# -------------------------------
# 1. Numerical Data → numbers (int, float)
# 2. Categorical Data → string labels / categories
#
# -------------------------------
# Types of Categorical Data
# -------------------------------
# a) Nominal Data:
#    - Categories have no order or ranking
#    - Example: Colors = [Red, Blue, Green]
#    - Real-world use: Gender, Country, City names
#    - Encoding: One-Hot Encoding
#
# b) Ordinal Data:
#    - Categories have a meaningful order
#    - Example: Ratings = [Poor < Average < Good]
#    - Real-world use: Education Level (School < UG < PG), Satisfaction levels
#    - Encoding: Ordinal Encoding
#
# -------------------------------
# Why Encoding?
# -------------------------------
# ML algorithms only understand numeric data,
# not string categories.
# Encoding converts categorical → numerical form.
#
# -------------------------------
# Encoding Types
# -------------------------------
# 1. Ordinal Encoding → for ordered categorical features (X variables)
# 2. Label Encoding → for target variable (y) classes
# 3. One-Hot Encoding → for nominal categorical features (no order)
#
# -------------------------------
# In this example:
# -------------------------------
# - We will use OrdinalEncoder on X (features)
# - We will use LabelEncoder on y (target)

# ================================================
# Step 1: Import libraries
# ================================================
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

# ================================================
# Step 2: Load dataset
# ================================================
# (Assume customer.csv exists in same folder)
df = pd.read_csv('customer.csv')
print("Sample data:")
print(df.sample(5))

# Drop unnecessary columns (keeping only relevant ones)
df = df.iloc[:, 2:]   # Keeping only categorical + target columns
print("\nProcessed Data:")
print(df.head())

# ================================================
# Step 3: Split into features (X) and target (y)
# ================================================
X = df.iloc[:, 0:2]   # First two columns → features
y = df.iloc[:, -1]    # Last column → target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ================================================
# Step 4: Apply Ordinal Encoding on X (features)
# ================================================
# Define category order manually
oe = OrdinalEncoder(categories=[['Poor', 'Average', 'Good'], 
                                ['School', 'UG', 'PG']])

# Fit and transform training data
oe.fit(X_train)

X_train = oe.transform(X_train)
X_test = oe.transform(X_test)

print("\nEncoded X_train (Ordinal Encoding):")
print(X_train[:5])
print("\nOrdinal categories learned:")
print(oe.categories_)

# ================================================
# Step 5: Apply Label Encoding on y (target)
# ================================================
le = LabelEncoder()

# Fit on training labels
le.fit(y_train)

print("\nLabel classes (y):")
print(le.classes_)

# Transform target values
y_train = le.transform(y_train)
y_test = le.transform(y_test)

print("\nEncoded y_train (Label Encoding):")
print(y_train[:10])
