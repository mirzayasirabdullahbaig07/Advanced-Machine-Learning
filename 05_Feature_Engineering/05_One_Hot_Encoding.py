# ==================================================
# One-Hot Encoding (OHE) - Nominal Categorical Data
# ==================================================
# Many ML models do not accept string (categorical) data, 
# so we need to convert categorical → numeric.
#
# If the data is **Nominal** (no inherent order), we use **One-Hot Encoding**.
#
# -------------------------------
# Key Concepts
# -------------------------------
# 1. Dummy Variable Trap:
#    - When we create one dummy variable per category, they become redundant.
#    - Example: For 3 brands [BMW, Audi, Toyota], if we have 3 dummy variables,
#      knowing two of them is enough to deduce the third.
#    - This causes "Multicollinearity" (variables correlated with each other).
#    - Solution: Drop one dummy column (n categories → n-1 columns).
#
# 2. Multicollinearity:
#    - Strong correlation between features.
#    - Bad for models like Linear Regression (inflates variance).
#
# 3. Handling Rare Categories:
#    - Some categories may appear very few times (low frequency).
#    - These can be grouped together into a single category (e.g., "Other" or "Uncommon").
#
# -------------------------------
# Approaches for OHE
# -------------------------------
# a) Pandas → pd.get_dummies()
# b) Scikit-learn → OneHotEncoder
#
# -------------------------------
# Let's apply both methods step by step
# -------------------------------

# ================================================
# Step 1: Import libraries
# ================================================
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# ================================================
# Step 2: Load dataset
# ================================================
df = pd.read_csv('cars.csv')
print("Original Data:")
print(df.head())

# ================================================
# Step 3: Explore 'brand' column
# ================================================
print("\nBrand Value Counts:")
print(df['brand'].value_counts())
print("\nNumber of unique brands:", df['brand'].nunique())

# ================================================
# Step 4: OHE using Pandas
# ================================================
# Method 1: Keep all dummy variables
df_ohe_all = pd.get_dummies(df, columns=['brand'])
print("\nData with all dummy columns:")
print(df_ohe_all.head())

# Method 2: Drop first column → avoid dummy trap
df_ohe_drop = pd.get_dummies(df, columns=['brand'], drop_first=True)
print("\nData with n-1 dummy columns:")
print(df_ohe_drop.head())

# ================================================
# Step 5: Train-Test Split
# ================================================
X = df.iloc[:, 0:4]   # First few columns as features
y = df.iloc[:, -1]    # Last column as target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nX_train sample:")
print(X_train.head())

# ================================================
# Step 6: OHE using Sklearn
# ================================================
# drop='first' → avoid dummy trap
# sparse=False → return array instead of sparse matrix
# dtype=np.int32 → output in integer format (0/1)
ohe = OneHotEncoder(drop='first', sparse=False, dtype=np.int32)

# Fit only on training data (prevent data leakage)
X_train_new = ohe.fit_transform(X_train[['brand']])
X_test_new = ohe.transform(X_test[['brand']])

print("\nShape after OHE:")
print("X_train_new:", X_train_new.shape)
print("X_test_new:", X_test_new.shape)

# Combine OHE results with numerical features
X_train_final = np.hstack((X_train[['km_driven']].values, X_train_new))
X_test_final = np.hstack((X_test[['km_driven']].values, X_test_new))

print("\nFinal X_train shape (with OHE + numeric features):", X_train_final.shape)

# ================================================
# Step 7: Handling Rare Categories
# ================================================
print("\nHandling Rare Categories:")
counts = df['brand'].value_counts()
threshold = 100   # if a brand appears <= 100 times, group it as 'uncommon'

# Replace rare brands with "uncommon"
df['brand_grouped'] = df['brand'].replace(counts[counts <= threshold].index, 'uncommon')

# Apply OHE again on grouped brands
df_ohe_grouped = pd.get_dummies(df, columns=['brand_grouped'], drop_first=True)
print("\nAfter grouping rare categories:")
print(df_ohe_grouped.head())
