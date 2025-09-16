"""
====================================================
 Categorical Imputation (Most Frequent & Missing Category)
====================================================

📌 What is Categorical Imputation?
------------------------------------
Categorical imputation is the process of handling missing values in categorical
variables (string or object type columns) by replacing them with suitable values.

There are two common approaches:
1. Most Frequent Value Imputation (Mode Imputation)
2. Missing Category Imputation

------------------------------------------------------
1️⃣ Most Frequent Value (Mode) Imputation
------------------------------------------------------
✔ What is it?
   - Replace missing values with the most frequently occurring category (mode).

✔ Where to use?
   - When missing data is small and random.
   - When the most common category is a good representative.

✔ How to use?
   - Find the mode (most frequent value).
   - Replace NaN with this mode.

✔ Benefits:
   - Easy to implement.
   - Works well when missingness is random and small.

✔ Drawback:
   - May distort relationships if missing values are not random.
   - Can reduce variability.

------------------------------------------------------
2️⃣ Missing Category Imputation
------------------------------------------------------
✔ What is it?
   - Replace missing values with a new category such as "Missing".

✔ Where to use?
   - When missing values might carry information (e.g., missing because the feature is not applicable).
   - When missing data is not random.

✔ Benefits:
   - Preserves all data (no row dropped).
   - Allows the model to capture the "missingness" effect.

✔ Drawback:
   - Introduces an artificial category.
   - Can cause overfitting if missingness is rare.

====================================================
"""

# =======================
# Import libraries
# =======================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# =======================
# Load dataset
# =======================
df = pd.read_csv('train.csv', usecols=['GarageQual', 'FireplaceQu', 'SalePrice'])
print(df.head())
print("\n% Missing Values:\n", df.isnull().mean() * 100)

# ============================================================
# PART 1: Most Frequent Value (Mode) Imputation
# ============================================================

# -------------------------------
# Step 1: Explore GarageQual
# -------------------------------
df['GarageQual'].value_counts().plot(kind='bar')
plt.title("GarageQual Distribution (Before Imputation)")
plt.show()

print("\nMode of GarageQual:", df['GarageQual'].mode()[0])

# -------------------------------
# Step 2: Compare SalePrice distribution
# -------------------------------
fig = plt.figure()
ax = fig.add_subplot(111)

df[df['GarageQual'] == 'TA']['SalePrice'].plot(kind='kde', ax=ax)
df[df['GarageQual'].isnull()]['SalePrice'].plot(kind='kde', ax=ax, color='red')

lines, labels = ax.get_legend_handles_labels()
labels = ['Houses with TA', 'Houses with NA']
ax.legend(lines, labels, loc='best')
plt.title('GarageQual - Before Imputation')
plt.show()

# -------------------------------
# Step 3: Impute missing values with Mode
# -------------------------------
df['GarageQual'].fillna('TA', inplace=True)
df['GarageQual'].value_counts().plot(kind='bar')
plt.title("GarageQual Distribution (After Mode Imputation)")
plt.show()

# -------------------------------
# Step 4: Distribution check after imputation
# -------------------------------
fig = plt.figure()
ax = fig.add_subplot(111)

df[df['GarageQual'] == 'TA']['SalePrice'].plot(kind='kde', ax=ax, color='red')
plt.title("GarageQual - After Mode Imputation")
plt.show()

# ============================================================
# Another Example: FireplaceQu
# ============================================================

df['FireplaceQu'].value_counts().plot(kind='bar')
plt.title("FireplaceQu Distribution (Before Imputation)")
plt.show()

print("\nMode of FireplaceQu:", df['FireplaceQu'].mode()[0])

fig = plt.figure()
ax = fig.add_subplot(111)

df[df['FireplaceQu'] == 'Gd']['SalePrice'].plot(kind='kde', ax=ax)
df[df['FireplaceQu'].isnull()]['SalePrice'].plot(kind='kde', ax=ax, color='red')

lines, labels = ax.get_legend_handles_labels()
labels = ['Houses with Gd', 'Houses with NA']
ax.legend(lines, labels, loc='best')
plt.title('FireplaceQu - Before Imputation')
plt.show()

# Impute with mode "Gd"
df['FireplaceQu'].fillna('Gd', inplace=True)
df['FireplaceQu'].value_counts().plot(kind='bar')
plt.title("FireplaceQu Distribution (After Mode Imputation)")
plt.show()

# ============================================================
# PART 2: Missing Category Imputation
# ============================================================

# -------------------------------
# Step 1: Visualize missing values
# -------------------------------
df2 = pd.read_csv('train.csv', usecols=['GarageQual', 'FireplaceQu', 'SalePrice'])
print("\n% Missing Values:\n", df2.isnull().mean() * 100)

df2['GarageQual'].value_counts().sort_values(ascending=False).plot.bar()
plt.title("GarageQual Distribution (Before Missing Category Imputation)")
plt.show()

# -------------------------------
# Step 2: Fill with "Missing"
# -------------------------------
df2['GarageQual'].fillna('Missing', inplace=True)
df2['GarageQual'].value_counts().sort_values(ascending=False).plot.bar()
plt.title("GarageQual Distribution (After Missing Category Imputation)")
plt.show()

# ============================================================
# Using Scikit-learn Imputers
# ============================================================

# -------------------------------
# Mode Imputation
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df.drop(columns=['SalePrice']), df['SalePrice'], test_size=0.2
)

imputer_mode = SimpleImputer(strategy='most_frequent')
X_train_mode = imputer_mode.fit_transform(X_train)
X_test_mode = imputer_mode.transform(X_test)

print("\nMost Frequent Imputer Statistics:", imputer_mode.statistics_)

# -------------------------------
# Missing Category Imputation
# -------------------------------
X_train2, X_test2, y_train2, y_test2 = train_test_split(
    df2.drop(columns=['SalePrice']), df2['SalePrice'], test_size=0.2
)

imputer_missing = SimpleImputer(strategy='constant', fill_value='Missing')
X_train_missing = imputer_missing.fit_transform(X_train2)
X_test_missing = imputer_missing.transform(X_test2)

print("\nMissing Category Imputer Statistics:", imputer_missing.statistics_)
