# ----------------------------------------------------------
# MICE Algorithm (Multivariate Imputation by Chained Equations)
# ----------------------------------------------------------
# WHAT IS MICE?
# - MICE (also called Iterative Imputer in sklearn) is a powerful method for 
#   imputing missing data by creating predictive models for each feature 
#   that has missing values.
# - It works iteratively: each feature with missing data is predicted 
#   using other features, and this process is repeated multiple times 
#   until values stabilize.
#
# WHY DO WE USE IT?
# - Unlike simple imputers (mean/median), MICE leverages relationships 
#   among multiple features (multivariate).
# - This results in more realistic imputations and better preservation 
#   of statistical properties.
#
# WHEN TO USE MICE?
# - When dataset has multiple features with missing values.
# - When features are correlated with each other.
# - Especially useful in healthcare, finance, and survey data.
#
# ASSUMPTIONS ABOUT MISSING DATA:
# - MCAR (Missing Completely At Random):
#   Probability of missingness is independent of data (observed or unobserved).
#   Example: A survey sheet gets lost randomly.
#
# - MAR (Missing At Random):
#   Probability of missingness depends on observed data but not on the 
#   missing data itself.
#   Example: Older patients are less likely to report income in a survey.
#
# - MNAR (Missing Not At Random):
#   Probability of missingness depends on the missing values themselves.
#   Example: People with very high income choose not to disclose their income.
#
# ADVANTAGES OF MICE:
# + Uses multivariate relationships, preserving data structure.
# + More accurate than mean/median imputations.
# + Can handle multiple columns with missing values.
#
# DISADVANTAGES OF MICE:
# - Computationally heavy (requires building multiple models iteratively).
# - Storage and memory intensive for large datasets.
# - Results may vary depending on number of iterations and models used.
#
# REAL-WORLD EXAMPLE:
# - Healthcare: imputing missing patient lab test values based on correlated features.
# - Business: imputing missing "Marketing Spend" using "R&D Spend" and "Administration".
# ----------------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Load dataset (scaled for simplicity)
df = np.round(pd.read_csv('50_Startups.csv')[['R&D Spend','Administration','Marketing Spend','Profit']] / 10000)

# Take random sample of 5 rows
np.random.seed(9)
df = df.sample(5)
print("Original Sample:\n", df)

# Drop Profit column to only impute features
df = df.iloc[:, 0:-1]
print("\nSelected Features:\n", df)

# Introduce artificial missing values
df.iloc[1,0] = np.NaN   # Missing in R&D Spend
df.iloc[3,1] = np.NaN   # Missing in Administration
df.iloc[-1,-1] = np.NaN # Missing in Marketing Spend
print("\nData with Missing Values:\n", df)

# ----------------------------------------------------------
# Step 1: Initial Imputation (0th Iteration) using column mean
# ----------------------------------------------------------
df0 = pd.DataFrame()
df0['R&D Spend'] = df['R&D Spend'].fillna(df['R&D Spend'].mean())
df0['Administration'] = df['Administration'].fillna(df['Administration'].mean())
df0['Marketing Spend'] = df['Marketing Spend'].fillna(df['Marketing Spend'].mean())
print("\n0th Iteration (Mean Imputation):\n", df0)

# ----------------------------------------------------------
# Step 2: Iterative Imputation (Manual Demonstration)
# ----------------------------------------------------------
# Iteration 1:
# Predict missing R&D Spend using other columns
df1 = df0.copy()
df1.iloc[1,0] = np.NaN
X = df1.iloc[[0,2,3,4], 1:3]
y = df1.iloc[[0,2,3,4], 0]
lr = LinearRegression()
lr.fit(X,y)
df1.iloc[1,0] = lr.predict(df1.iloc[1,1:].values.reshape(1,2))[0]

# Predict missing Administration
df1.iloc[3,1] = np.NaN
X = df1.iloc[[0,1,2,4],[0,2]]
y = df1.iloc[[0,1,2,4],1]
lr = LinearRegression()
lr.fit(X,y)
df1.iloc[3,1] = lr.predict(df1.iloc[3,[0,2]].values.reshape(1,2))[0]

# Predict missing Marketing Spend
df1.iloc[4,-1] = np.NaN
X = df1.iloc[0:4,0:2]
y = df1.iloc[0:4,-1]
lr = LinearRegression()
lr.fit(X,y)
df1.iloc[4,-1] = lr.predict(df1.iloc[4,0:2].values.reshape(1,2))[0]

print("\nAfter 1st Iteration (MICE):\n", df1)

# Subtract differences between Iteration 0 and Iteration 1
print("\nChange after 1st Iteration:\n", df1 - df0)

# ----------------------------------------------------------
# Iteration 2 (repeat process)
# ----------------------------------------------------------
df2 = df1.copy()
df2.iloc[1,0] = np.NaN
X = df2.iloc[[0,2,3,4],1:3]
y = df2.iloc[[0,2,3,4],0]
lr = LinearRegression()
lr.fit(X,y)
df2.iloc[1,0] = lr.predict(df2.iloc[1,1:].values.reshape(1,2))[0]

df2.iloc[3,1] = np.NaN
X = df2.iloc[[0,1,2,4],[0,2]]
y = df2.iloc[[0,1,2,4],1]
lr = LinearRegression()
lr.fit(X,y)
df2.iloc[3,1] = lr.predict(df2.iloc[3,[0,2]].values.reshape(1,2))[0]

df2.iloc[4,-1] = np.NaN
X = df2.iloc[0:4,0:2]
y = df2.iloc[0:4,-1]
lr = LinearRegression()
lr.fit(X,y)
df2.iloc[4,-1] = lr.predict(df2.iloc[4,0:2].values.reshape(1,2))[0]

print("\nAfter 2nd Iteration (MICE):\n", df2)
print("\nChange after 2nd Iteration:\n", df2 - df1)

# ----------------------------------------------------------
# Iteration 3 (again, process continues until stable values)
# ----------------------------------------------------------
df3 = df2.copy()
df3.iloc[1,0] = np.NaN
X = df3.iloc[[0,2,3,4],1:3]
y = df3.iloc[[0,2,3,4],0]
lr = LinearRegression()
lr.fit(X,y)
df3.iloc[1,0] = lr.predict(df3.iloc[1,1:].values.reshape(1,2))[0]

df3.iloc[3,1] = np.NaN
X = df3.iloc[[0,1,2,4],[0,2]]
y = df3.iloc[[0,1,2,4],1]
lr = LinearRegression()
lr.fit(X,y)
df3.iloc[3,1] = lr.predict(df3.iloc[3,[0,2]].values.reshape(1,2))[0]

df3.iloc[4,-1] = np.NaN
X = df3.iloc[0:4,0:2]
y = df3.iloc[0:4,-1]
lr = LinearRegression()
lr.fit(X,y)
df3.iloc[4,-1] = lr.predict(df3.iloc[4,0:2].values.reshape(1,2))[0]

print("\nAfter 3rd Iteration (MICE):\n", df3)
print("\nChange after 3rd Iteration:\n", df3 - df2)

# ----------------------------------------------------------
# Conclusion:
# - With each iteration, imputed values become more refined.
# - The process continues until differences between iterations 
#   become negligible (convergence).
# - This demonstrates the essence of the MICE algorithm.
# ----------------------------------------------------------
