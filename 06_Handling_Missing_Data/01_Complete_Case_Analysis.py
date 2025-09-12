# ----------------------------------------
# Complete Case Analysis (CCA)
# ----------------------------------------
# Missing data is not handled by sklearn by default.
# So, we must handle it manually.
# One common method: remove rows with missing values (Complete Case Analysis).

# ----------------------------------------
# What is Complete Case Analysis?
# ----------------------------------------
# - CCA (also called list-wise deletion) means discarding rows where 
#   any variable has missing values.
# - Only observations with full information across all variables are analyzed.

# Assumption for CCA:
# - Data is MCAR (Missing Completely At Random).

# Advantages:
# - Easy to implement (no imputation needed).
# - Preserves variable distribution (if MCAR, reduced dataset should reflect original).

# Disadvantages:
# - Can exclude large fraction of dataset if missing data is abundant.
# - Dropped observations may contain useful information (if data is not MCAR).
# - Model in production won’t know how to handle missing data.

# When to use CCA?
# - If data is MCAR.
# - If missing data < 5% (otherwise avoid).

# ----------------------------------------
# Example with dataset
# ----------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('data_science_job.csv')

# First look
print(df.head())
print(df.isnull().mean() * 100)   # % missing per column
print("Original Shape:", df.shape)

# Select columns with < 5% missing values
cols = [var for var in df.columns if 0 < df[var].isnull().mean() < 0.05]
print("Columns with <5% missing values:", cols)

# Sample of those columns
print(df[cols].sample(5))

# Check distribution before dropping
print("Education Levels:\n", df['education_level'].value_counts())

# Fraction of data retained after CCA
retained_fraction = len(df[cols].dropna()) / len(df)
print("Fraction of data retained after CCA:", retained_fraction)

# Apply CCA
new_df = df[cols].dropna()
print("Original shape:", df.shape, " -> After CCA:", new_df.shape)

# ----------------------------------------
# Compare Distributions
# ----------------------------------------

# Training hours (hist + density)
fig, ax = plt.subplots()
df['training_hours'].hist(bins=50, ax=ax, density=True, color='red')
new_df['training_hours'].hist(bins=50, ax=ax, density=True, color='green', alpha=0.8)
plt.title("Training Hours Distribution: Original vs CCA")
plt.show()

fig, ax = plt.subplots()
df['training_hours'].plot.density(color='red', ax=ax)
new_df['training_hours'].plot.density(color='green', ax=ax)
plt.title("Training Hours Density: Original vs CCA")
plt.show()

# City Development Index
fig, ax = plt.subplots()
df['city_development_index'].hist(bins=50, ax=ax, density=True, color='red')
new_df['city_development_index'].hist(bins=50, ax=ax, density=True, color='green', alpha=0.8)
plt.title("City Development Index: Original vs CCA")
plt.show()

fig, ax = plt.subplots()
df['city_development_index'].plot.density(color='red', ax=ax)
new_df['city_development_index'].plot.density(color='green', ax=ax)
plt.title("City Development Index Density: Original vs CCA")
plt.show()

# Experience
fig, ax = plt.subplots()
df['experience'].hist(bins=50, ax=ax, density=True, color='red')
new_df['experience'].hist(bins=50, ax=ax, density=True, color='green', alpha=0.8)
plt.title("Experience: Original vs CCA")
plt.show()

fig, ax = plt.subplots()
df['experience'].plot.density(color='red', ax=ax)
new_df['experience'].plot.density(color='green', ax=ax)
plt.title("Experience Density: Original vs CCA")
plt.show()

# ----------------------------------------
# Compare Categorical Distributions
# ----------------------------------------
# Enrolled University
temp = pd.concat([
    df['enrolled_university'].value_counts() / len(df),
    new_df['enrolled_university'].value_counts() / len(new_df)
], axis=1)
temp.columns = ['original', 'cca']
print("\nEnrolled University Distribution:\n", temp)

# Education Level
temp = pd.concat([
    df['education_level'].value_counts() / len(df),
    new_df['education_level'].value_counts() / len(new_df)
], axis=1)
temp.columns = ['original', 'cca']
print("\nEducation Level Distribution:\n", temp)
