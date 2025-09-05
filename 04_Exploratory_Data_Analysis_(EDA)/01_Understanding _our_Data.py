# Topic 1 Understanding YOUR DATA

import pandas as pd

# Load dataset
df = pd.read_csv('train.csv')

# 1. Size of the data (rows, columns)
print("Shape of dataset:", df.shape)

# 2. How does the data look like?
print("\nFirst 5 rows of data:")
print(df.head())

print("\nRandom 5 sample rows:")
print(df.sample(5))

# 3. Data types of columns
print("\nInfo of dataset:")
print(df.info())

# 4. Missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# 5. Mathematical summary of data
print("\nStatistical description of data:")
print(df.describe())

# 6. Are there any duplicate values?
print("\nNumber of duplicate rows:", df.duplicated().sum())

# 7. Correlation between numerical columns
print("\nCorrelation matrix:")
print(df.corr(numeric_only=True))

# Correlation with target variable (example: 'Survived')
if 'Survived' in df.columns:
    print("\nCorrelation of features with 'Survived':")
    print(df.corr(numeric_only=True)['Survived'].sort_values(ascending=False))
