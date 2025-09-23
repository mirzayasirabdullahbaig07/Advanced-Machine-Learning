# ============================================================
#        FEATURE CONSTRUCTION | FEATURE SPLITTING
# ============================================================

# -------------------------------
# Why We Use Feature Construction?
# -------------------------------
# - To create new features from existing ones that better
#   represent the underlying problem.
# - Helps models capture hidden patterns.

# -------------------------------
# Importance & Benefits
# -------------------------------
# 1. Improves model accuracy by providing meaningful inputs.
# 2. Helps in reducing noise in raw features.
# 3. Converts raw data into more interpretable form.
# 4. Enables models to learn complex relationships.

# -------------------------------
# When to Use Feature Construction?
# -------------------------------
# - When existing features are not sufficient to capture
#   relationships in data.
# - When domain knowledge suggests useful transformations.
# - When we want to reduce dimensionality but still keep
#   useful information.

# ============================================================
#                 IMPLEMENTATION IN PYTHON
# ============================================================

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

# Load dataset (Titanic dataset for example)
df = pd.read_csv('train.csv')[['Age','Pclass','SibSp','Parch','Survived']]
print(df.head())

# Drop missing values
df.dropna(inplace=True)

# Features and target
X = df.iloc[:,0:4]
y = df.iloc[:,-1]

print("Baseline Accuracy:", 
      np.mean(cross_val_score(LogisticRegression(), X, y, scoring='accuracy', cv=20)))

# ============================================================
#             FEATURE CONSTRUCTION EXAMPLE
# ============================================================

# Construct new feature: Family Size
X['Family_size'] = X['SibSp'] + X['Parch'] + 1
print(X.head())

# Create categorical feature from Family Size
def family_type(num):
    if num == 1:
        return 0   # Alone
    elif num > 1 and num <= 4:
        return 1   # Small Family
    else:
        return 2   # Large Family

X['Family_type'] = X['Family_size'].apply(family_type)
print(X.head())

# Drop raw features after construction
X.drop(columns=['SibSp','Parch','Family_size'], inplace=True)
print(X.head())

# ============================================================
#             FEATURE SPLITTING EXAMPLE
# ============================================================

df = pd.read_csv('train.csv')
print(df[['Name','Survived']].head())

# Extract Title from Name
df['Title'] = df['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]
print(df[['Title','Name']].head())

# Check survival rate by Title
print((df.groupby('Title').mean()['Survived']).sort_values(ascending=False))

# Construct binary marital feature
df['Is_Married'] = 0
df.loc[df['Title'] == 'Mrs', 'Is_Married'] = 1
print(df[['Title','Is_Married']].head())


