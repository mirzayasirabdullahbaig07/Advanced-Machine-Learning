
# Power Transformer (Box–Cox, Yeo–Johnson) Demonstration

# This script explains and demonstrates:
# 1. Box–Cox Transformation
#    - Works only for strictly positive data.
#    - Searches λ (lambda) between -5 to 5.
#    - Uses Maximum Likelihood Estimation (MLE) to find the best λ.
#    - Cannot handle 0 or negative values.

# 2. Yeo–Johnson Transformation
#    - Works for positive, negative, and zero values.
#    - Extension of Box–Cox.
#    - Useful when data contains zeros or negatives.

# Steps:
# - Load dataset (Concrete dataset as example).
# - Check distributions before transformation.
# - Apply Box–Cox & Yeo–Johnson.
# - Visualize changes in distributions and Q-Q plots.
# - Apply transformation to the whole dataset.
# - Compare model performance before & after transformation.


# =========================
# 1. Import Libraries
# =========================
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import r2_score

# =========================
# 2. Load Dataset
# =========================
df = pd.read_csv("concetete_data.csv")

print("Dataset Shape:", df.shape)
print(df.head())
print(df.isnull().sum())
print(df.describe())

# =========================
# 3. Visualize Original Distribution
# =========================
feature = "cement"  # choose one column from dataset

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
sns.histplot(df[feature], kde=True)
plt.title(f"Original Distribution: {feature}")

plt.subplot(1,2,2)
stats.probplot(df[feature], dist="norm", plot=plt)
plt.title("Q-Q Plot Before Transformation")
plt.show()

# =========================
# 4. Box–Cox Transformation
# =========================
# Only positive values allowed
positive_data = df[feature][df[feature] > 0]

boxcox_transformed, best_lambda = stats.boxcox(positive_data)
print("\n[Box–Cox] Best λ (lambda):", best_lambda)

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
sns.histplot(boxcox_transformed, kde=True)
plt.title(f"After Box–Cox Transformation (λ={best_lambda:.4f})")

plt.subplot(1,2,2)
stats.probplot(boxcox_transformed, dist="norm", plot=plt)
plt.title("Q-Q Plot After Box–Cox")
plt.show()

# =========================
# 5. Yeo–Johnson Transformation
# =========================
pt = PowerTransformer(method="yeo-johnson")
yeo_transformed = pt.fit_transform(df[[feature]])

print("[Yeo–Johnson] Optimal λ:", pt.lambdas_[0])

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
sns.histplot(yeo_transformed.flatten(), kde=True)
plt.title("After Yeo–Johnson Transformation")

plt.subplot(1,2,2)
stats.probplot(yeo_transformed.flatten(), dist="norm", plot=plt)
plt.title("Q-Q Plot After Yeo–Johnson")
plt.show()

# =========================
# 6. Apply Yeo–Johnson to Entire Dataset
# =========================
df_transformed = df.copy()
pt_full = PowerTransformer(method="yeo-johnson")
df_transformed[:] = pt_full.fit_transform(df)

print("\nLambdas for each feature:")
print(pt_full.lambdas_)

# Compare one feature before & after
sns.kdeplot(df[feature], label="Original")
sns.kdeplot(df_transformed[feature], label="Transformed")
plt.title(f"Distribution Comparison for {feature}")
plt.legend()
plt.show()

# =========================
# 7. Model Performance Check
# =========================
# Assume "strength" is target column
X = df.drop("strength", axis=1)
y = df["strength"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model without transformation
model1 = LinearRegression()
model1.fit(X_train, y_train)
y_pred1 = model1.predict(X_test)
print("\nModel R2 score (before transform):", r2_score(y_test, y_pred1))

# Apply Yeo–Johnson to features
pt_model = PowerTransformer(method="yeo-johnson")
X_train_trans = pt_model.fit_transform(X_train)
X_test_trans = pt_model.transform(X_test)

model2 = LinearRegression()
model2.fit(X_train_trans, y_train)
y_pred2 = model2.predict(X_test_trans)
print("Model R2 score (after transform):", r2_score(y_test, y_pred2))

# =========================
# END
# =========================
