# ===========================================================
# RIDGE REGRESSION (with Full Explanation + Implementation)
# ===========================================================

# Objective:
# -------------
# Understand what regularization is, how Ridge Regression helps
# prevent overfitting, and implement it using sklearn.

# ------------------------------------------------------------
# WHAT IS OVERFITTING?
# ------------------------------------------------------------
# Overfitting occurs when a model performs very well on the training data
# but fails to generalize on unseen (testing) data.

# Example:
# Your model memorizes patterns, noise, and outliers from the training data
# instead of learning the true underlying relationship.

# Model Performance:
# - Training accuracy → High
# - Testing accuracy  → Low

# ------------------------------------------------------------
# WHAT IS REGULARIZATION?
# ------------------------------------------------------------
# Regularization is a technique used to reduce **overfitting** by
# adding a penalty term to the loss function.

# It keeps model coefficients small and avoids extreme weights.

# Regularization modifies the cost function by adding a term
# that penalizes large parameter values.

# ------------------------------------------------------------
# TYPES OF REGULARIZATION
# ------------------------------------------------------------

#  **Ridge Regression (L2 Regularization)**
#    ➤ Adds squared magnitude of coefficients as penalty term.
#    ➤ Penalizes large weights but keeps all features.
#    ➤ Formula:
#       Cost = RSS + α * Σ(wᵢ²)
#       where α (lambda) is the regularization strength.

#  **Lasso Regression (L1 Regularization)**
#    ➤ Adds absolute value of coefficients as penalty term.
#    ➤ Can shrink some coefficients to 0 (feature selection).
#    ➤ Formula:
#       Cost = RSS + α * Σ(|wᵢ|)

#  **Elastic Net (Combination of L1 + L2)**
#    ➤ Combines Ridge and Lasso benefits.
#    ➤ Formula:
#       Cost = RSS + α₁ * Σ(wᵢ²) + α₂ * Σ(|wᵢ|)

# ------------------------------------------------------------
# WHY REGULARIZATION IS USEFUL?
# ------------------------------------------------------------
# Prevents overfitting.
# Improves generalization on test data.
# Reduces model variance.
# Handles multicollinearity (high correlation among features).
# Keeps coefficients stable.

# ------------------------------------------------------------
# MATHEMATICAL INTUITION (RIDGE)
# ------------------------------------------------------------
# We minimize the following function:

#     J(w) = Σ(yᵢ - ŷᵢ)² + α * Σ(wⱼ²)

# Where:
# - yᵢ → actual values
# - ŷᵢ → predicted values
# - α  → regularization parameter (controls penalty)
# - wⱼ → model coefficients

# As α increases → coefficients shrink → model becomes simpler.

# If α = 0 → same as Linear Regression (no regularization).

# ------------------------------------------------------------
# REAL-WORLD USE CASE
# ------------------------------------------------------------
# Ridge Regression is useful when:
# - We have many correlated features.
# - Model is overfitting.
# - We need to reduce coefficient magnitude.

# Example industries:
# Finance → Predicting stock returns
# Healthcare → Disease progression models
# Real Estate → House price predictions
# ------------------------------------------------------------


# ============================================================
# IMPORT LIBRARIES
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures


# ============================================================
# LOAD DIABETES DATASET
# ============================================================

data = load_diabetes()
X = data.data
y = data.target

print(data.DESCR)  # Dataset description

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=45)


# ============================================================
# 1. LINEAR REGRESSION (BASE MODEL)
# ============================================================

L = LinearRegression()
L.fit(X_train, y_train)

print("\n📊 Linear Regression Coefficients:\n", L.coef_)
print("Intercept:", L.intercept_)

# Predictions
y_pred = L.predict(X_test)

# Evaluation
print("\nLinear Regression Results:")
print("R2 Score:", r2_score(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))


# ============================================================
# 2. RIDGE REGRESSION (REGULARIZED MODEL)
# ============================================================

R = Ridge(alpha=100000)  # alpha controls regularization strength
R.fit(X_train, y_train)

print("\n📊 Ridge Regression Coefficients:\n", R.coef_)
print("Intercept:", R.intercept_)

# Predictions
y_pred1 = R.predict(X_test)

# Evaluation
print("\nRidge Regression Results:")
print("R2 Score:", r2_score(y_test, y_pred1))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred1)))


# ============================================================
# VISUALIZATION: EFFECT OF ALPHA ON POLYNOMIAL RIDGE MODEL
# ============================================================

# Generate some synthetic non-linear data
m = 100
x1 = 5 * np.random.rand(m, 1) - 2
x2 = 0.7 * x1 ** 2 - 2 * x1 + 3 + np.random.randn(m, 1)

plt.scatter(x1, x2, color='blue', label='Original Data')
plt.title("Generated Non-linear Data")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.show()


# Function to fit polynomial Ridge model and return predictions
def get_preds_ridge(x1, x2, alpha):
    model = Pipeline([
        ('poly_feats', PolynomialFeatures(degree=16)),
        ('ridge', Ridge(alpha=alpha))
    ])
    model.fit(x1, x2)
    return model.predict(x1)


# Try different alpha values
alphas = [0, 20, 200]
colors = ['r', 'g', 'b']

plt.figure(figsize=(10, 6))
plt.scatter(x1, x2, color='black', label='Data Points')

for alpha, c in zip(alphas, colors):
    preds = get_preds_ridge(x1, x2, alpha)
    plt.plot(sorted(x1[:, 0]),
             preds[np.argsort(x1[:, 0])],
             c,
             label=f'Alpha = {alpha}')

plt.title("Effect of Regularization Strength (Alpha) on Ridge Model")
plt.xlabel("x1")
plt.ylabel("Predicted x2")
plt.legend()
plt.show()

# ===========================================================
# OBSERVATIONS:
# ===========================================================

# Alpha = 0 → behaves like normal Linear Regression (can overfit).
# Alpha = 20 → moderate penalty → smoother curve → better generalization.
# Alpha = 200 → strong penalty → underfitting (too smooth).

# ------------------------------------------------------------
# Summary:
# ------------------------------------------------------------
# Linear Regression → No regularization → may overfit.
# Ridge Regression → Penalizes large weights → reduces overfitting.
# Higher alpha → simpler model → less variance → possibly more bias.

# ------------------------------------------------------------
# Key Idea:
# ------------------------------------------------------------
# Ridge doesn’t eliminate features — it shrinks coefficients smoothly.
# Lasso can remove (zero out) features → good for feature selection.

# ------------------------------------------------------------
# Use Ridge when:
# ------------------------------------------------------------
# - You have correlated predictors
# - You want to prevent overfitting
# - You don’t want to eliminate any feature
# ------------------------------------------------------------


# ================================================================
#               Ridge Regression — Math & Intuition
# ================================================================

# Regularization is a technique used to reduce overfitting.
# Ridge Regression (L2 Regularization) penalizes large weight coefficients
# by adding a term λ * Σ(w_j^2) to the cost function.

# ----------------------------------------------------------------
# 1. Linear Regression (Without Regularization)
# ----------------------------------------------------------------
# Objective: minimize Mean Squared Error (MSE)
#
# J(w, b) = (1 / 2m) * Σ(y_i - (w^T x_i + b))^2
#
# Derivation gives:
# w = (X^T X)^(-1) X^T y
#
# Problem: If features are correlated (multicollinearity),
# then X^T X becomes nearly singular (non-invertible),
# causing unstable or large coefficient estimates → overfitting.

# ----------------------------------------------------------------
# 2. Ridge Regression (With Regularization)
# ----------------------------------------------------------------
# To overcome overfitting, we modify the cost function by adding
# a penalty term for large coefficients (weights).

# New cost function:
#
# J_ridge(w, b) = (1 / 2m) * Σ(y_i - (w^T x_i + b))^2 + (λ / 2m) * Σ(w_j^2)
#
# where:
# λ (alpha) = regularization strength
# w_j^2 = L2 penalty (squared coefficients)
#
# The first term ensures good fit to the data,
# the second term penalizes model complexity (large weights).

# ----------------------------------------------------------------
# 3. Mathematical Derivation
# ----------------------------------------------------------------
# To minimize J(w), take the derivative with respect to w and set to 0:
#
# ∂J/∂w = -X^T(y - Xw) + λw = 0
#
# => X^T X w + λI w = X^T y
#
# Solving for w:
#
# w = (X^T X + λI)^(-1) X^T y
#
# This is the Ridge Regression solution.
#
# Note:
#   - I is the identity matrix (bias not penalized)
#   - (X^T X + λI) is always invertible, even if X^T X isn’t

# ----------------------------------------------------------------
# 4. Effect of λ (Alpha)
# ----------------------------------------------------------------
# λ = 0     → behaves like Linear Regression (no penalty)
# λ small   → small shrinkage (slightly reduces overfitting)
# λ large   → large shrinkage (coefficients move toward zero, underfitting risk)

# ----------------------------------------------------------------
# 5. Ridge vs Linear Regression
# ----------------------------------------------------------------
# Linear Regression:
#   w = (X^T X)^(-1) X^T y
# Ridge Regression:
#   w = (X^T X + λI)^(-1) X^T y
#
# The +λI term stabilizes matrix inversion and controls model complexity.

# ----------------------------------------------------------------
# 6. 1D Ridge Regression Formula
# ----------------------------------------------------------------
# For a single feature (simple regression):
#
# m = Σ(x_i - x̄)(y_i - ȳ) / [Σ(x_i - x̄)^2 + λ]
# b = ȳ - m * x̄
#
# The α (lambda) is directly added to the denominator to shrink the slope.

# ----------------------------------------------------------------
# 7. Implementation Example
# ----------------------------------------------------------------

from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
import numpy as np

# Generate synthetic regression data with noise
X, y = make_regression(n_samples=100, n_features=1, noise=20, random_state=13)

plt.scatter(X, y)
plt.title("Generated Data")
plt.show()

# ----------------------------------------------------------------
# Linear Regression (No Regularization)
# ----------------------------------------------------------------
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X, y)

print("Linear Regression Coefficients:", lr.coef_)
print("Intercept:", lr.intercept_)

# ----------------------------------------------------------------
# Ridge Regression (α = 10 and α = 100)
# ----------------------------------------------------------------
from sklearn.linear_model import Ridge

rr_10 = Ridge(alpha=10)
rr_10.fit(X, y)

rr_100 = Ridge(alpha=100)
rr_100.fit(X, y)

print("Ridge (α=10) Coefficients:", rr_10.coef_)
print("Ridge (α=100) Coefficients:", rr_100.coef_)

# ----------------------------------------------------------------
# Visual Comparison
# ----------------------------------------------------------------
plt.scatter(X, y, color='blue', label='Data points')
plt.plot(X, lr.predict(X), color='red', label='Linear Regression (α=0)')
plt.plot(X, rr_10.predict(X), color='green', label='Ridge α=10')
plt.plot(X, rr_100.predict(X), color='orange', label='Ridge α=100')
plt.legend()
plt.title("Effect of Ridge Regularization on Regression Line")
plt.show()

# ----------------------------------------------------------------
# 8. Custom Ridge Implementation (Manual)
# ----------------------------------------------------------------
def linear_regression(X, y, alpha=1):
    """
    Implements simple Ridge Regression manually (1D feature)
    """
    x_mean = X.mean()
    y_mean = y.mean()

    num = 0  # numerator
    den = 0  # denominator

    for i in range(X.shape[0]):
        num += (y[i] - y_mean) * (X[i] - x_mean)
        den += (X[i] - x_mean) ** 2

    # Ridge penalty applied in denominator
    m = num / (den + alpha)
    b = y_mean - m * x_mean

    return m, b


class MeraRidge:
    """
    Custom Ridge Regression (1D)
    """

    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.m = None
        self.b = None

    def fit(self, X_train, y_train):
        num = 0
        den = 0

        for i in range(X_train.shape[0]):
            num += (y_train[i] - y_train.mean()) * (X_train[i] - X_train.mean())
            den += (X_train[i] - X_train.mean()) ** 2

        # Apply ridge regularization
        self.m = num / (den + self.alpha)
        self.b = y_train.mean() - self.m * X_train.mean()

        print(f"Slope (m): {self.m}")
        print(f"Intercept (b): {self.b}")

    def predict(self, X_test):
        return self.m * X_test + self.b


# Instantiate and train custom Ridge model
reg = MeraRidge(alpha=100)
reg.fit(X, y)

# ----------------------------------------------------------------
# 9. Key Takeaways
# ----------------------------------------------------------------
# - Ridge Regression controls overfitting by penalizing large coefficients.
# - It does not eliminate features (unlike Lasso), but reduces their effect.
# - λ (alpha) is the tuning parameter controlling shrinkage strength.
# - Ideal when features are correlated (multicollinearity).
# - Formula adds λI to make (X^T X) invertible and more stable.

 