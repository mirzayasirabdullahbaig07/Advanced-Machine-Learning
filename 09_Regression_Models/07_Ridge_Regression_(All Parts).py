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


# ==================================================================================
# RIDGE REGRESSION USING GRADIENT DESCENT
#=================================================================================
# What is Ridge Regression?

# Ridge Regression is a type of **Linear Regression** that includes **L2 Regularization**.
# It adds a penalty (α * sum of squared coefficients) to the loss function to prevent overfitting.
#
# It helps control large coefficients by shrinking them toward zero, 
#    making the model more stable and less sensitive to noise.
#
# ----------------------------------------------------------------------------------
#When to Use Ridge Regression?

# • When your data shows **multicollinearity** (features are highly correlated).
# • When your model overfits on the training data.
# • When you want a **simpler, smoother** model that generalizes better.
# ----------------------------------------------------------------------------------------------------
#
# Mathematical Formula
# ----------------------------------------------------------------------------------------------------
# Ordinary Linear Regression tries to minimize:
#       J(θ) = (1/2m) * Σ (yᵢ - Xᵢθ)²
#
# Ridge Regression adds a regularization term:
#       J(θ) = (1/2m) * [Σ (yᵢ - Xᵢθ)² + α * Σ θⱼ²]
#
# Where:
#   • m = number of samples
#   • α = regularization parameter (controls strength of penalty)
#   • θ = coefficients (weights)
#
# The gradient for Ridge Regression:
#       ∇J(θ) = (1/m) * [ Xᵀ(Xθ - y) + αθ ]
#
# Gradient Descent update rule:
#       θ_new = θ_old - η * ∇J(θ)
#
# Where:
#   • η = learning rate (step size)
#
# ----------------------------------------------------------------------------------------------------
# Benefits
# ----------------------------------------------------------------------------------------------------
# • Reduces model complexity.
# • Prevents overfitting.
# • Works well with multicollinear data.
# • Produces stable coefficient estimates.
#
# ----------------------------------------------------------------------------------------------------
# Disadvantages
# ----------------------------------------------------------------------------------------------------
# • It does not perform feature selection (unlike Lasso).
# • α (regularization strength) must be tuned carefully.
# • Can underfit if α is too large.
# ----------------------------------------------------------------------------------------------------
#
# Real-World Examples
# ----------------------------------------------------------------------------------------------------
# • Predicting house prices with correlated features (area, rooms, location, etc.)
# • Medical data analysis (e.g., diabetes prediction)
# • Economic forecasting
# ----------------------------------------------------------------------------------------------------
#
#Let’s implement it step-by-step and compare:
#   Ridge Regression via SGDRegressor
#   Ridge Regression via sklearn.Ridge
#   Custom Ridge Regression using Gradient Descent (MeraRidgeGD)
# ----------------------------------------------------------------------------------------------------


# ==================================================================================
#  Using SGDRegressor with L2 Regularization
# ==================================================================================

from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor, Ridge

# Load the diabetes dataset
X, y = load_diabetes(return_X_y=True)

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

# Define a Ridge-type model using SGDRegressor
reg = SGDRegressor(
    penalty='l2',          # L2 regularization = Ridge
    max_iter=500,          # number of epochs
    eta0=0.1,              # learning rate
    learning_rate='constant',
    alpha=0.001            # regularization strength
)

# Train the model
reg.fit(X_train, y_train)

# Predict test data
y_pred = reg.predict(X_test)

# Evaluate the model
print("R2 score:", r2_score(y_test, y_pred))
print("Coefficients:", reg.coef_)
print("Intercept:", reg.intercept_)

# Example Output:
# R2 score 0.4408


# ==================================================================================
# Using Ridge Regression from sklearn
# ==================================================================================

reg = Ridge(alpha=0.001, max_iter=500, solver='sparse_cg')
reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)
print("R2 score:", r2_score(y_test, y_pred))
print("Coefficients:", reg.coef_)
print("Intercept:", reg.intercept_)

# Example Output:
# R2 score 0.4623

# ====================================================================================================
# Custom Implementation: MeraRidgeGD (Ridge Regression using Gradient Descent)
# ====================================================================================================

class MeraRidgeGD:
    """
    Custom Ridge Regression Implementation using Gradient Descent
    -------------------------------------------------------------
    This model minimizes the cost function:
        J(θ) = (1/2m) * [ (y - Xθ)² + α * ||θ||² ]
    
    Using the update rule:
        θ = θ - η * [ (XᵀXθ - Xᵀy) + αθ ]
    """

    def __init__(self, epochs, learning_rate, alpha):
        self.learning_rate = learning_rate  # η
        self.epochs = epochs                # number of iterations
        self.alpha = alpha                  # regularization parameter
        self.coef_ = None                   # model weights
        self.intercept_ = None              # bias term

    def fit(self, X_train, y_train):
        # Initialize weights (θ) and intercept (bias)
        self.coef_ = np.ones(X_train.shape[1])
        self.intercept_ = 0

        # Combine intercept and weights into one vector
        theta = np.insert(self.coef_, 0, self.intercept_)

        # Add bias column (1s) to training data
        X_train = np.insert(X_train, 0, 1, axis=1)

        # Gradient Descent loop
        for i in range(self.epochs):
            # Compute the gradient: ∇J(θ)
            theta_der = np.dot(X_train.T, X_train).dot(theta) - np.dot(X_train.T, y_train) + self.alpha * theta

            # Update parameters
            theta = theta - self.learning_rate * theta_der

        # Extract final weights and bias
        self.coef_ = theta[1:]
        self.intercept_ = theta[0]

    def predict(self, X_test):
        # Prediction formula: y_pred = Xw + b
        return np.dot(X_test, self.coef_) + self.intercept_


# Train custom Ridge Regression
reg = MeraRidgeGD(epochs=500, alpha=0.001, learning_rate=0.005)
reg.fit(X_train, y_train)

# Predict test set
y_pred = reg.predict(X_test)

# Evaluate performance
print("R2 score:", r2_score(y_test, y_pred))
print("Coefficients:", reg.coef_)
print("Intercept:", reg.intercept_)

# Example Output:
# R2 score 0.4737


# ====================================================================================================
# Summary and Comparison
# ----------------------------------------------------------------------------------------------------
# • SGDRegressor (L2):         R² ≈ 0.44
# • Ridge (sklearn):            R² ≈ 0.46
# • Custom MeraRidgeGD:         R² ≈ 0.47
#
# Our custom Gradient Descent Ridge performed slightly better due to optimized weight updates.
# Regularization (α) helped control large weights and improved generalization.
# ----------------------------------------------------------------------------------------------------
# Key Takeaways
# ----------------------------------------------------------------------------------------------------
# • Ridge Regression = Linear Regression + L2 Penalty
# • Use when you have multicollinearity or overfitting
# • Gradient Descent can efficiently learn weights when dataset is large
# • α (regularization parameter) must be tuned carefully

 