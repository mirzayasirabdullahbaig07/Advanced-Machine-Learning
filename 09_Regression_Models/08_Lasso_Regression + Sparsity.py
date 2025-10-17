# ==============================================================
#  LASSO REGRESSION (Least Absolute Shrinkage and Selection Operator)
# ==============================================================

# --------------------------------------------------------------
#  What is Lasso Regression?
# --------------------------------------------------------------
# Lasso Regression is a type of linear regression that adds L1 regularization
# to the cost function. This helps prevent overfitting and performs
# automatic feature selection by shrinking some coefficients to zero.
#
# Mathematically:
#     Cost Function = MSE + α * Σ|w|
# Where:
#     - MSE = Mean Squared Error
#     - α (alpha) = Regularization parameter
#     - w = model coefficients
#
# The term α * Σ|w| penalizes large coefficients, pushing less useful ones to zero.
# Hence, it introduces **sparsity** in the model.

# --------------------------------------------------------------
#  Imports
# --------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import Lasso, LinearRegression
from sklearn.datasets import make_regression, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from mlxtend.evaluate import bias_variance_decomp


# --------------------------------------------------------------
#  1. BASIC CONCEPT: Lasso vs Linear Regression
# --------------------------------------------------------------
# Generate a simple linear dataset with noise
X, y = make_regression(
    n_samples=100, n_features=1, n_informative=1,
    n_targets=1, noise=20, random_state=13
)

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Visualize dataset
plt.scatter(X, y)
plt.title("Sample Data Distribution")
plt.xlabel("Feature")
plt.ylabel("Target")
plt.show()

# Train a simple Linear Regression model (no regularization)
reg = LinearRegression()
reg.fit(X_train, y_train)
print("Coefficient:", reg.coef_)
print("Intercept:", reg.intercept_)

# Compare Lasso with different alpha values
alphas = [0, 1, 5, 10, 30]
plt.figure(figsize=(12, 6))
plt.scatter(X, y)
for i in alphas:
    L = Lasso(alpha=i)
    L.fit(X_train, y_train)
    plt.plot(X_test, L.predict(X_test), label=f'alpha={i}')
plt.legend()
plt.title("Lasso Regression with Different Alpha Values")
plt.show()

# --------------------------------------------------------------
#  Note:
# When alpha = 0, Lasso = Linear Regression
# But a warning appears since Lasso is not optimized for alpha=0.
# --------------------------------------------------------------


# --------------------------------------------------------------
#  2. Lasso with Polynomial Features (Non-linear case)
# --------------------------------------------------------------
# Generate a non-linear dataset
m = 100
x1 = 5 * np.random.rand(m, 1) - 2
x2 = 0.7 * x1**2 - 2 * x1 + 3 + np.random.randn(m, 1)

plt.scatter(x1, x2)
plt.title("Non-linear Relationship Example")
plt.show()

# Define a helper function to train Lasso with polynomial features
def get_preds_lasso(x1, x2, alpha):
    model = Pipeline([
        ('poly_feats', PolynomialFeatures(degree=16)),
        ('lasso', Lasso(alpha=alpha))
    ])
    model.fit(x1, x2)
    return model.predict(x1)

# Compare predictions for different alpha values
alphas = [0, 0.1, 1]
colors = ['r', 'g', 'b']

plt.figure(figsize=(10, 6))
plt.plot(x1, x2, 'b+', label='Data points')
for alpha, color in zip(alphas, colors):
    preds = get_preds_lasso(x1, x2, alpha)
    plt.plot(sorted(x1[:, 0]), preds[np.argsort(x1[:, 0])],
             color, label=f'Alpha: {alpha}')
plt.legend()
plt.title("Lasso on Polynomial Data (Effect of Alpha)")
plt.show()

# --------------------------------------------------------------
#  Key Point 1: How are Coefficients Affected?
# --------------------------------------------------------------
# Load the Diabetes dataset (a standard regression dataset)
data = load_diabetes()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['TARGET'] = data.target
print(df.head())

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=2)

# Compare coefficients for different alpha values
coefs = []
r2_scores = []

for alpha in [0, 0.1, 1, 10]:
    reg = Lasso(alpha=alpha)
    reg.fit(X_train, y_train)
    coefs.append(reg.coef_.tolist())
    y_pred = reg.predict(X_test)
    r2_scores.append(r2_score(y_test, y_pred))

# Plot coefficient shrinkage
plt.figure(figsize=(14, 9))
for i, alpha in enumerate([0, 0.1, 1, 10]):
    plt.subplot(2, 2, i+1)
    plt.bar(data.feature_names, coefs[i])
    plt.title(f'Alpha = {alpha}, R² = {round(r2_scores[i], 2)}')
plt.tight_layout()
plt.show()

# --------------------------------------------------------------
#  Key Point 2: Higher Coefficients Shrink More
# --------------------------------------------------------------
alphas = [0, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
coefs = []

for alpha in alphas:
    reg = Lasso(alpha=alpha)
    reg.fit(X_train, y_train)
    coefs.append(reg.coef_.tolist())

coef_df = pd.DataFrame(np.array(coefs), columns=data.feature_names, index=alphas)

# Visualize coefficient shrinkage trend
plt.figure(figsize=(15, 8))
plt.plot(alphas, np.zeros(len(alphas)), color='black', linewidth=4)
for feature in data.feature_names:
    plt.plot(alphas, coef_df[feature], label=feature)
plt.xscale('log')
plt.xlabel("Alpha (log scale)")
plt.ylabel("Coefficient Values")
plt.title("Lasso Coefficient Shrinkage with Increasing Alpha")
plt.legend()
plt.show()

# --------------------------------------------------------------
#  Key Point 3: Impact on Bias and Variance
# --------------------------------------------------------------
m = 100
X = 5 * np.random.rand(m, 1) - 2
y = 0.7 * X**2 - 2 * X + 3 + np.random.randn(m, 1)

plt.scatter(X, y)
plt.title("Dataset for Bias-Variance Demo")
plt.show()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Polynomial features
poly = PolynomialFeatures(degree=10)
X_train = poly.fit_transform(X_train)
X_test = poly.transform(X_test)

# Analyze bias-variance tradeoff for different alpha values
alphas = np.linspace(0, 30, 100)
loss, bias, variance = [], [], []

for alpha in alphas:
    reg = Lasso(alpha=alpha)
    avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(
        reg, X_train, y_train, X_test, y_test,
        loss='mse', random_seed=123
    )
    loss.append(avg_expected_loss)
    bias.append(avg_bias)
    variance.append(avg_var)

# Plot bias-variance tradeoff
plt.figure(figsize=(10, 6))
plt.plot(alphas, loss, label='Loss')
plt.plot(alphas, bias, label='Bias')
plt.plot(alphas, variance, label='Variance')
plt.xlabel('Alpha')
plt.title('Effect of Regularization on Bias & Variance')
plt.legend()
plt.show()

# --------------------------------------------------------------
#  Key Point 4: Effect of Regularization on Loss Function
# --------------------------------------------------------------
# Demonstration of how L1 penalty changes the loss landscape
X, y = make_regression(
    n_samples=100, n_features=1, n_informative=1,
    n_targets=1, noise=20, random_state=13
)
plt.scatter(X, y)
plt.title("Loss Function Demo Dataset")
plt.show()

reg = LinearRegression()
reg.fit(X, y)
print("Coefficient:", reg.coef_)
print("Intercept:", reg.intercept_)

# Define custom loss calculation with L1 penalty
def cal_loss(m, alpha):
    return np.sum((y - (m * X.ravel() + 2.29))**2) + alpha * abs(m)

m_values = np.linspace(-45, 100, 100)
plt.figure(figsize=(12, 10))

# Plot multiple loss curves for different alpha values
for alpha in [0, 100, 500, 1000, 2500, 3500, 4500, 5500]:
    loss_vals = [cal_loss(m, alpha) for m in m_values]
    plt.plot(m_values, loss_vals, label=f'alpha = {alpha}')

plt.xlabel('Coefficient (m)')
plt.ylabel('Loss')
plt.title('Effect of α on Lasso Loss Function')
plt.legend()
plt.show()

# --------------------------------------------------------------
#  LASSO SUMMARY
# --------------------------------------------------------------
#  Use Lasso when you have many features but expect some to be irrelevant.
#  It automatically performs feature selection (sparsity).
#  Helps reduce overfitting.
#  Large alpha → more shrinkage (higher bias, lower variance).
#  Small alpha → less shrinkage (lower bias, higher variance).
#  Choose alpha via cross-validation (e.g., LassoCV).

# --------------------------------------------------------------
# Difference between Ridge and Lasso
# --------------------------------------------------------------
# Ridge → L2 regularization → Σ(w²)
# Lasso → L1 regularization → Σ|w|
# Ridge shrinks coefficients but doesn’t make them zero.
# Lasso can shrink some coefficients exactly to zero → Feature selection.

# --------------------------------------------------------------
# When to Choose Lasso
# --------------------------------------------------------------
# Always choose Lasso when:
# - You have high-dimensional data
# - You suspect some features are irrelevant
# - You want a simpler, more interpretable model
# --------------------------------------------------------------

# ===============================================================
# MATHS BEHIND LASSO REGRESSION
# ===============================================================

# ---------------------------------------------------------------
# Lasso Loss Function
# ---------------------------------------------------------------
# The loss function of Lasso regression combines:
#     Mean Squared Error (MSE) + L1 Regularization (absolute weight penalty)
#
# Formula:
#     L = Σ (y_i - ŷ_i)^2 + λ * |m|
#
# where:
#     y_i   → actual value
#     ŷ_i   → predicted value (m * x_i + b)
#     m     → slope or model coefficient
#     λ     → regularization parameter (controls penalty strength)
#
# The key part here is the absolute value |m| which makes Lasso special.
# It introduces *non-differentiability* at zero, which is the source of sparsity.

# ---------------------------------------------------------------
# Differentiation of Lasso Loss
# ---------------------------------------------------------------
# Let's take the derivative of L with respect to m (the slope):
#
#     dL/dm = -2 Σ (y_i - m*x_i) * x_i + λ * sign(m)
#
# where:
#     sign(m) =
#         +1  if m > 0
#         -1  if m < 0
#         ∈ [-1, +1]  if m = 0  (subgradient case)
#
# Because of the |m| term, Lasso is not differentiable at m = 0,
# which creates the "stopping" effect that leads to sparsity.

# ---------------------------------------------------------------
# Solving for m (Coefficient)
# ---------------------------------------------------------------
# From the derivation (as shown in your image):
#
#     m = [ Σ(y_i - ȳ)(x_i - x̄) - λ ] / Σ(x_i - x̄)²    when m > 0
#     m = [ Σ(y_i - ȳ)(x_i - x̄) + λ ] / Σ(x_i - x̄)²    when m < 0
#
# For m = 0, we stop updating m when the absolute correlation between
# x and y is smaller than λ (i.e., |S| ≤ λ).

# ---------------------------------------------------------------
# Soft Thresholding Rule (Final Form)
# ---------------------------------------------------------------
# Define:
#     S = Σ(y_i - ȳ)(x_i - x̄)
#     D = Σ(x_i - x̄)²
#
# Then Lasso solution for m is:
#
#     m =
#         (S - λ) / D    if S > λ
#         (S + λ) / D    if S < -λ
#         0              if |S| ≤ λ
#
# This is called the **soft thresholding operator**.
# It smoothly shrinks coefficients toward zero, and stops them exactly at zero
# if they are not strong enough (|S| ≤ λ).

# ---------------------------------------------------------------
# Why Does Lasso Stop at Zero?
# ---------------------------------------------------------------
# The L1 penalty λ * |m| adds a constant "pull" toward zero.
# When the strength of the data (represented by S) is weaker than λ,
# the cost of increasing m outweighs the gain in reducing error.
# Therefore, the best choice is m = 0 (no change).
#
# This is why Lasso "stops" coefficients at zero — it’s mathematically optimal.

# ---------------------------------------------------------------
# Sparsity in Lasso
# ---------------------------------------------------------------
# Sparsity means: some coefficients (weights) become exactly 0.
# So the model automatically performs feature selection.
#
# Geometric intuition:
#   - Lasso uses L1 regularization → forms a diamond-shaped constraint region.
#   - The MSE error contours are elliptical.
#   - The ellipse tends to touch the diamond’s *corners*, where one or more
#     coefficients are exactly zero.
#
# Therefore, Lasso promotes *sparse models* — models with only a few
# non-zero coefficients.

# ---------------------------------------------------------------
# Why Ridge Regression Cannot Produce Sparsity
# ---------------------------------------------------------------
# Ridge regression uses an L2 penalty (λ * m²).
# This penalty is smooth and differentiable everywhere, so coefficients
# are only *shrunk* but never exactly zero.
#
# Comparison:
#   Ridge → Continuous shrinkage, no feature removal.
#   Lasso → Shrinkage + exact zero coefficients (feature selection).

# ---------------------------------------------------------------
#  Summary
# ---------------------------------------------------------------
# Case-wise explanation of coefficient behavior:
#
#   if m > 0:     m = (S - λ) / D        → shrinks toward 0
#   if m < 0:     m = (S + λ) / D        → shrinks toward 0
#   if m = 0:     when |S| ≤ λ           → coefficient becomes 0
#
#  Lasso automatically selects features by setting some weights to zero.
#  Ridge never reaches zero (only reduces magnitude).
#  The amount of shrinkage depends on λ.
#
# In short:
#   Lasso = Linear Regression + L1 penalty = Simple, sparse, and interpretable.
