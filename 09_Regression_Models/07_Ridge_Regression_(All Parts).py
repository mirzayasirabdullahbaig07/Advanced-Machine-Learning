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

# RIDGE REGRESSION USING GRADIENT DESCENT

# What is Ridge Regression?

# Ridge Regression is a type of **Linear Regression** that includes **L2 Regularization**.
# It adds a penalty (α * sum of squared coefficients) to the loss function to prevent overfitting.

# It helps control large coefficients by shrinking them toward zero, making the model more stable and less sensitive to noise.

#When to Use Ridge Regression?

# • When your data shows **multicollinearity** (features are highly correlated).
# • When your model overfits on the training data.
# • When you want a **simpler, smoother** model that generalizes better.

# Mathematical Formula

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

# Benefits

# • Reduces model complexity.
# • Prevents overfitting.
# • Works well with multicollinear data.
# • Produces stable coefficient estimates.

# Disadvantages

# • It does not perform feature selection (unlike Lasso).
# • α (regularization strength) must be tuned carefully.
# • Can underfit if α is too large.

# Real-World Examples

# • Predicting house prices with correlated features (area, rooms, location, etc.)
# • Medical data analysis (e.g., diabetes prediction)
# • Economic forecasting

#Let’s implement it step-by-step and compare:
#   Ridge Regression via SGDRegressor
#   Ridge Regression via sklearn.Ridge
#   Custom Ridge Regression using Gradient Descent (MeraRidgeGD)

#  Using SGDRegressor with L2 Regularization

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

# Using Ridge Regression from sklearn

reg = Ridge(alpha=0.001, max_iter=500, solver='sparse_cg')
reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)
print("R2 score:", r2_score(y_test, y_pred))
print("Coefficients:", reg.coef_)
print("Intercept:", reg.intercept_)

# Example Output:
# R2 score 0.4623

# Custom Implementation: MeraRidgeGD (Ridge Regression using Gradient Descent)

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

# Summary and Comparison

# • SGDRegressor (L2):         R² ≈ 0.44
# • Ridge (sklearn):            R² ≈ 0.46
# • Custom MeraRidgeGD:         R² ≈ 0.47
#
# Our custom Gradient Descent Ridge performed slightly better due to optimized weight updates.
# Regularization (α) helped control large weights and improved generalization.

# Key Takeaways

# • Ridge Regression = Linear Regression + L2 Penalty
# • Use when you have multicollinearity or overfitting
# • Gradient Descent can efficiently learn weights when dataset is large
# • α (regularization parameter) must be tuned carefully


# RIDGE REGRESSION — DEEP EXPLANATION (with Visualization)

# 5 Key Concepts of Ridge Regression

# 1 How coefficients are affected?
# 2 Which coefficients shrink more?
# 3 Bias-Variance tradeoff
# 4 Effect on the Loss Function
# 5 Why it’s called Ridge Estimate (vs OLS Estimate)

# 1 HOW COEFFICIENTS ARE AFFECTED

# Ridge regression adds a penalty term α * Σ(wᵢ²) to the cost function.
# As α increases → the penalty grows → coefficients shrink toward zero.
# However, unlike Lasso, they never become exactly zero.
#
#  Cost Function:
#     J(w) = Σ(yᵢ - (Xw + b))² + αΣ(wᵢ²)
#
#  Effect:
#     • Small α → behaves like Linear Regression (no penalty)
#     • Large α → coefficients shrink closer to zero → model becomes smoother

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=2
)

coefs = []
r2_scores = []

for alpha in [0, 10, 100, 1000]:
    reg = Ridge(alpha=alpha)
    reg.fit(X_train, y_train)
    coefs.append(reg.coef_.tolist())
    y_pred = reg.predict(X_test)
    r2_scores.append(r2_score(y_test, y_pred))

plt.figure(figsize=(14, 9))
plt.subplot(221)
plt.bar(data.feature_names, coefs[0])
plt.title(f'Alpha = 0 , R2 = {round(r2_scores[0], 2)}')

plt.subplot(222)
plt.bar(data.feature_names, coefs[1])
plt.title(f'Alpha = 10 , R2 = {round(r2_scores[1], 2)}')

plt.subplot(223)
plt.bar(data.feature_names, coefs[2])
plt.title(f'Alpha = 100 , R2 = {round(r2_scores[2], 2)}')

plt.subplot(224)
plt.bar(data.feature_names, coefs[3])
plt.title(f'Alpha = 1000 , R2 = {round(r2_scores[3], 2)}')

plt.suptitle(" Effect of Regularization on Ridge Coefficients", fontsize=15)
plt.show()

#  As α increases, all coefficients shrink toward zero (less magnitude).
#  The model becomes smoother and less likely to overfit.

# 2 HIGHER COEFFICIENTS ARE IMPACTED MORE

# Ridge penalizes large weights more heavily due to the squared term (w²).
# As α increases, features with larger coefficients get reduced more.
# This helps handle multicollinearity (when features are correlated).

alphas = [0, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
coefs = []

for alpha in alphas:
    reg = Ridge(alpha=alpha)
    reg.fit(X_train, y_train)
    coefs.append(reg.coef_.tolist())

coef_df = pd.DataFrame(coefs, columns=data.feature_names, index=alphas)
print(coef_df)

# Visualization: how each coefficient changes as α increases
input_array = np.array(coefs).T
plt.figure(figsize=(15, 8))
plt.plot(alphas, np.zeros(len(alphas)), color='black', linewidth=3)
for i in range(input_array.shape[0]):
    plt.plot(alphas, input_array[i], label=data.feature_names[i])
plt.xscale('log')
plt.xlabel('Alpha (Regularization Strength)')
plt.ylabel('Coefficient Value')
plt.title('Coefficient Shrinkage Path (Ridge Regularization)')
plt.legend()
plt.show()

#  Larger coefficients shrink faster.
#  Small coefficients remain relatively stable.
#  This ensures balance and stability in the model.

# 3 IMPACT ON BIAS AND VARIANCE

# Ridge increases Bias slightly but reduces Variance significantly.

# • Low α → low bias, high variance (model fits training data tightly)
# • High α → high bias, low variance (model becomes smoother, more stable)
#
# This tradeoff improves model generalization.

from sklearn.preprocessing import PolynomialFeatures
from mlxtend.evaluate import bias_variance_decomp

# Generate nonlinear data
m = 100
X = 5 * np.random.rand(m, 1) - 2
y = 0.7 * X ** 2 - 2 * X + 3 + np.random.randn(m, 1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y.ravel(), test_size=0.2, random_state=2
)

poly = PolynomialFeatures(degree=15)
X_train = poly.fit_transform(X_train)
X_test = poly.transform(X_test)

alphas = np.linspace(0, 30, 100)
loss, bias, variance = [], [], []

for alpha in alphas:
    reg = Ridge(alpha=alpha)
    avg_loss, avg_bias, avg_var = bias_variance_decomp(
        reg, X_train, y_train, X_test, y_test, loss='mse', random_seed=123
    )
    loss.append(avg_loss)
    bias.append(avg_bias)
    variance.append(avg_var)

plt.plot(alphas, loss, label='Loss')
plt.plot(alphas, bias, label='Bias²')
plt.plot(alphas, variance, label='Variance')
plt.xlabel('Alpha (Regularization Strength)')
plt.ylim(0, 5)
plt.legend()
plt.title('Bias-Variance Tradeoff in Ridge Regression')
plt.show()

# As α increases:
#    • Bias increases (model becomes simpler)
#    • Variance decreases (model becomes stable)
#    • Total loss decreases until optimal α, then rises again.


# 4 EFFECT OF REGULARIZATION ON LOSS FUNCTION

# Ridge modifies the standard loss function by adding an αw² term.

# Loss = Σ(y - (wx + b))² + αw²

# Increasing α makes the cost curve steeper → discourages large weights.

from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression

X, y = make_regression(n_samples=100, n_features=1, noise=20, random_state=13)
plt.scatter(X, y)
plt.title("Generated Data for Loss Function Visualization")
plt.show()

reg = LinearRegression()
reg.fit(X, y)
print("Coefficient:", reg.coef_[0])
print("Intercept:", reg.intercept_)

def cal_loss(w, alpha):
    return np.sum((y - (w * X.ravel() + reg.intercept_)) ** 2) + alpha * (w ** 2)

w_values = np.linspace(-50, 100, 100)
plt.figure(figsize=(5, 6))

for alpha in [0, 10, 20, 30, 40, 50, 100]:
    loss = [cal_loss(w, alpha) for w in w_values]
    plt.plot(w_values, loss, label=f'alpha = {alpha}')

plt.legend()
plt.xlabel('Weight (w)')
plt.ylabel('Loss')
plt.title('Effect of α on Loss Function Shape')
plt.show()

# As α increases, the loss curve becomes more convex and steeper.
# This prevents weights from growing too large (overfitting control).

# 5 WHY CALLED "RIDGE ESTIMATE" vs "OLS ESTIMATE"

# • OLS (Ordinary Least Squares) minimizes the sum of squared errors:
#       J(w) = Σ(yᵢ - Xw)²

# • Ridge modifies this by adding αΣ(w²):
#       J(w) = Σ(yᵢ - Xw)² + αΣ(w²)

# Hence, Ridge “adds a ridge” (penalty ridge) to the cost function surface.
# This ridge prevents parameters from exploding and keeps them within a stable range.

# Summary:
# • Coefficients shrink but never become zero.
# • Large weights are penalized more.
# • Bias ↑ and Variance ↓ → better generalization.
# • Loss function becomes smoother and convex.
# • Called Ridge because penalty forms a ridge in the optimization landscape.

