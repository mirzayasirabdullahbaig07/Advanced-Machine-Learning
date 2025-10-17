# ===============================================================
# ELASTICNET REGRESSION
# ===============================================================

# ---------------------------------------------------------------
# What is ElasticNet Regression?
# ---------------------------------------------------------------
# ElasticNet Regression is a **regularized linear regression technique**
# that combines both **L1 (Lasso)** and **L2 (Ridge)** penalties.
#
# It’s useful when:
#   - You have **many correlated features**
#   - You want **sparse coefficients** (like Lasso)
#   - You also want **stability** (like Ridge)
#
# In short:
#   ElasticNet = Lasso + Ridge

# ---------------------------------------------------------------
# Where It Can Be Used
# ---------------------------------------------------------------
# ElasticNet is used when:
#    Dataset has **high-dimensional** features (many predictors)
#    Some features are **correlated**
#    We don’t know whether to use Ridge or Lasso
#    We want to balance between **feature selection** and **shrinkage**
#
# Common use cases:
#   - Genomics data (many correlated genes)
#   - Text or NLP data with large feature space
#   - Financial and economic models
#   - Machine learning feature selection pipelines

# ---------------------------------------------------------------
# How It Works with Lasso and Ridge Regression
# ---------------------------------------------------------------
# ElasticNet combines the penalties of both Lasso and Ridge:
#
#     Loss = RSS + α * [ (1 - r)/2 * Σ(θ_j²) + r * Σ|θ_j| ]
#
# where:
#     RSS   = Residual Sum of Squares (like linear regression)
#     α     = overall regularization strength
#     r     = l1_ratio → determines balance between Lasso & Ridge
#     θ_j   = model coefficients
#
# If:
#     r = 1  → becomes Lasso Regression (only L1 penalty)
#     r = 0  → becomes Ridge Regression (only L2 penalty)
#     0 < r < 1 → combination of both

# ---------------------------------------------------------------
# Mathematical Formula for ElasticNet
# ---------------------------------------------------------------
#     J(θ) = (1/2n) * Σ(y_i - ŷ_i)²  +  α * [ (1 - r)/2 * ||θ||²₂  +  r * ||θ||₁ ]
#
# where:
#     ||θ||²₂ → L2 norm (sum of squares of coefficients)
#     ||θ||₁ → L1 norm (sum of absolute coefficients)
#     α → regularization strength (controls overall penalty)
#     r → l1_ratio (balance between L1 and L2)
#
# Interpretation:
#   - The L1 term helps eliminate unimportant features (sparsity)
#   - The L2 term helps stabilize the model when features are correlated

# ---------------------------------------------------------------
# When to Use Which Model
# ---------------------------------------------------------------
# Ridge Regression:
#     - When **all features are important**
#     - Handles multicollinearity well (correlated features)
#
# Lasso Regression:
#     - When **only a subset of features are important**
#     - Automatically performs feature selection (sets some coefficients to 0)
#
# ElasticNet Regression:
#     - When **dataset is very large and complex**
#     - When **you don’t know** whether Lasso or Ridge is better
#     - It provides a balance between both worlds

# ---------------------------------------------------------------
# Benefits of ElasticNet Regression
# ---------------------------------------------------------------
# 1. Combines Lasso (feature selection) and Ridge (stability)
# 2. Works well with correlated features
# 3. Reduces model variance and prevents overfitting
# 4. More flexible due to two hyperparameters (alpha, l1_ratio)
# 5. Performs automatic variable selection and shrinkage together

# ---------------------------------------------------------------
# Disadvantages of ElasticNet Regression
# ---------------------------------------------------------------
# 1. Requires tuning of two hyperparameters (α and l1_ratio)
# 2. Can be computationally expensive for large datasets
# 3. Lasso part may still bias coefficients toward zero too strongly
# 4. Less interpretable when both penalties interact

# ---------------------------------------------------------------
# Example: Comparing Linear, Ridge, Lasso, and ElasticNet
# ---------------------------------------------------------------

from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Load dataset
X, y = load_diabetes(return_X_y=True)

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# ---------------------------------------------------------------
# Linear Regression (Baseline Model)
# ---------------------------------------------------------------
reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
print("Linear Regression R2 Score:", r2_score(y_test, y_pred))
# Output: 0.4399
# → No regularization, might overfit slightly.

# ---------------------------------------------------------------
# Ridge Regression
# ---------------------------------------------------------------
reg = Ridge(alpha=0.1)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
print("Ridge Regression R2 Score:", r2_score(y_test, y_pred))
# Output: 0.4519
# → Slight improvement due to L2 penalty (reduces overfitting).

# ---------------------------------------------------------------
# Lasso Regression
# ---------------------------------------------------------------
reg = Lasso(alpha=0.01)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
print("Lasso Regression R2 Score:", r2_score(y_test, y_pred))
# Output: 0.4411
# → Performs feature selection (some coefficients become 0).

# ---------------------------------------------------------------
# ElasticNet Regression
# ---------------------------------------------------------------
reg = ElasticNet(alpha=0.005, l1_ratio=0.9)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
print("ElasticNet Regression R2 Score:", r2_score(y_test, y_pred))
# Output: 0.4531
# → Best of both worlds: combines L1 (feature selection) and L2 (stability).

# ---------------------------------------------------------------
# Summary
# ---------------------------------------------------------------
# Model        Regularization     Feature Selection   R² Score
# ---------------------------------------------------------------
# Linear       None               No                  0.4399
# Ridge        L2                 No                  0.4519
# Lasso        L1                 Yes (sparse)        0.4411
# ElasticNet   L1 + L2            Yes + Stable        0.4531
#
# Insight:
# ElasticNet gives the best generalization performance because it
# combines both regularization strengths while controlling overfitting.
