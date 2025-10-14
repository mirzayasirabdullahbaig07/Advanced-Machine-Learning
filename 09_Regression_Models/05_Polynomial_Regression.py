# ==========================================================================================================
# POLYNOMIAL REGRESSION - COMPLETE EXPLANATION + CODE
# ==========================================================================================================

# ----------------------------------------------------------------------------------------------------------
# 1. WHAT IS POLYNOMIAL REGRESSION?
# ----------------------------------------------------------------------------------------------------------
# Polynomial Regression is a type of regression analysis in which the relationship between 
# the independent variable (X) and dependent variable (y) is modeled as an nth degree polynomial.
#
# It’s basically an extension of Linear Regression, but it can fit non-linear relationships.
#
# Example:
# y = β₀ + β₁x + β₂x² + β₃x³ + ... + βₙxⁿ + ε
#
# Where:
#   β₀, β₁, β₂, ... βₙ are the model coefficients
#   ε is the random error term
# ----------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------
# 2. WHERE IS IT USEFUL?
# ----------------------------------------------------------------------------------------------------------
#  When data shows a **curved or non-linear trend**.
#  Useful in:
#    - Growth rate prediction (population, business, plants, etc.)
#    - Physics (trajectory, motion)
#    - Economics (cost, demand curves)
#    - Machine Learning feature engineering for non-linear relationships
# ----------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------
# 3. MATHEMATICAL FORMULATION
# ----------------------------------------------------------------------------------------------------------
# For a dataset (x₁, y₁), (x₂, y₂), …, (xₙ, yₙ)
#
# Polynomial equation:
# y = β₀ + β₁x + β₂x² + ... + βₙxⁿ
#
# The goal is to find coefficients β₀, β₁, β₂, …, βₙ
# that minimize the sum of squared errors:
#
# Cost Function:
#     J(β) = (1/2m) * Σ (ŷᵢ - yᵢ)²
#
# where ŷᵢ = predicted value = β₀ + β₁xᵢ + β₂xᵢ² + ... + βₙxᵢⁿ
# ----------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------
# 4. ADVANTAGES
# ----------------------------------------------------------------------------------------------------------
#  Can model complex, non-linear relationships
#  Easy to implement using sklearn’s PolynomialFeatures
#  Extends simple linear regression
#
# ----------------------------------------------------------------------------------------------------------
# ➖ 5. DISADVANTAGES
# ----------------------------------------------------------------------------------------------------------
#  Prone to overfitting for high-degree polynomials
#  Sensitive to outliers
#  Extrapolation beyond data range is unreliable
#  Computationally expensive for large degrees or features
# ----------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------
# 6. IMPLEMENTATION EXAMPLE (1D Polynomial Regression)
# ----------------------------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline

# ----------------------------------------------------------------------------------------------------------
# STEP 1: Generate synthetic non-linear data
# y = 0.8x² + 0.9x + 2 + noise
# ----------------------------------------------------------------------------------------------------------
X = 6 * np.random.rand(200, 1) - 3
y = 0.8 * X**2 + 0.9 * X + 2 + np.random.randn(200, 1)

# Visualize dataset
plt.plot(X, y, 'b.')
plt.xlabel("X")
plt.ylabel("y")
plt.title("Non-linear Data")
plt.show()

# ----------------------------------------------------------------------------------------------------------
# STEP 2: Train-Test Split
# ----------------------------------------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# ----------------------------------------------------------------------------------------------------------
# STEP 3: Apply Simple Linear Regression (for comparison)
# ----------------------------------------------------------------------------------------------------------
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
print("Linear Regression R² Score:", r2_score(y_test, y_pred))

# Visualize poor linear fit
plt.plot(X_train, lr.predict(X_train), color='r')
plt.plot(X, y, "b.")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Linear Regression Fit")
plt.show()

# ----------------------------------------------------------------------------------------------------------
# STEP 4: Apply Polynomial Regression (Degree = 2)
# ----------------------------------------------------------------------------------------------------------
poly = PolynomialFeatures(degree=2, include_bias=True)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

lr_poly = LinearRegression()
lr_poly.fit(X_train_poly, y_train)
y_pred_poly = lr_poly.predict(X_test_poly)

print("Polynomial Regression R² Score:", r2_score(y_test, y_pred_poly))
print("Coefficients:", lr_poly.coef_)
print("Intercept:", lr_poly.intercept_)

# Visualize polynomial fit
X_new = np.linspace(-3, 3, 200).reshape(200, 1)
X_new_poly = poly.transform(X_new)
y_new = lr_poly.predict(X_new_poly)

plt.plot(X_new, y_new, "r-", linewidth=2, label="Polynomial Fit")
plt.plot(X_train, y_train, "b.", label="Training points")
plt.plot(X_test, y_test, "g.", label="Testing points")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.title("Polynomial Regression (Degree = 2)")
plt.show()

# ----------------------------------------------------------------------------------------------------------
# STEP 5: Experiment with Higher Degrees (Pipeline + StandardScaler)
# ----------------------------------------------------------------------------------------------------------
def polynomial_regression(degree):
    """
    Visualize polynomial regression fits for different degrees.
    """
    poly_features = PolynomialFeatures(degree=degree, include_bias=False)
    std_scaler = StandardScaler()
    lin_reg = LinearRegression()

    pipeline = Pipeline([
        ("poly_features", poly_features),
        ("std_scaler", std_scaler),
        ("lin_reg", lin_reg),
    ])

    pipeline.fit(X, y)
    X_new = np.linspace(-3, 3, 100).reshape(100, 1)
    y_new = pipeline.predict(X_new)

    plt.plot(X_new, y_new, 'r', label=f"Degree {degree}", linewidth=2)
    plt.plot(X_train, y_train, "b.", label="Train")
    plt.plot(X_test, y_test, "g.", label="Test")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()
    plt.title(f"Polynomial Regression (Degree = {degree})")
    plt.show()

# Example (Be careful: very high degree may cause overflow or overfitting)
polynomial_regression(5)

# ----------------------------------------------------------------------------------------------------------
# STEP 6: Using Stochastic Gradient Descent for Polynomial Regression
# ----------------------------------------------------------------------------------------------------------
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

sgd = SGDRegressor(max_iter=1000, tol=1e-3)
sgd.fit(X_train_poly, y_train.ravel())  # flatten y for SGDRegressor
y_pred_sgd = sgd.predict(X_test_poly)

plt.plot(X_train, y_train, "b.", label='Train')
plt.plot(X_test, y_test, "g.", label='Test')
plt.plot(X_new, sgd.predict(poly.transform(X_new)), "r-", label=f"SGD Predictions (R²={r2_score(y_test, y_pred_sgd):.2f})")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.title("Polynomial Regression using Gradient Descent")
plt.show()

# ----------------------------------------------------------------------------------------------------------
# STEP 7: 3D POLYNOMIAL REGRESSION EXAMPLE
# ----------------------------------------------------------------------------------------------------------
import plotly.express as px
import plotly.graph_objects as go

# Generate synthetic 3D data
x = 7 * np.random.rand(100, 1) - 2.8
y_ = 7 * np.random.rand(100, 1) - 2.8
z = x**2 + y_**2 + 0.2*x + 0.2*y_ + 0.1*x*y_ + 2 + np.random.randn(100, 1)

# Visualize 3D scatter
fig = px.scatter_3d(x=x.ravel(), y=y_.ravel(), z=z.ravel(), title="3D Data Points")
fig.show()

# Fit Polynomial Regression (degree = 2)
X_multi = np.concatenate([x, y_], axis=1)
poly = PolynomialFeatures(degree=2)
X_multi_poly = poly.fit_transform(X_multi)
lr = LinearRegression()
lr.fit(X_multi_poly, z)

# Create grid for surface prediction
x_input = np.linspace(x.min(), x.max(), 20)
y_input = np.linspace(y_.min(), y_.max(), 20)
xGrid, yGrid = np.meshgrid(x_input, y_input)
grid_points = np.c_[xGrid.ravel(), yGrid.ravel()]
z_pred = lr.predict(poly.transform(grid_points)).reshape(20, 20)

# 3D surface plot
fig = go.Figure()
fig.add_trace(go.Scatter3d(x=x.ravel(), y=y_.ravel(), z=z.ravel(),
                           mode='markers', marker=dict(size=4, color='blue')))
fig.add_trace(go.Surface(x=x_input, y=y_input, z=z_pred, opacity=0.7))
fig.update_layout(title="3D Polynomial Regression Surface", scene=dict(zaxis=dict(range=[0, 35])))
fig.show()

# ----------------------------------------------------------------------------------------------------------
# SUMMARY:
# ----------------------------------------------------------------------------------------------------------
# Polynomial Regression is a powerful technique for modeling non-linear data.
# - Simple to implement using sklearn
# - Degree selection is crucial to avoid overfitting
# - Works well for small to medium datasets
# ----------------------------------------------------------------------------------------------------------
