# 📘-------------------------------------------------------------
# 📚 MULTIPLE LINEAR REGRESSION — COMPLETE EXPLANATION
# ---------------------------------------------------------------
# ✅ INTUITION:
# Linear Regression predicts a **continuous target variable (y)** 
# using one or more **independent input features (X)**.
#
# When there is **only one input**, it’s called *Simple Linear Regression*.
# When there are **two or more inputs**, it’s called *Multiple Linear Regression*.
#
# Example:
#   Predicting house price = f(area, bedrooms, location)
#
# Formula:
#   y = β0 + β1*x1 + β2*x2 + ... + βn*xn
#
# Where:
#   y   = target (dependent variable)
#   β0  = intercept (bias)
#   β1, β2, ... βn = coefficients (weights)
#   x1, x2, ... xn = input features
#
# ---------------------------------------------------------------
# ✅ WHY WE USE:
# - To predict a **continuous numerical value**.
# - To understand **how different variables impact the target**.
# - To study **relationships** between dependent and independent variables.
#
# ---------------------------------------------------------------
# ✅ WHEN TO USE:
# - When the **target variable is continuous**.
# - When you have **multiple input features**.
# - When the relationship between inputs (X) and output (y) is **linear**.
#
# ---------------------------------------------------------------
# ✅ MATHEMATICAL FORMULATION (CORE MATH BEHIND IT):
#
# Equation in matrix form:
#     y = Xβ + ε
#
# Where:
#   y → vector of observed outputs (m×1)
#   X → matrix of inputs (m×(n+1)) → includes 1s for intercept
#   β → vector of parameters (coefficients)
#   ε → error term (residual)
#
# Goal → minimize the **sum of squared errors**:
#     minimize  Σ(y - Xβ)²
#
# Solution using the **Normal Equation** (Ordinary Least Squares):
#     β = (XᵀX)^(-1) Xᵀ y
#
# This gives the **best-fit coefficients** that minimize the squared error.
#
# ---------------------------------------------------------------
# ✅ KEY METRICS:
# - MAE (Mean Absolute Error)
# - MSE (Mean Squared Error)
# - R² Score (Goodness of Fit)
#
# ---------------------------------------------------------------
# ✅ ADVANTAGES:
# - Easy to understand and interpret.
# - Fast and computationally efficient.
# - Works well when relationships are linear.
# - Foundation for advanced models like Ridge & Lasso.
#
# ✅ DISADVANTAGES:
# - Assumes linear relationships only.
# - Sensitive to **outliers**.
# - Affected by **multicollinearity** (correlated features).
# - Performs poorly on **non-linear** data.
#
# ---------------------------------------------------------------
# ✅ ISSUES:
# - Overfitting with many correlated features.
# - Violation of regression assumptions (linearity, independence, etc.).
# - Not ideal for categorical or highly skewed data.
#
# ---------------------------------------------------------------
# ✅ BENEFITS:
# - Simple yet powerful for trend prediction.
# - Coefficients show the **importance of each variable**.
# - Good baseline model for regression tasks.
#
# ---------------------------------------------------------------
# ✅ FORMULA SUMMARY:
# Concept: Predict continuous y using linear combination of features.
# Formula:   y = β0 + β1x1 + β2x2 + ... + βnxn
# Objective: Minimize squared error (Σ(y - ŷ)²)
# Solution:  β = (XᵀX)^(-1) Xᵀ y
# Key Metrics: MAE, MSE, R²
# Libraries: sklearn, numpy, pandas, plotly
#
# ---------------------------------------------------------------
# ✅ STEP 1: GENERATE SAMPLE DATA
# ---------------------------------------------------------------
from sklearn.datasets import make_regression
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Create synthetic regression dataset with 2 input features
X, y = make_regression(n_samples=100, n_features=2, n_informative=2, noise=50)

# Convert into DataFrame for better visualization
df = pd.DataFrame({'feature1': X[:, 0], 'feature2': X[:, 1], 'target': y})

# View top 5 rows
print(df.head())

# 3D Visualization of the dataset
fig = px.scatter_3d(df, x='feature1', y='feature2', z='target', title="3D Scatter Plot of Multiple Regression Data")
fig.show()

# ---------------------------------------------------------------
# ✅ STEP 2: SPLIT DATA INTO TRAIN AND TEST SETS
# ---------------------------------------------------------------
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

# ---------------------------------------------------------------
# ✅ STEP 3: TRAIN USING SKLEARN'S LINEAR REGRESSION
# ---------------------------------------------------------------
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)

# Predictions
y_pred = lr.predict(X_test)

# Evaluate performance
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Coefficients and Intercept
print("Coefficients (β1, β2):", lr.coef_)
print("Intercept (β0):", lr.intercept_)

# ---------------------------------------------------------------
# ✅ STEP 4: MATHEMATICAL IMPLEMENTATION — NORMAL EQUATION
# ---------------------------------------------------------------
# Formula: β = (XᵀX)^(-1) Xᵀy
# Let’s implement our own regression model from scratch using NumPy

class MeraLR:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X_train, y_train):
        # Add bias column (1s) to X_train for intercept
        X_train = np.insert(X_train, 0, 1, axis=1)
        
        # Apply Normal Equation → β = (XᵀX)^(-1) Xᵀy
        betas = np.linalg.inv(X_train.T.dot(X_train)).dot(X_train.T).dot(y_train)
        
        # Extract intercept and coefficients
        self.intercept_ = betas[0]
        self.coef_ = betas[1:]

    def predict(self, X_test):
        # Prediction: ŷ = Xβ + intercept
        return np.dot(X_test, self.coef_) + self.intercept_

# ---------------------------------------------------------------
# ✅ STEP 5: TEST CUSTOM MODEL ON REAL DATA (DIABETES DATASET)
# ---------------------------------------------------------------
from sklearn.datasets import load_diabetes
X, y = load_diabetes(return_X_y=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Train custom model
lr_custom = MeraLR()
lr_custom.fit(X_train, y_train)

# Predictions
y_pred_custom = lr_custom.predict(X_test)

# R² Score of custom model
print("Custom Model R² Score:", r2_score(y_test, y_pred_custom))

# Compare with sklearn’s LinearRegression
reg = LinearRegression()
reg.fit(X_train, y_train)
print("Sklearn R² Score:", r2_score(y_test, reg.predict(X_test)))

# Both results are nearly identical ✅

# ---------------------------------------------------------------
# ✅ STEP 6: INTERPRETATION
# ---------------------------------------------------------------
# - Coefficients (β1...βn): show how much target (y) changes 
#   when the respective feature increases by 1 unit (keeping others constant).
# - Intercept (β0): predicted y value when all features = 0.
#
# Example:
#   If β1 = 5 → y increases by 5 when x1 increases by 1 (others constant).

# ---------------------------------------------------------------
# ✅ STEP 7: MODEL ASSUMPTIONS
# ---------------------------------------------------------------
# 1️⃣ Linearity → Relationship between X and y is linear
# 2️⃣ Independence → Residuals are independent
# 3️⃣ Homoscedasticity → Constant variance of residuals
# 4️⃣ Normality → Errors follow a normal distribution
# 5️⃣ No Multicollinearity → Features are not highly correlated

# ---------------------------------------------------------------
# ✅ SUMMARY
# ---------------------------------------------------------------
# Multiple Linear Regression is one of the **most fundamental** ML algorithms.
#
# It is best used when:
# - The relationship is linear
# - The target is continuous
# - Interpretability is important
#
# Math Backbone:
#   β = (XᵀX)^(-1) Xᵀy
#
# Evaluation Metrics:
#   MAE, MSE, R²
#
# Strengths:
#   ✔ Simple
#   ✔ Interpretable
#   ✔ Analytical Solution
#
# Weaknesses:
#   ❌ Sensitive to outliers
#   ❌ Assumes linearity
#   ❌ Multicollinearity issues
#
# ---------------------------------------------------------------
# 🚀 End of Complete Multiple Linear Regression Explanation
# ---------------------------------------------------------------
