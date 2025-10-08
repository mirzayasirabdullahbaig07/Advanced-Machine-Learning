# ==========================================
# SIMPLE LINEAR REGRESSION (Intuition + Math + Code)
# ==========================================

# ==========================================
# 1️⃣ What is Linear Regression?
# ==========================================
# Linear Regression is a supervised learning algorithm used to predict a continuous target variable (Y)
# using one or more input variables (X).
# It assumes a linear relationship between input and output:
# 
#   y = m*x + b
# 
# where:
#   y = predicted output
#   x = input variable
#   m = slope (how much y changes when x increases by 1 unit)
#   b = intercept (value of y when x = 0)

# ==========================================
# 2️⃣ Why Learn Linear Regression?
# ==========================================
# ✅ Simple and easy to understand
# ✅ Foundation for many other ML algorithms
# ✅ Helps to visualize how models learn relationships between input and output

# ==========================================
# 3️⃣ Types of Linear Regression
# ==========================================
# 1. Simple Linear Regression       → One input (x) and one output (y)
# 2. Multiple Linear Regression     → Multiple inputs (x1, x2, ...) and one output (y)
# 3. Polynomial Regression          → Data is non-linear but can be fitted with powers (x², x³, ...)
# 4. Regularized Regression         → Linear models with penalty terms (Ridge, Lasso, ElasticNet)

# ==========================================
# 4️⃣ Example:
# ==========================================
# Predicting a student’s salary package based on their CGPA.
# Dataset Columns:
#   CGPA   → Independent variable (X)
#   Package → Dependent variable (Y)

# ==========================================
# 5️⃣ The Best Fit Line
# ==========================================
# The best fit line minimizes the total squared error between actual and predicted values.
# It minimizes:
# 
#   E(m, b) = Σ (yᵢ - (m*xᵢ + b))²
#
# This method is called Ordinary Least Squares (OLS).

# ==========================================
# 6️⃣ Derivation of m and b
# ==========================================
# We want to find m and b that minimize E(m, b).

# Step 1: Start with the cost function
#   E(m, b) = Σ (yᵢ - m*xᵢ - b)²

# Step 2: Take partial derivatives and set them to zero
#   ∂E/∂m = -2 Σ xᵢ (yᵢ - m*xᵢ - b) = 0
#   ∂E/∂b = -2 Σ (yᵢ - m*xᵢ - b) = 0

# Step 3: Solve the equations to get:
#   m = Σ(xᵢ - x̄)(yᵢ - ȳ) / Σ(xᵢ - x̄)²
#   b = ȳ - m*x̄

# Hence, 
#   m = Cov(x, y) / Var(x)
#   b = ȳ - m*x̄

# ==========================================
# 7️⃣ Closed-form vs Non-closed-form Solution
# ==========================================
# - Closed-form (OLS) → Direct mathematical formula (for small or medium datasets)
# - Non-closed-form → Iterative approximation using Gradient Descent (for large datasets)

# ==========================================
# 8️⃣ Gradient Descent (Non-Closed Form)
# ==========================================
# Start with random m and b, then iteratively update:
#   m = m - α * (∂E/∂m)
#   b = b - α * (∂E/∂b)
# 
# where α = learning rate, controls step size

# ==========================================
# 9️⃣ Implementation using Scikit-learn
# ==========================================

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Step 1: Load dataset
df = pd.read_csv('placement.csv')
print(df.head())

# Step 2: Visualize the relationship
plt.scatter(df['cgpa'], df['package'], color='blue')
plt.xlabel('CGPA')
plt.ylabel('Package (in LPA)')
plt.title('CGPA vs Package')
plt.show()

# Step 3: Prepare data
X = df[['cgpa']]      # Independent variable
y = df['package']     # Dependent variable

# Step 4: Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2
)

# Step 5: Train the model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Step 6: Predict
y_pred = lr.predict(X_test)

# Step 7: Visualize best fit line
plt.scatter(df['cgpa'], df['package'], color='blue')
plt.plot(X_train, lr.predict(X_train), color='red')
plt.xlabel('CGPA')
plt.ylabel('Package (in LPA)')
plt.title('Linear Regression Fit')
plt.show()

# Step 8: Extract parameters
m = lr.coef_[0]     # Slope
b = lr.intercept_   # Intercept

print("Slope (m):", m)
print("Intercept (b):", b)

# Step 9: Predict for a specific value
predicted_package = m * 8.58 + b
print("Predicted Package for CGPA 8.58 =", predicted_package)

# ==========================================
# 10️⃣ Manual Calculation (OLS Example)
# ==========================================
# Suppose you have 3 data points:
#   x = [7, 8, 9]
#   y = [5, 7, 8]
#
# Step 1: Compute means
#   x̄ = 8
#   ȳ = 6.67
#
# Step 2: Compute slope
#   m = Σ(xᵢ - x̄)(yᵢ - ȳ) / Σ(xᵢ - x̄)²
#     = ((7-8)*(5-6.67) + (8-8)*(7-6.67) + (9-8)*(8-6.67)) / ((7-8)² + (8-8)² + (9-8)²)
#     = 1.5
#
# Step 3: Compute intercept
#   b = ȳ - m*x̄ = 6.67 - 1.5*8 = -5.33
#
# Final Equation:
#   y = 1.5x - 5.33

# ==========================================
# 11️⃣ Error (Residuals)
# ==========================================
# Each data point has an error (residual):
#   dᵢ = yᵢ - ŷᵢ = yᵢ - (m*xᵢ + b)
#
# The goal is to minimize:
#   E = Σ (dᵢ)² = Σ (yᵢ - (m*xᵢ + b))²
#
# When E is minimum, the line is optimal.

# ==========================================
# ✅ SUMMARY
# ==========================================
# Equation: y = m*x + b
# Slope (m): Cov(x, y) / Var(x)
# Intercept (b): ȳ - m*x̄
# Error Function: E = Σ (yᵢ - (m*xᵢ + b))²
# Optimization: OLS (closed form) or Gradient Descent (iterative)
# Concept: Find the best straight line that minimizes prediction error

# ==========================================
# END OF SIMPLE LINEAR REGRESSION EXPLANATION
# ==========================================
