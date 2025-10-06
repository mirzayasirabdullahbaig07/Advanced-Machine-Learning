# ==========================================
# SIMPLE LINEAR REGRESSION (Intuition + Code + Math)
# ==========================================

# What is Linear Regression?
# Linear Regression is a supervised learning algorithm used to predict a continuous output variable (y)
# based on one or more input variables (x).
# It tries to find the best straight line that fits the data points.

# ------------------------------------------
# Why Learn Linear Regression?
# ------------------------------------------
# It is very easy to understand and implement.
# It is the foundation for many other algorithms (like logistic regression, SVM, etc.).
# Helps you understand how models learn relationships between input and output.

# ------------------------------------------
# Linear Regression belongs to:
# ------------------------------------------
# Type: Supervised Learning
# Problem Type: Regression Problem (predicting continuous values like salary, price, temperature, etc.)

# ------------------------------------------
# Types of Linear Regression:
# ------------------------------------------
# 1. Simple Linear Regression       → One input (x) and one output (y)
# 2. Multiple Linear Regression     → Multiple inputs (x1, x2, x3, ...) and one output (y)
# 3. Polynomial Linear Regression   → Data is non-linear, but we can fit it using polynomial terms (x², x³, ...)
# 4. Regularization Regression      → Linear model with penalty terms (like Ridge, Lasso, ElasticNet) to prevent overfitting.

# ------------------------------------------
# Simple Linear Regression
# ------------------------------------------
# Works with only ONE independent variable (X) and ONE dependent variable (Y)
# Example: Predicting a student's salary package based on their CGPA.

# Formula:  y = m*x + b
# where,
# m = slope of the line
# b = y-intercept (where line crosses y-axis)

# ------------------------------------------
# What is Best Fit Line?
# ------------------------------------------
# The best fit line is the line that minimizes the error between actual data points and predicted values.
# The algorithm finds the best m (slope) and b (intercept) to minimize these errors.
# Mathematically, it minimizes the "Sum of Squared Errors (SSE)" using Ordinary Least Squares (OLS) method.

# ------------------------------------------
# Let's Implement It
# ------------------------------------------

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Step 1: Load Dataset
df = pd.read_csv('placement.csv')
df.head()

# Step 2: Visualize Data
plt.scatter(df['cgpa'], df['package'])
plt.xlabel('CGPA')
plt.ylabel('Package (in LPA)')
plt.title('CGPA vs Package')

# Step 3: Prepare Data
X = df.iloc[:, 0:1]   # Independent variable (CGPA)
y = df.iloc[:, -1]    # Dependent variable (Package)

# Step 4: Split into Train & Test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2
)

# Step 5: Train the Model
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)

# Step 6: Predict
y_pred = lr.predict(X_test)

# Step 7: Visualize the Best Fit Line
plt.scatter(df['cgpa'], df['package'])
plt.plot(X_train, lr.predict(X_train), color='red')  # best fit line
plt.xlabel('CGPA')
plt.ylabel('Package (in LPA)')
plt.title('Linear Regression Fit')

# Step 8: Get Equation Parameters
m = lr.coef_        # slope
b = lr.intercept_   # intercept

print("Slope (m):", m)
print("Intercept (b):", b)

# Example: Predict package for CGPA = 8.58
predicted_package = m * 8.58 + b
print("Predicted Package for CGPA 8.58 =", predicted_package)

# ------------------------------------------
# How Does Linear Regression Work (Intuition)
# ------------------------------------------
# Step 1: Start with a random line (random m and b)
# Step 2: Calculate predictions for all data points using y = m*x + b
# Step 3: Compute error = actual_y - predicted_y
# Step 4: Use these errors to adjust m and b to minimize the total squared error
# Step 5: Repeat until the best line (minimum error) is found

# ------------------------------------------
# How Humans Understand It:
# ------------------------------------------
# Imagine you plot data points of CGPA vs Salary.
# You try to draw a straight line that best passes through most of the points.
# The closer your line is to the points, the better your prediction.
# The computer does this mathematically by adjusting slope (m) and intercept (b) until it minimizes the error.

# ==========================================
# Summary
# ==========================================
# Linear Regression helps find relationships between variables.
# It’s simple, powerful, and forms the base for understanding more advanced ML models.
