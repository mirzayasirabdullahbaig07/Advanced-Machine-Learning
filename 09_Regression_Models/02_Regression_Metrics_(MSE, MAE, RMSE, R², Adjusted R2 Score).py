# ==========================================
# 02_Regression Metrics (MAE, MSE, RMSE, R², Adjusted R²)
# ==========================================

# These metrics are used to evaluate the performance of regression models.
# The goal of regression evaluation metrics is to measure how close
# the predicted values are to the actual (true) values.

# ==========================================
# 1️⃣ MEAN ABSOLUTE ERROR (MAE)
# ==========================================
# 🔹 What is MAE?
# Mean Absolute Error measures the average magnitude of errors in predictions,
# without considering their direction (positive or negative).
# It simply averages the absolute difference between predicted and actual values.

# 🔹 Why is it called “Absolute”?
# Because it uses the absolute difference |yᵢ - ŷᵢ|, ensuring that negative errors
# do not cancel out positive ones.

# 🔹 Intuition:
# It tells you on average, how much your predictions deviate from the actual values.
# For example, MAE = 2 means predictions are off by 2 units on average.

# 🔹 Mathematical Formula:
#     MAE = (1/n) * Σ |yᵢ - ŷᵢ|
# where:
#     yᵢ = actual value
#     ŷᵢ = predicted value
#     n = total number of samples

# 🔹 Advantages:
# ✅ Easy to interpret and understand
# ✅ Not sensitive to large errors (less effect of outliers)

# 🔹 Disadvantages:
# ❌ Doesn’t penalize large errors strongly
# ❌ Not differentiable at 0 (problem in gradient-based optimization)


# ==========================================
# 2️⃣ MEAN SQUARED ERROR (MSE)
# ==========================================
# 🔹 What is MSE?
# Mean Squared Error measures the average squared difference between
# actual and predicted values.

# 🔹 Why “Squared”?
# Because the errors are squared to make all values positive and to penalize
# larger errors more heavily.

# 🔹 Intuition:
# Squaring the errors gives more weight to large deviations,
# meaning if your model makes a few big mistakes, MSE increases sharply.

# 🔹 Mathematical Formula:
#     MSE = (1/n) * Σ (yᵢ - ŷᵢ)²

# 🔹 Advantages:
# ✅ Strongly penalizes large errors
# ✅ Smooth and differentiable (good for optimization)

# 🔹 Disadvantages:
# ❌ Sensitive to outliers (large errors dominate)
# ❌ Units are squared (harder to interpret in original scale)


# ==========================================
# 3️⃣ ROOT MEAN SQUARED ERROR (RMSE)
# ==========================================
# 🔹 What is RMSE?
# RMSE is simply the square root of MSE. It brings the error back
# to the same unit as the original target variable.

# 🔹 Intuition:
# RMSE gives an idea of the standard deviation of residuals.
# It shows how concentrated the data is around the best-fit line.

# 🔹 Mathematical Formula:
#     RMSE = √( (1/n) * Σ (yᵢ - ŷᵢ)² )

# 🔹 Advantages:
# ✅ Same unit as target variable
# ✅ Penalizes larger errors more

# 🔹 Disadvantages:
# ❌ Sensitive to outliers (like MSE)
# ❌ Slightly harder to interpret than MAE


# ==========================================
# 4️⃣ R² SCORE (Coefficient of Determination)
# ==========================================
# 🔹 What is R²?
# R² measures how well the regression model fits the data.
# It represents the proportion of variance in the dependent variable
# that is predictable from the independent variable(s).

# 🔹 Why called “R-squared”?
# Because it is derived from the square of the correlation coefficient (r).

# 🔹 Intuition:
# R² = 1 means perfect fit (model explains 100% of variance)
# R² = 0 means the model does no better than predicting the mean
# R² < 0 means the model performs worse than the mean

# 🔹 Mathematical Formula:
#     R² = 1 - [Σ(yᵢ - ŷᵢ)² / Σ(yᵢ - ȳ)²]
# where:
#     ȳ = mean of actual values

# 🔹 Advantages:
# ✅ Easy to interpret (percentage of explained variance)
# ✅ Works well for comparing different models

# 🔹 Disadvantages:
# ❌ R² always increases when new predictors are added (even if useless)
# ❌ Doesn’t indicate if a model is unbiased


# ==========================================
# 5️⃣ ADJUSTED R² SCORE
# ==========================================
# 🔹 Why Adjusted R²?
# Adding new features always increases R² — even if they add no predictive power.
# Adjusted R² fixes this by penalizing the addition of unnecessary variables.

# 🔹 Intuition:
# It adjusts R² based on the number of predictors (p) and samples (n).

# 🔹 Mathematical Formula:
#     Adjusted R² = 1 - [(1 - R²) * (n - 1) / (n - p - 1)]
# where:
#     n = number of samples
#     p = number of independent variables

# 🔹 Advantages:
# ✅ Fair comparison between models with different numbers of predictors
# ✅ Prevents overfitting by penalizing irrelevant features

# 🔹 Disadvantages:
# ❌ Slightly harder to interpret
# ❌ Only valid for linear regression type models


# ==========================================
# 6️⃣ CODE IMPLEMENTATION (Using Placement Dataset)
# ==========================================

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv('placement.csv')

# Visualize the data
plt.scatter(df['cgpa'], df['package'])
plt.xlabel('CGPA')
plt.ylabel('Package (in LPA)')
plt.title('CGPA vs Package')
plt.show()

# Prepare data
X = df[['cgpa']]
y = df['package']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Train Linear Regression Model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Make Predictions
y_pred = lr.predict(X_test)

# ==========================================
# 7️⃣ EVALUATE MODEL USING DIFFERENT METRICS
# ==========================================

print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R² Score:", r2_score(y_test, y_pred))

# ==========================================
# 8️⃣ MANUAL CALCULATION OF ADJUSTED R²
# ==========================================
r2 = r2_score(y_test, y_pred)
n = X_test.shape[0]
p = X_test.shape[1]

adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))
print("Adjusted R² Score:", adjusted_r2)


# ==========================================
# 9️⃣ Visual Understanding of Adjusted R²
# ==========================================

# Case 1: Add a random feature (no real relationship)
new_df1 = df.copy()
new_df1['random_feature'] = np.random.random(200)
X = new_df1[['cgpa', 'random_feature']]
y = new_df1['package']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

r2 = r2_score(y_test, y_pred)
adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - 2 - 1))
print("After adding random feature:")
print("R² =", r2)
print("Adjusted R² =", adjusted_r2)

# Case 2: Add a meaningful feature (related to target)
new_df2 = df.copy()
new_df2['iq'] = new_df2['package'] + (np.random.randint(-12, 12, 200) / 10)
X = new_df2[['cgpa', 'iq']]
y = new_df2['package']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

r2 = r2_score(y_test, y_pred)
adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - 2 - 1))
print("\nAfter adding meaningful feature:")
print("R² =", r2)
print("Adjusted R² =", adjusted_r2)


# ==========================================
# ✅ SUMMARY
# ==========================================
# Metric        | Measures                     | Penalizes Outliers | Range         | Best Value
# --------------------------------------------------------------------------------------------
# MAE           | Average absolute error        | ❌ No               | [0, ∞)        | 0
# MSE           | Average squared error         | ✅ Yes              | [0, ∞)        | 0
# RMSE          | Root of MSE                   | ✅ Yes              | [0, ∞)        | 0
# R² Score      | Variance explained            | ❌ No               | (-∞, 1]       | 1
# Adjusted R²   | Penalized R² for extra vars   | ❌ No               | (-∞, 1]       | 1

# ==========================================
# END OF REGRESSION METRICS EXPLANATION
# ==========================================
