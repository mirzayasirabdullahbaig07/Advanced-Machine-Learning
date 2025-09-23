# ============================================================
#                OUTLIERS IN MACHINE LEARNING
# ============================================================

# -------------------------------
# What are Outliers?
# -------------------------------
# Outliers are data points that deviate significantly 
# from the rest of the dataset. They are unusually high 
# or low values compared to the general distribution 
# of the data.

# -------------------------------
# Why Outliers are Dangerous?
# -------------------------------
# 1. Outliers can distort statistical measures such as 
#    mean, variance, and standard deviation.
# 2. They reduce model accuracy by misleading learning 
#    patterns.
# 3. Outliers increase the error rate in predictions 
#    and may cause overfitting or underfitting.
# 4. Especially harmful for weight-based models like:
#    - Linear Regression
#    - Logistic Regression
#    - Deep Learning (Neural Networks)

# -------------------------------
# Effects of Outliers on Models
# -------------------------------
# - Linear Regression: Outliers pull the regression line 
#   toward themselves, leading to biased coefficients.
# - Deep Learning: Outliers cause unstable training and 
#   large weight updates.
# - Distance-based Models (e.g., KNN, clustering): Outliers 
#   distort distance calculations.
# - Statistical Models: Outliers inflate variance, 
#   reducing reliability.

# -------------------------------
# Techniques to Handle Outliers
# -------------------------------
# 1. Trimming:
#    - Remove extreme data points completely.
#    - Risk: Loss of potentially useful information.
#
# 2. Capping (Winsorization):
#    - Set upper and lower limits for values.
#    - Replace extreme values with boundary values.
#
# 3. Imputation:
#    - Replace outliers with mean, median, or mode.
#    - Useful when data is small.

# -------------------------------
# Outlier Detection Approaches
# -------------------------------
# 1. Normal Distribution:
#    - In normally distributed data, ~99.7% of values 
#      lie within 3 standard deviations (Z-score).
#
# 2. Skewed Distribution:
#    - Median and percentiles are better than mean here.
#
# 3. Percentile Range:
#    - Check for values below the 1st percentile or above 
#      the 99th percentile.

# -------------------------------
# Statistical Techniques
# -------------------------------
# 1. Z-Score Method:
#    - Z = (X - mean) / std
#    - Mark data as outlier if |Z| > threshold (commonly 3).
#
# 2. IQR Method (Interquartile Range):
#    - IQR = Q3 - Q1
#    - Lower Bound = Q1 - 1.5*IQR
#    - Upper Bound = Q3 + 1.5*IQR
#    - Data outside this range is an outlier.
#
# 3. Percentile Method:
#    - Define limits based on specific percentiles (e.g., 
#      1st and 99th percentile).
#
# 4. Winsorization:
#    - Replace extreme values with nearest acceptable 
#      boundary (e.g., cap values above 95th percentile 
#      to 95th percentile).

