# ==============================================================
#                BINNING & BINARIZATION (Feature Encoding)
# ==============================================================

# Purpose: Convert numerical data into categorical data
# Two main techniques:
#   1. Discretization (Binning)
#   2. Binarization

# --------------------------------------------------------------
# 1. DISCRETIZATION (BINNING)
# --------------------------------------------------------------
# - The process of transforming continuous variables into discrete variables.
# - Achieved by creating a set of contiguous intervals ("bins").
# - Each bin represents a range of values (categorical grouping).
# - Also called "binning", where "bin" = interval.

# Why use Discretization?
#   - Handle outliers by grouping extreme values
#   - Improve data spread and interpretability
#   - Useful in some ML algorithms requiring categorical input

# --------------------------------------------------------------
# TYPES OF DISCRETIZATION
# --------------------------------------------------------------
# (A) UNSUPERVISED BINNING
#     1. Equal Width (Uniform Binning)
#        - Divides the range into equal-sized intervals
#        - Example: [0–10], [10–20], [20–30] ...
#        - Used when data is evenly distributed
#
#     2. Equal Frequency (Quantile Binning)
#        - Each bin has (approximately) the same number of observations
#        - Example: Quartiles (25%, 50%, 75%, 100%)
#        - Used when distribution is skewed
#
#     3. K-Means Binning
#        - Cluster-based binning using K-Means algorithm
#        - Groups data into "k" clusters
#        - Useful when natural clusters exist in data

# (B) SUPERVISED BINNING
#     1. Decision Tree Binning
#        - Uses decision tree splits to create bins
#        - Supervised: requires target variable
#        - Useful in classification/regression tasks
#
#     2. Custom Domain-Based Binning
#        - Manually defined bins using domain knowledge
#        - Example: Age groups → [0–12] = Child, [13–19] = Teen, [20–60] = Adult
#        - Useful in business/real-world rules

# --------------------------------------------------------------
# Encoding the Discretized Variable
# --------------------------------------------------------------
# - After binning, categorical encoding (Label Encoding / One-Hot Encoding)
#   is applied to make it ML-friendly.

# --------------------------------------------------------------
# Example: With vs Without Binning
# --------------------------------------------------------------
# Without Binning:
#   Age = [18, 19, 25, 32, 40, 65]
#   Model might treat each age as unique
#
# With Binning:
#   Age Group = ["Teen", "Teen", "Adult", "Adult", "Adult", "Senior"]
#   Model generalizes better and handles outliers.

# --------------------------------------------------------------
# 2. BINARIZATION
# --------------------------------------------------------------
# - Converts numerical values into binary (0/1) based on a threshold.
# - Example: Salary > 50000 → 1, else 0
#
# Why use Binarization?
#   - Simplifies numerical features
#   - Helps with threshold-based decisions (e.g., "high" vs "low")
#
# Where is it used?
#   - Text classification (word presence = 1, absence = 0)
#   - Risk analysis (income above threshold = high risk)
#   - Logistic regression or models requiring binary features
