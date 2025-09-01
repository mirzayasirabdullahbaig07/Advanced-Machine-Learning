# ================================
# Instance-Based Learning vs Model-Based Learning
# ================================

# 1. Instance-Based Learning
# ----------------------------
# - Also known as "memory-based learning."
# - The algorithm memorizes the entire training dataset.
# - No explicit model is created.
# - During prediction, it compares the new data point with stored examples
#   and makes decisions based on similarity (distance measures).
# - It does not have a separate training phase (lazy learning).
# - Example: K-Nearest Neighbors (KNN).
#
# Pros:
#   - Simple and intuitive.
#   - Works well for small datasets.
#   - Can adapt quickly with new data.
#
# Cons:
#   - Prediction is slow (needs to search whole dataset).
#   - Requires large memory for storing all instances.
#   - Struggles with noisy or irrelevant features.

# 2. Model-Based Learning
# ----------------------------
# - Instead of memorizing, it learns a general function from data.
# - It creates a mathematical model or decision boundary.
# - Has a distinct training phase (eager learning).
# - Once trained, the model can generalize to unseen data.
# - Example: Linear Regression, Logistic Regression, Decision Trees, SVM, Neural Networks.
#
# Pros:
#   - Fast prediction once the model is trained.
#   - Efficient use of memory (doesn’t need to store all data).
#   - Can handle large datasets and generalize better.
#
# Cons:
#   - Training can be computationally expensive.
#   - Requires assumptions (e.g., linearity in linear regression).
#   - Might underfit or overfit if the model is not well chosen.

# ================================
# Quick Summary
# ================================
# Instance-Based:
#   - Memorizes data
#   - No explicit training
#   - Example: KNN
#
# Model-Based:
#   - Learns a function
#   - Needs training
#   - Example: Linear Regression, SVM
