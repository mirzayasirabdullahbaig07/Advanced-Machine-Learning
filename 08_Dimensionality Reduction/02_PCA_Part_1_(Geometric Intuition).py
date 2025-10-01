# ==============================================
# PCA (Principal Component Analysis) - Part 1
# Geometric Intuition
# ==============================================
# PCA is a dimensionality reduction technique widely used in 
# machine learning, statistics, and data visualization.
#
# It transforms high-dimensional data into fewer dimensions 
# while preserving as much variance (information) as possible.
# ==============================================

# -----------------------------
# What is PCA?
# -----------------------------
# - PCA = Principal Component Analysis.
# - It finds "new axes" (directions) called Principal Components.
# - These axes maximize the variance of the data.
# - Instead of using original correlated features, PCA projects data 
#   into fewer, uncorrelated features.

# -----------------------------
# How it works?
# -----------------------------
# 1. Standardize the data (mean=0, variance=1).
# 2. Compute covariance matrix of the data.
# 3. Find eigenvalues and eigenvectors of covariance matrix.
# 4. Eigenvectors = Principal Components (new axes).
# 5. Project the original data onto these principal components.
# 6. Keep top-k components (based on largest eigenvalues = max variance).

# -----------------------------
# Where to use it?
# -----------------------------
# - Data compression (reduce dimensionality).
# - Noise reduction.
# - Feature extraction.
# - Visualization of high-dimensional data (2D/3D plots).
# - Preprocessing step for ML algorithms.

# -----------------------------
# When to use it?
# -----------------------------
# - When dataset has high number of features.
# - When features are correlated.
# - When you want faster execution of ML algorithms.
# - When visualization in 2D/3D is needed.

# -----------------------------
# Why Variance is Important?
# -----------------------------
# - Variance represents how spread out data is.
# - More variance = more information.
# - PCA keeps the directions with maximum variance 
#   and discards less informative ones.

# -----------------------------
# Geometric Intuition
# -----------------------------
# - Imagine data points in 2D scattered in an ellipse shape.
# - PCA finds the axis (line) that best fits the "longest spread" 
#   of the data (first principal component).
# - Then it finds the next axis orthogonal to the first 
#   (second principal component).
# - In higher dimensions, PCA generalizes the same idea.

# -----------------------------
# Problem Formulation
# -----------------------------
# Goal: Reduce d-dimensional data into k-dimensional (k < d) 
# while retaining maximum variance.
#
# Step 1: Maximize variance in new coordinate system.
# Step 2: New axes must be orthogonal.
# Step 3: Use top-k eigenvectors as new basis.

# -----------------------------
# Step by Step Solution
# -----------------------------
# 1. Collect data (X).
# 2. Standardize X.
# 3. Compute Covariance matrix (Σ).
# 4. Find Eigenvalues (λ) and Eigenvectors (v) of Σ.
# 5. Sort eigenvectors by decreasing eigenvalues.
# 6. Select top-k eigenvectors.
# 7. Transform data into new subspace.

# -----------------------------
# Coding the Steps (Skeleton)
# -----------------------------
# import numpy as np
#
# # Step 1: Standardize data
# X_meaned = X - np.mean(X, axis=0)
#
# # Step 2: Covariance matrix
# cov_matrix = np.cov(X_meaned, rowvar=False)
#
# # Step 3: Eigen decomposition
# eigen_values, eigen_vectors = np.linalg.eigh(cov_matrix)
#
# # Step 4: Sort eigenvectors
# sorted_idx = np.argsort(eigen_values)[::-1]
# eigen_values = eigen_values[sorted_idx]
# eigen_vectors = eigen_vectors[:, sorted_idx]
#
# # Step 5: Select top-k
# k = 2
# eigen_vectors = eigen_vectors[:, :k]
#
# # Step 6: Transform data
# X_reduced = np.dot(X_meaned, eigen_vectors)

# -----------------------------
# Practical Example
# -----------------------------
# - Using PCA on MNIST dataset (handwritten digits).
# - MNIST has 784 dimensions (28x28 pixels).
# - PCA can reduce it to 50 dimensions for ML models.
# - Or reduce to 2 dimensions for visualization.

# -----------------------------
# Visualization of MNIST Dataset
# -----------------------------
# - Use PCA to project MNIST data into 2D.
# - Plot digits in 2D scatter plot.
# - Observe clusters forming (digits close to each other).

# -----------------------------
# Explained Variance
# -----------------------------
# - Eigenvalues represent variance captured by each component.
# - Explained Variance Ratio = (Eigenvalue_i / Sum of all Eigenvalues).
# - Helps decide how many components are enough.

# -----------------------------
# Finding Optimum Number of Components
# -----------------------------
# - Plot Cumulative Explained Variance.
# - Choose k where curve "elbows" (e.g., 95% variance).
# - This balances dimensionality reduction and information retention.

# ==============================================
# In summary:
# PCA reduces high-dimensional data into fewer dimensions 
# by keeping directions of maximum variance. 
# It improves efficiency, helps visualization, 
# and avoids redundancy in features.
# ==============================================
