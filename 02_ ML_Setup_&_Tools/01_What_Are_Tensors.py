# ============================================================
# TENSORS IN MACHINE LEARNING & DEEP LEARNING
# ============================================================

# What is a Tensor?
# A tensor is simply a data structure (like arrays in NumPy) used to store numbers.
# - Scalars, Vectors, and Matrices are all special cases of tensors.
# - In ML/DL, all data (images, text, audio, video, etc.) is represented as tensors.
# - A Tensor = n-dimensional array.

# Why study Tensors?
# Because all computations in Machine Learning & Deep Learning use tensors.
# Examples:
# - A single prediction loss value = scalar tensor
# - A list of features = vector tensor
# - A dataset (rows × columns) = matrix tensor
# - Images, batches, videos = higher-dimensional tensors

import numpy as np

# ============================================================
# 0D Tensor → Scalar
# ============================================================
# A scalar is just a single number (0D tensor).
# Rank = 0, Axis = 0
a = np.array(5)
print("Scalar:", a)
print("Rank (ndim):", a.ndim)     # 0 → scalar has no dimensions
print("Shape:", a.shape)          # () → empty tuple
print("Size:", a.size)            # 1 element only

# ML Example: Loss value (e.g., 0.25) is a scalar tensor


# ============================================================
# 1D Tensor → Vector
# ============================================================
# A vector is a collection of scalars (1D tensor).
# Rank = 1, Axis = 1
arr = np.array([1, 2, 3, 4])
print("\nVector:", arr)
print("Rank (ndim):", arr.ndim)   # 1
print("Shape:", arr.shape)        # (4,) → 4 elements along one axis
print("Size:", arr.size)          # 4 total elements

# ML Example: A student's features → [height, weight, age]


# ============================================================
# 2D Tensor → Matrix
# ============================================================
# A matrix is a 2D tensor (rows × columns).
# Rank = 2, Axis = 2
mat = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])
print("\nMatrix:\n", mat)
print("Rank (ndim):", mat.ndim)   # 2
print("Shape:", mat.shape)        # (3,3) → 3 rows, 3 cols
print("Size:", mat.size)          # 9 total elements

# ML Example: A dataset of students
# NLP
# [[height, weight, age],
#  [height, weight, age],
#  [height, weight, age]]


# ============================================================
# 3D Tensor
# ============================================================
# A stack of matrices → like pages in a book.
# Rank = 3, Axis = 3
tensor3D = np.array([[[1, 2, 3],
                      [4, 5, 6]],

                     [[7, 8, 9],
                      [10, 11, 12]]])
print("\n3D Tensor:\n", tensor3D)
print("Rank (ndim):", tensor3D.ndim)  # 3
print("Shape:", tensor3D.shape)       # (2, 2, 3)
print("Size:", tensor3D.size)         # 12 elements

# ML Example: A color image (RGB) → shape = (height, width, channels) Time Series


# ============================================================
# 4D Tensor
# ============================================================
# Collection of 3D tensors → batch of images.
# Rank = 4, Axis = 4
tensor4D = np.random.rand(10, 64, 64, 3)  # 10 images, 64×64 pixels, 3 channels (RGB)
print("\n4D Tensor Shape:", tensor4D.shape)

# ML Example: Batch of images
# shape = (batch_size, height, width, channels)


# ============================================================
# 5D Tensor
# ============================================================
# Collection of 4D tensors → video dataset.
# Rank = 5, Axis = 5
tensor5D = np.random.rand(5, 10, 64, 64, 3)
# 5 videos, 10 frames each, 64×64 pixels, RGB
print("\n5D Tensor Shape:", tensor5D.shape)

# ML Example: Video dataset
# shape = (videos, frames, height, width, channels)


# ============================================================
# General Notes
# ============================================================
# - Rank = number of axes (ndim)
# - Axis = direction in a tensor
# - Shape = size along each axis
# - Size = total number of elements in tensor
#
# Quick Recap:
# 0D → Scalar (single number)
# 1D → Vector (list of numbers/features)
# 2D → Matrix (dataset, weight matrix)
# 3D → Tensor (images, time series data)
# 4D → Batch of images
# 5D → Video dataset
