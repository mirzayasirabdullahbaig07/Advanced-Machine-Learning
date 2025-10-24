# Logistic Regression (Complete Intuition + Math + Code Roadmap)

# Logistic Regression is a supervised learning algorithm used for binary classification problems (0 or 1 output).
# It predicts the probability that a given input belongs to a particular class using the **Sigmoid Function**.

# ---------------------------------------------------------------
# STEP 1: Equation of Logistic Regression
# ---------------------------------------------------------------
# Just like Linear Regression:
#     z = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ
# But instead of predicting continuous values, we pass 'z' into a **sigmoid** function to get a probability between 0 and 1.

#     ŷ = σ(z) = 1 / (1 + e^(-z))
# where:
#     σ(z) is the sigmoid function
#     ŷ (y-hat) = predicted probability (0 < ŷ < 1)

# ---------------------------------------------------------------
# STEP 2: Decision Boundary (Perceptron Trick)
# ---------------------------------------------------------------
# The Perceptron rule helps define how we label regions in space:
#     If (w · x + b) > 0   → Positive Class (y = 1)
#     If (w · x + b) < 0   → Negative Class (y = 0)
#     If (w · x + b) = 0   → Point lies exactly on the decision boundary

# The line (or hyperplane) that separates the classes is:
#     w₁x₁ + w₂x₂ + b = 0
# where w₁ and w₂ define slope and b (bias) shifts the line.

# --------------------------------------------------------------
# STEP 3: Moving the Line (Effect of Parameters)
# ---------------------------------------------------------------
# - Changing **b (bias)** → moves the line **up or down** (parallel shift)
# - Changing **w₁ (weight of x₁)** → rotates line horizontally (affects slope)
# - Changing **w₂ (weight of x₂)** → rotates line vertically (affects slope)
# Example: If you increase b → line moves downward; decrease b → line moves upward.

# ---------------------------------------------------------------
# STEP 4: Sigmoid Transformation
# ---------------------------------------------------------------
# The sigmoid function “squeezes” all real-valued inputs (z) into the range (0,1)
# making it interpretable as a probability.

# If ŷ >= 0.5 → predict 1 (Positive class)
# If ŷ < 0.5  → predict 0 (Negative class)

# ---------------------------------------------------------------
# STEP 5: Loss Function (Binary Cross Entropy)
# ---------------------------------------------------------------
# Loss = -[ y*log(ŷ) + (1 - y)*log(1 - ŷ) ]
# It penalizes wrong predictions more heavily when the model is confident but wrong.

# ---------------------------------------------------------------
# STEP 6: Gradient Descent Update Rule
# ---------------------------------------------------------------
# For each weight wᵢ:
#     wᵢ_new = wᵢ_old - η * ∂(Loss)/∂wᵢ
# where η (eta) = learning rate

# The gradient is computed using the derivative of the sigmoid and loss function:
#     ∂(Loss)/∂wᵢ = (ŷ - y) * xᵢ
# So,
#     w_new = w_old - η * (ŷ - y) * xᵢ

# ---------------------------------------------------------------
# In Summary:
# Logistic Regression = Linear Model + Sigmoid Transformation
# Step-by-step:
# 1. Compute z = w·x + b
# 2. Apply sigmoid: ŷ = 1 / (1 + e^(-z))
# 3. Compute loss: -[ y*log(ŷ) + (1 - y)*log(1 - ŷ) ]
# 4. Update weights using gradient descent

# ---------------------------------------------------------------
# This concept connects with Perceptron logic (from your image) where:
# - if (∑ wᵢxᵢ > 0 and sample in negative region) → w_new = w_old - ηxᵢ
# - if (∑ wᵢxᵢ < 0 and sample in positive region) → w_new = w_old + ηxᵢ
# which generalizes to:
#     w_new = w_old + η * (y - ŷ) * xᵢ
# ---------------------------------------------------------------

# ==========================================================
# Logistic Regression Part 2 | Perceptron Trick Visualization
# ==========================================================

# Import Required Libraries
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ----------------------------------------------------------
# STEP 1: Generate a Binary Classification Dataset
# ----------------------------------------------------------
X, y = make_classification(
    n_samples=100,
    n_features=2,
    n_informative=1,
    n_redundant=0,
    n_classes=2,
    n_clusters_per_class=1,
    random_state=41,
    hypercube=False,
    class_sep=10
)

# Visualize Dataset
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='winter', s=100)
plt.title("Generated Binary Classification Data", fontsize=14)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# ----------------------------------------------------------
# STEP 2: Define Perceptron Model
# ----------------------------------------------------------
def step(z):
    """Step activation function."""
    return 1 if z > 0 else 0


def perceptron(X, y):
    """Implements the perceptron learning algorithm."""
    X = np.insert(X, 0, 1, axis=1)   # Add bias term
    weights = np.ones(X.shape[1])    # Initialize weights
    lr = 0.1                         # Learning rate

    # Training for 1000 iterations
    for _ in range(1000):
        j = np.random.randint(0, len(y))
        y_hat = step(np.dot(X[j], weights))
        weights = weights + lr * (y[j] - y_hat) * X[j]

    return weights[0], weights[1:]


# Train Perceptron
intercept_, coef_ = perceptron(X, y)
print("Weights:", coef_)
print("Intercept:", intercept_)

# Decision Boundary
m = -(coef_[0] / coef_[1])
b = -(intercept_ / coef_[1])

x_input = np.linspace(-3, 3, 100)
y_input = m * x_input + b

# Plot the Perceptron Decision Boundary
plt.figure(figsize=(10, 6))
plt.plot(x_input, y_input, color='red', linewidth=3, label="Perceptron Boundary")
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='winter', s=100)
plt.ylim(-3, 2)
plt.legend()
plt.title("Perceptron Decision Boundary")
plt.show()


# ----------------------------------------------------------
# STEP 3: Visualize Learning (Perceptron Trick)
# ----------------------------------------------------------
def perceptron_animation(X, y):
    """Track slope (m) and intercept (b) during training for visualization."""
    m_vals, b_vals = [], []
    X = np.insert(X, 0, 1, axis=1)
    weights = np.ones(X.shape[1])
    lr = 0.1

    for _ in range(200):
        j = np.random.randint(0, len(y))
        y_hat = step(np.dot(X[j], weights))
        weights = weights + lr * (y[j] - y_hat) * X[j]
        m_vals.append(-(weights[1] / weights[2]))
        b_vals.append(-(weights[0] / weights[2]))

    return m_vals, b_vals


m_vals, b_vals = perceptron_animation(X, y)

# Create Animation
fig, ax = plt.subplots(figsize=(9, 5))
x_i = np.arange(-3, 3, 0.1)
line, = ax.plot(x_i, x_i * m_vals[0] + b_vals[0], 'r-', linewidth=2)
ax.scatter(X[:, 0], X[:, 1], c=y, cmap='winter', s=100)
plt.ylim(-3, 3)

def update(i):
    """Update line for each epoch."""
    label = f'Epoch {i + 1}'
    line.set_ydata(x_i * m_vals[i] + b_vals[i])
    ax.set_xlabel(label)
    return line, ax

anim = FuncAnimation(fig, update, repeat=True, frames=200, interval=100)
plt.show()


# ----------------------------------------------------------
# STEP 4: Compare with Logistic Regression
# ----------------------------------------------------------
lor = LogisticRegression()
lor.fit(X, y)

# Logistic Regression Line
m_lr = -(lor.coef_[0][0] / lor.coef_[0][1])
b_lr = -(lor.intercept_ / lor.coef_[0][1])
x_input1 = np.linspace(-3, 3, 100)
y_input1 = m_lr * x_input1 + b_lr

# Plot Both Decision Boundaries
plt.figure(figsize=(10, 6))
plt.plot(x_input, y_input, color='red', linewidth=3, label="Perceptron")
plt.plot(x_input1, y_input1, color='black', linewidth=3, label="Logistic Regression")
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='winter', s=100)
plt.ylim(-3, 2)
plt.legend()
plt.title("Comparison: Perceptron vs Logistic Regression")
plt.show()

# --------------------------------------------------------------
# Logistic Regression — Loss Function, MLE, and Cross-Entropy
# --------------------------------------------------------------

# Logistic Regression is used for binary classification problems.
# It predicts probabilities between 0 and 1 using the sigmoid function.

# --------------------------------------------------------------
# Step 1: The Model
# --------------------------------------------------------------
# Suppose we have input features X and target labels y (0 or 1).
# The model predicts probability that y = 1 as:
#     P(y=1 | x) = σ(z)
# where z = w·x + b
# and σ(z) = 1 / (1 + e^(-z))   ← sigmoid function

# --------------------------------------------------------------
# Step 2: Likelihood Function (Maximum Likelihood Estimation)
# --------------------------------------------------------------
# We want to find parameters (w, b) that make the observed data most likely.
# For each training example (x_i, y_i):
#     P(y_i | x_i) = (σ(z_i))^y_i * (1 - σ(z_i))^(1 - y_i)
# Why this formula?
# - If y_i = 1, it keeps σ(z_i)
# - If y_i = 0, it keeps (1 - σ(z_i))

# For all samples, the likelihood L is the product of all probabilities:
#     L = Π [ (σ(z_i))^y_i * (1 - σ(z_i))^(1 - y_i) ]

# --------------------------------------------------------------
# Step 3: Log-Likelihood (for easier math)
# --------------------------------------------------------------
# Taking log of likelihood (because log turns products into sums):
#     log(L) = Σ [ y_i * log(σ(z_i)) + (1 - y_i) * log(1 - σ(z_i)) ]

# We want to maximize log(L).
# However, optimization libraries usually *minimize* loss functions,
# so we take the negative of the log-likelihood.

# --------------------------------------------------------------
# Step 4: Binary Cross-Entropy Loss (Negative Log-Likelihood)
# --------------------------------------------------------------
# Loss = - (1/N) * Σ [ y_i * log(σ(z_i)) + (1 - y_i) * log(1 - σ(z_i)) ]
# This is known as:
#      Binary Cross-Entropy Loss
# or
#      Log Loss
# It penalizes wrong confident predictions heavily.

# --------------------------------------------------------------
# Step 5: Why do we use it?
# --------------------------------------------------------------
# Because maximizing the likelihood of data is equivalent to
# minimizing the cross-entropy loss.
# It helps our logistic regression model learn parameters (w, b)
# that make predicted probabilities as close as possible to true labels.

# --------------------------------------------------------------
# Step 6: Example in Python (with math)
# --------------------------------------------------------------
import numpy as np

# Sample data
X = np.array([[1], [2], [3], [4]])  # feature
y = np.array([0, 0, 1, 1])          # target labels

# Initialize weights
w = np.random.randn()
b = 0

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Predictions
z = w * X + b
y_pred = sigmoid(z)

# Binary Cross-Entropy Loss
loss = -np.mean(y * np.log(y_pred + 1e-9) + (1 - y) * np.log(1 - y_pred + 1e-9))

print("Predicted Probabilities:", y_pred)
print("Binary Cross Entropy Loss:", loss)

# --------------------------------------------------------------
# Step 7: Gradient Descent Intuition
# --------------------------------------------------------------
# To minimize the loss, we compute gradients:
#     ∂L/∂w = (1/N) * Σ (σ(z_i) - y_i) * x_i
#     ∂L/∂b = (1/N) * Σ (σ(z_i) - y_i)
# Then update:
#     w = w - α * ∂L/∂w
#     b = b - α * ∂L/∂b
# where α is the learning rate.

# --------------------------------------------------------------
# Step 8: Summary
# --------------------------------------------------------------
# MLE → find parameters that maximize likelihood of data.
# Taking log → log-likelihood → easier to differentiate.
# Negating → gives us binary cross-entropy loss.
# Minimizing BCE = Maximizing data likelihood = Better model.
# --------------------------------------------------------------

# ==============================================
# Derivation of the Sigmoid Function Derivative
# ==============================================

# Let's start from the definition of the sigmoid function:
# σ(x) = 1 / (1 + e^(-x))

# -----------------------------------------------
# Step 1: Write it in a form that's easier to differentiate
# -----------------------------------------------
# σ(x) = (1 + e^(-x))^(-1)

# -----------------------------------------------
# Step 2: Differentiate using the chain rule
# -----------------------------------------------
# dσ(x)/dx = -1 * (1 + e^(-x))^(-2) * d/dx(1 + e^(-x))

# -----------------------------------------------
# Step 3: Derivative of (1 + e^(-x)) with respect to x
# -----------------------------------------------
# d/dx(1 + e^(-x)) = -e^(-x)

# -----------------------------------------------
# Step 4: Substitute this back into the equation
# -----------------------------------------------
# dσ(x)/dx = -1 * (1 + e^(-x))^(-2) * (-e^(-x))
# Simplify the negatives:
# dσ(x)/dx = e^(-x) / (1 + e^(-x))^2

# -----------------------------------------------
# Step 5: Multiply and divide by (1 + e^(-x))
# to express in terms of σ(x)
# -----------------------------------------------
# We know:
# σ(x) = 1 / (1 + e^(-x))
# Therefore:
# 1 - σ(x) = e^(-x) / (1 + e^(-x))

# Substitute these forms:
# dσ(x)/dx = σ(x) * (1 - σ(x))

# -----------------------------------------------
# Final Simplified Derivative:
# -----------------------------------------------
# σ'(x) = σ(x) * (1 - σ(x))
# This is the most compact and elegant form of the derivative.

# -----------------------------------------------
# Step 6: Verify with Python Implementation
# -----------------------------------------------

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

# Test
x = 0
print("σ(x) =", sigmoid(x))           # Output: 0.5
print("σ'(x) =", sigmoid_derivative(x))  # Output: 0.25

# -----------------------------------------------
# Step 7: Intuition Behind the Derivative
# -----------------------------------------------
# - The derivative is highest at x = 0 → σ'(0) = 0.25
# - For very large +x or -x, σ'(x) ≈ 0
#   → meaning the neuron "saturates" (gradient vanishes)
# - That’s why sigmoid is less used in deep networks’ hidden layers
#   but still used in output layers of binary classification.

# ==========================================================
# Logistic Regression (All Parts: Sigmoid, Loss, Gradient Descent)
# ==========================================================

# ----------------------------------------------------------
# WHEN TO USE LOGISTIC REGRESSION
# ----------------------------------------------------------
# Logistic Regression is used for binary or multi-class classification.
# Example use-cases:
# - Spam (1) or Not Spam (0)
# - Disease (1) or No Disease (0)
# - Pass (1) or Fail (0)
#
# It works well when:
# - Relationship between features and target is approximately linear.
# - The output needs to be a probability between 0 and 1.
# ----------------------------------------------------------


# ----------------------------------------------------------
# HOW LOGISTIC REGRESSION WORKS
# ----------------------------------------------------------
# Step 1: Compute a linear combination (z):
#          z = w1*x1 + w2*x2 + ... + b
# Step 2: Apply the sigmoid activation:
#          σ(z) = 1 / (1 + e^(-z))
# Step 3: Predict class based on probability:
#          y_pred = 1 if σ(z) > 0.5 else 0
# ----------------------------------------------------------


# ----------------------------------------------------------
# FORMULA (Sigmoid + Loss + Gradient)
# ----------------------------------------------------------
# Sigmoid Function: σ(z) = 1 / (1 + e^(-z))
# Loss Function (Log Loss):
#   L = - (1/m) * Σ [ y*log(y_hat) + (1-y)*log(1-y_hat) ]
# Gradient Descent Updates:
#   w = w - α * (1/m) * Σ( (y_hat - y) * x )
#   b = b - α * (1/m) * Σ( (y_hat - y) )
# ----------------------------------------------------------


# ----------------------------------------------------------
# STEP 1: Import Libraries and Create Data
# ----------------------------------------------------------
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt

# Generate a simple binary classification dataset
X, y = make_classification(
    n_samples=100,
    n_features=2,
    n_informative=1,
    n_redundant=0,
    n_classes=2,
    n_clusters_per_class=1,
    random_state=41,
    hypercube=False,
    class_sep=20
)

# Visualize dataset
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='winter', s=100)
plt.title("Generated Binary Classification Dataset", fontsize=14)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()


# ----------------------------------------------------------
# STEP 2: Logistic Regression using sklearn (for reference)
# ----------------------------------------------------------
lor = LogisticRegression(penalty='none', solver='sag')
lor.fit(X, y)

print("Weights (coefficients):", lor.coef_)
print("Intercept (bias):", lor.intercept_)

# Equation of decision boundary:
# w1*x1 + w2*x2 + b = 0  →  x2 = -(w1/w2)x1 - (b/w2)
m1 = -(lor.coef_[0][0] / lor.coef_[0][1])
b1 = -(lor.intercept_ / lor.coef_[0][1])

x_input = np.linspace(-3, 3, 100)
y_input = m1 * x_input + b1

# Plot decision boundary from sklearn
plt.figure(figsize=(10, 6))
plt.plot(x_input, y_input, color='red', linewidth=3, label="sklearn Logistic Regression")
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='winter', s=100)
plt.title("Decision Boundary using Logistic Regression (sklearn)")
plt.legend()
plt.ylim(-3, 2)
plt.show()


# ----------------------------------------------------------
# STEP 3: Implement Logistic Regression from Scratch
# ----------------------------------------------------------

def sigmoid(z):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-z))


def gd(X, y):
    """
    Gradient Descent for Logistic Regression
    - Adds bias column to X
    - Initializes weights
    - Iteratively updates weights using gradients
    """
    X = np.insert(X, 0, 1, axis=1)  # Add bias column
    weights = np.ones(X.shape[1])   # Initialize weights
    lr = 0.5                        # Learning rate

    # Iterate over epochs
    for i in range(5000):
        # Forward propagation (prediction)
        y_hat = sigmoid(np.dot(X, weights))
        
        # Weight update rule (Gradient Descent)
        weights = weights + lr * (np.dot((y - y_hat), X) / X.shape[0])

    # Return slope and intercept
    return weights[1:], weights[0]


# Train model using custom gradient descent
coef_, intercept_ = gd(X, y)

# Compute line parameters
m2 = -(coef_[0] / coef_[1])
b2 = -(intercept_ / coef_[1])

x_input1 = np.linspace(-3, 3, 100)
y_input1 = m2 * x_input1 + b2

# ----------------------------------------------------------
# STEP 4: Plot Comparison (Sklearn vs Scratch)
# ----------------------------------------------------------
plt.figure(figsize=(10, 6))
plt.plot(x_input, y_input, color='red', linewidth=3, label="sklearn Logistic Regression")
plt.plot(x_input1, y_input1, color='black', linewidth=3, label="From Scratch (Gradient Descent)")
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='winter', s=100)
plt.title("Comparison: Logistic Regression (sklearn) vs From Scratch")
plt.legend()
plt.ylim(-3, 2)
plt.show()


# ----------------------------------------------------------
# STEP 5: Understanding Input Transformation
# ----------------------------------------------------------
# When we add a bias column:
# Original X shape = (100, 2)
# After insertion -> (100, 3)
# Each row: [1, x1, x2]
# This allows the bias term to be included in the dot product.
X1 = np.insert(X, 0, 1, axis=1)
print("X after bias column insertion:\n", X1[:5])
print("Initial weights (ones):", np.ones(X1.shape[1]))
