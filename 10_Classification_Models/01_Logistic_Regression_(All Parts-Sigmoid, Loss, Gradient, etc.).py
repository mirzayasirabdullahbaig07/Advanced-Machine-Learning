# Logistic Regression (Complete Intuition + Math + Code Roadmap)
# https://atifalikhokhar.my.canva.site/
# https://www.linkedin.com/in/atifalikhokhar/
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

# Logistic Regression Part 2 | Perceptron Trick Code

from sklearn.datasets import make_classification
import numpy as np
X, y = make_classification(n_samples=100, n_features=2, n_informative=1,n_redundant=0,
                           n_classes=2, n_clusters_per_class=1, random_state=41,hypercube=False,class_sep=10)
import matplotlib.pyplot as plt
plt.figure(figsize=(10,6))
plt.scatter(X[:,0],X[:,1],c=y,cmap='winter',s=100)
<matplotlib.collections.PathCollection at 0x1def70216d0>

def perceptron(X,y):
    
    X = np.insert(X,0,1,axis=1)
    weights = np.ones(X.shape[1])
    lr = 0.1
    
    for i in range(1000):
        j = np.random.randint(0,100)
        y_hat = step(np.dot(X[j],weights))
        weights = weights + lr*(y[j]-y_hat)*X[j]
        
    return weights[0],weights[1:]
        
def step(z):
    return 1 if z>0 else 0
intercept_,coef_ = perceptron(X,y)
print(coef_)
print(intercept_)
[1.44152475 0.10464821]
0.9
m = -(coef_[0]/coef_[1])
b = -(intercept_/coef_[1])
x_input = np.linspace(-3,3,100)
y_input = m*x_input + b
plt.figure(figsize=(10,6))
plt.plot(x_input,y_input,color='red',linewidth=3)
plt.scatter(X[:,0],X[:,1],c=y,cmap='winter',s=100)
plt.ylim(-3,2)
(-3.0, 2.0)

def perceptron(X,y):
    
    m = []
    b = []
    
    X = np.insert(X,0,1,axis=1)
    weights = np.ones(X.shape[1])
    lr = 0.1
    
    for i in range(200):
        j = np.random.randint(0,100)
        y_hat = step(np.dot(X[j],weights))
        weights = weights + lr*(y[j]-y_hat)*X[j]
        
        m.append(-(weights[1]/weights[2]))
        b.append(-(weights[0]/weights[2]))
        
    return m,b
m,b = perceptron(X,y)
%matplotlib notebook
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
fig, ax = plt.subplots(figsize=(9,5))

x_i = np.arange(-3, 3, 0.1)
y_i = x_i*m[0] +b[0]
ax.scatter(X[:,0],X[:,1],c=y,cmap='winter',s=100)
line, = ax.plot(x_i, x_i*m[0] +b[0] , 'r-', linewidth=2)
plt.ylim(-3,3)
def update(i):
    label = 'epoch {0}'.format(i + 1)
    line.set_ydata(x_i*m[i] + b[i])
    ax.set_xlabel(label)
    # return line, ax

anim = FuncAnimation(fig, update, repeat=True, frames=200, interval=100)

from sklearn.linear_model import LogisticRegression
lor = LogisticRegression()
lor.fit(X,y)
LogisticRegression()
m = -(lor.coef_[0][0]/lor.coef_[0][1])
b = -(lor.intercept_/lor.coef_[0][1])
x_input1 = np.linspace(-3,3,100)
y_input1 = m*x_input + b
plt.figure(figsize=(10,6))
plt.plot(x_input,y_input,color='red',linewidth=3)
plt.plot(x_input1,y_input1,color='black',linewidth=3)
plt.scatter(X[:,0],X[:,1],c=y,cmap='winter',s=100)
plt.ylim(-3,2)
(-3.0, 2.0)

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
# 🔹 MLE → find parameters that maximize likelihood of data.
# 🔹 Taking log → log-likelihood → easier to differentiate.
# 🔹 Negating → gives us binary cross-entropy loss.
# 🔹 Minimizing BCE = Maximizing data likelihood = Better model.
# --------------------------------------------------------------
