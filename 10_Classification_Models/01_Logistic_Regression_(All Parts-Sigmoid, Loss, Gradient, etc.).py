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
