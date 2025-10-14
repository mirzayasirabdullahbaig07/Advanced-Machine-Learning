# ================================================================
# BIAS-VARIANCE TRADEOFF (Overfitting vs Underfitting)
# ================================================================

# The Bias-Variance Tradeoff helps us understand how well a model
# generalizes to new, unseen data. 
# It's about finding the right balance between:
#   - Bias (error due to wrong assumptions)
#   - Variance (error due to sensitivity to training data)
#
# If bias is too high  -> model is too simple (Underfitting)
# If variance is too high -> model is too complex (Overfitting)
# Goal: Achieve a good balance for best generalization performance.

# ---------------------------------------------------------------
# BIAS AND VARIANCE EXPLAINED
# ---------------------------------------------------------------
# Bias:
#   - Error from overly simple assumptions in the model.
#   - Example: Using a linear model to fit non-linear data.
#   - Effect: Causes underfitting.
#
# Variance:
#   - Error from model’s sensitivity to small fluctuations in training data.
#   - Example: Deep neural network trained on small dataset.
#   - Effect: Causes overfitting.

# ---------------------------------------------------------------
# OVERFITTING
# ---------------------------------------------------------------
# - Model learns the training data too well, including noise.
# - High training accuracy, poor test accuracy.
# - Example: A student memorizes answers instead of understanding.
#
# Causes of Overfitting:
#   - Model too complex
#   - Too many parameters
#   - Not enough training data
#   - No regularization

# ---------------------------------------------------------------
# UNDERFITTING
# ---------------------------------------------------------------
# - Model is too simple to capture data patterns.
# - Poor performance on both training and test sets.
# - Example: A student studies too little and can't answer questions.
#
# Causes of Underfitting:
#   - Model too simple
#   - Missing important features
#   - Too few parameters

# ---------------------------------------------------------------
# TRAINING SET
# ---------------------------------------------------------------
# - The training set is the portion of data used to train the model.
# - The model learns the relationships between input features and outputs.
# - Common data split:
#       70–80% → Training Set
#       20–30% → Testing/Validation Set

# ---------------------------------------------------------------
# BIAS-VARIANCE TRADEOFF SUMMARY
# ---------------------------------------------------------------
# Simple Model  → High Bias, Low Variance  → Underfitting
# Complex Model → Low Bias, High Variance → Overfitting
# Optimal Model → Balanced Bias and Variance → Best Generalization

# ---------------------------------------------------------------
# BENEFITS OF UNDERSTANDING THE TRADEOFF
# ---------------------------------------------------------------
# - Helps choose the right model complexity
# - Improves model generalization
# - Prevents overfitting and underfitting
# - Optimizes performance

# ---------------------------------------------------------------
# DISADVANTAGES / CHALLENGES
# ---------------------------------------------------------------
# - Hard to achieve perfect balance in real-world data
# - Requires hyperparameter tuning
# - Complex models may be computationally expensive
# - Needs proper validation (e.g., cross-validation)
# ================================================================
