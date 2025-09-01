# ===========================================
# Online Machine Learning
# ===========================================

# Definition:
# - Online Learning is when a model is continuously updated with new data.
# - The model dynamically evolves instead of being static.
# - Works well in environments where data arrives in a stream (real-time).

# Example:
# - Chatbots: Predict responses and continuously improve as new conversations are added.
# - Stock price prediction, fraud detection, recommendation systems.


# ===========================================
# Batch Learning vs Online Learning
# ===========================================

# Batch (Offline) Learning:
# - Model trained on a fixed dataset (static).
# - Must be retrained with old + new data to update.
# - Simpler, easier to implement, fewer computations after training.
# - Good when data patterns are stable (e.g., image classification).

# Online Learning:
# - Model updates itself incrementally with new incoming data.
# - Dynamic → evolves as new patterns emerge.
# - More complex, requires continuous computation and monitoring.
# - Useful in cases of "concept drift" (when data patterns change over time).


# ===========================================
# When to Use
# ===========================================

# Use Batch Learning:
# - When data is static and does not change frequently.
# - When retraining costs are acceptable.
# - Example: Predicting house prices, image classification.

# Use Online Learning:
# - When data arrives continuously (streaming).
# - When patterns change over time (concept drift).
# - When cost-effective real-time solutions are needed.
# - Example: Fraud detection, recommendation systems, finance & economics.


# ===========================================
# Key Concepts in Online Learning
# ===========================================

# 1. Learning Rate:
# - Defines how much the model updates with each new data point.
# - Too high → model may forget old knowledge (unstable).
# - Too low → model learns too slowly (inefficient).

# 2. Out-of-Core Learning:
# - Used when dataset is too large to fit into memory.
# - Data is divided into mini-batches (subsets), and the model learns incrementally.
# - Example: SGD (Stochastic Gradient Descent) with partial_fit() in scikit-learn.


# ===========================================
# Disadvantages of Online Learning
# ===========================================

# - Complex to implement and maintain.
# - Risky: Model can learn wrong patterns quickly if incoming data is noisy.
# - Requires monitoring, validation, and fine-tuned learning rate.
# - Computationally expensive due to continuous updates.


# ===========================================
# Summary
# ===========================================
# - Batch ML → Train once on static data, retrain periodically. Simple but static.
# - Online ML → Learn continuously from streaming data. Dynamic but complex.
# - Choose based on data nature:
#   → Stable data → Batch ML
#   → Streaming + changing patterns → Online ML
