# ===========================================
# Types of Machine Learning
# ===========================================

# Machine Learning is generally categorized into 4 types:
# 1. Supervised Learning
# 2. Unsupervised Learning
# 3. Semi-Supervised Learning
# 4. Reinforcement Learning


# -------------------------------------------
# 1. Supervised Machine Learning
# -------------------------------------------

# Definition:
# - The model is trained on labeled data (input + output).
# - The algorithm learns mapping from input → output.

# Example dataset (students):
#   Features (Input): IQ, CGPA
#   Target (Output): Placement (Yes/No)

# Types of Data:
# - Numerical Data (continuous values like salary, marks, temperature)
# - Categorical Data (discrete labels like Yes/No, Male/Female, Colors)

# Two Main Tasks:
# 1. Regression → Predict numerical values
#    Example: Predicting house prices, predicting salary from experience
#
# 2. Classification → Predict categorical values
#    Example: Spam (Yes/No), Disease detection (Positive/Negative),
#             Image classification (Dog/Cat)


# -------------------------------------------
# 2. Unsupervised Machine Learning
# -------------------------------------------

# Definition:
# - Only input data is available, no labeled output.
# - Goal: Find hidden patterns or groupings in data.

# Key Techniques:
# 1. Clustering → Grouping similar data points together.
#    Example: Customer segmentation in marketing, grouping news articles.
#
# 2. Dimensionality Reduction → Reducing number of features while keeping 
#    important information.
#    Example: PCA (Principal Component Analysis) to reduce columns in high-dimensional datasets.
#
# 3. Anomaly Detection → Detecting unusual patterns/outliers.
#    Example: Fraud detection in credit cards, network intrusion detection.
#
# 4. Association Rule Learning → Finding relationships between variables.
#    Example: "People who buy milk also buy bread" (Market Basket Analysis).


# -------------------------------------------
# 3. Semi-Supervised Machine Learning
# -------------------------------------------

# Definition:
# - A mix of supervised and unsupervised learning.
# - Uses a small amount of labeled data + a large amount of unlabeled data.
# - Helps when manual labeling is expensive or time-consuming.

# Example:
# - Label a small portion of medical images manually,
#   and let ML model learn the rest from unlabeled images.


# -------------------------------------------
# 4. Reinforcement Learning
# -------------------------------------------

# Definition:
# - Learning through interaction with an environment.
# - An "Agent" learns by performing actions and receiving feedback 
#   in the form of rewards (positive) or punishments (negative).

# Key Idea:
# - Agent → Environment → Reward/Punishment → Learn optimal actions.

# Example:
# - Self-driving cars (agent learns to drive safely by trial and error).
# - Game playing AI (Chess, Go, Atari games).
# - Robotics navigation.

# ===========================================
# Summary
# ===========================================
# 1. Supervised → Learn from labeled data (Regression, Classification).
# 2. Unsupervised → Find patterns in unlabeled data (Clustering, PCA, Anomaly detection).
# 3. Semi-Supervised → Mix of labeled + unlabeled data.
# 4. Reinforcement → Learn by interacting with environment (Rewards/Punishments).

