# ===========================================
# Production vs Deployment Environment
# ===========================================

# Production Environment:
# - The real-world environment where the model or application is actively used by end-users.
# - Example: A fraud detection ML model running in a banking system, checking transactions in real-time.
# - Must be stable, reliable, and secure.

# Deployment Environment:
# - The environment where we prepare and test our model before going live.
# - It acts as a "staging ground" for deployment.
# - Example: Deploying a trained ML model on cloud (AWS, GCP, Azure) or local servers 
#            to test APIs, integration, and scalability before moving to production.


# ===========================================
# Batch Machine Learning
# ===========================================

# Definition:
# - In batch ML, a model is trained on a fixed dataset (historical data).
# - Once trained, the model becomes static (does not update with new data automatically).
# - To improve the model, it must be retrained with new + old data at regular intervals.

# Example:
# - Predicting customer churn:
#   Model is trained on last year’s customer data.
#   After 6 months, new data arrives → model must be retrained to stay accurate.


# ===========================================
# Problems with Batch Machine Learning
# ===========================================

# 1. Static Learning:
#    - The model cannot learn continuously from incoming data.
#    - Needs manual retraining after some time.

# 2. Data Storage Issues:
#    - Requires large datasets to retrain.
#    - Storing and managing historical data can be expensive.

# 3. Hardware Limitations:
#    - Training on large datasets requires powerful CPUs/GPUs and large memory.

# 4. Availability Issues:
#    - Retraining takes time, so the model may be unavailable or outdated during retraining.

# ===========================================
# Summary
# ===========================================
# - Production Environment → Real-world usage by end-users.
# - Deployment Environment → Testing/hosting ground before production.
# - Batch Machine Learning → Model trained on static data, retrained periodically.
# - Disadvantages → Static learning, heavy storage needs, hardware limitations, downtime.

