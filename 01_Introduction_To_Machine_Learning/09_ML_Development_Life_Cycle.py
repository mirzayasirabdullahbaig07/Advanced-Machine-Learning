# ================================
# Machine Learning Development Life Cycle (MLDLC)
# ================================
# Just like Software Development Life Cycle (SDLC),
# ML projects follow a structured cycle called MLDLC.
# It has ~9 steps from problem framing to optimization.

# ------------------------------------------------
# 1. Frame the Problem (Define Objective)
# ------------------------------------------------
# - First, define what problem we are solving using ML.
# - Decide if it's classification, regression, clustering, etc.
# Example:
#   - Predicting customer churn (classification: churn or not churn)
#   - Forecasting house prices (regression)
#   - Grouping customers (clustering for segmentation)

# ------------------------------------------------
# 2. Data Gathering (Collection)
# ------------------------------------------------
# - Collect raw data from various sources:
#   - CSV files
#   - Databases
#   - APIs
#   - Web scraping
#   - IoT sensors
# - Usually, data is stored in data warehouses after ETL (Extract, Transform, Load).
# Example:
#   - Customer transaction logs from a bank
#   - Tweets from Twitter API

# ------------------------------------------------
# 3. Data Preprocessing (Cleaning)
# ------------------------------------------------
# - Raw data is messy; we need to clean it:
#   - Remove duplicates
#   - Handle missing values (imputation or deletion)
#   - Remove outliers (e.g., salary = 9999999 in student data)
#   - Normalize/standardize numerical features
# Example:
#   - Replacing NaN in "Age" column with average age
#   - Scaling income column between 0–1

# ------------------------------------------------
# 4. Exploratory Data Analysis (EDA)
# ------------------------------------------------
# - Analyze and visualize data to understand patterns:
#   - Univariate analysis (single column: histograms, mean, std)
#   - Bivariate analysis (scatter plots, correlation heatmap)
#   - Graphs and summary statistics
# Example:
#   - Plot salary vs experience to check linear relationship
#   - Correlation between "loan amount" and "default rate"

# ------------------------------------------------
# 5. Feature Engineering & Feature Selection
# ------------------------------------------------
# - Create new features, remove irrelevant ones, or transform data.
# - Feature Engineering:
#     - Extract year from a "Date" column
#     - Create BMI from "Height" and "Weight"
# - Feature Selection:
#     - Drop highly correlated columns
#     - Use methods like Chi-square, PCA
# Example:
#   - In e-commerce: create "total_spent" = price × quantity

# ------------------------------------------------
# 6. Model Training, Evaluation, & Selection
# ------------------------------------------------
# - Train ML models on preprocessed data.
# - Try different algorithms:
#     - Logistic Regression
#     - Decision Trees
#     - Random Forest
#     - Naive Bayes
#     - Neural Networks
# - Evaluate using metrics:
#     - Classification → Accuracy, Precision, Recall, F1-score
#     - Regression → RMSE, MAE, R²
# - Select the best model based on performance.
# Example:
#   - Churn prediction → Random Forest performed best with 89% accuracy

# ------------------------------------------------
# 7. Model Deployment
# ------------------------------------------------
# - Deploy the model in real-world environment:
#   - APIs (Flask, FastAPI)
#   - Streamlit/Dash apps
#   - Cloud services (AWS Sagemaker, GCP AI, Azure ML)
# Example:
#   - Loan approval model deployed as REST API for banking system

# ------------------------------------------------
# 8. Model Testing
# ------------------------------------------------
# - Check how the deployed model performs with new unseen data.
# - Test stability, accuracy, latency, and scalability.
# Example:
#   - A credit scoring model tested on 10,000 new loan applications

# ------------------------------------------------
# 9. Model Optimization & Maintenance
# ------------------------------------------------
# - Improve performance by:
#     - Hyperparameter tuning (GridSearch, RandomSearch, Bayesian optimization)
#     - Retraining with new data
#     - Model compression (for edge deployment)
# - Continuous monitoring for drift (data distribution change).
# Example:
#   - Retraining recommendation engine every month as new products arrive.

# ================================
# Difference Between SDLC and MLDLC
# ================================
# SDLC:
# - Traditional software development cycle (requirements, design, coding, testing, deployment).
# - Works with deterministic logic.

# MLDLC:
# - Works with data-driven models.
# - Includes special steps like Data Gathering, EDA, Feature Engineering, and Model Optimization.
# - Models improve as more data comes in.

# ================================
# Summary
# ================================
# MLDLC has 9 key steps:
# 1. Frame the Problem
# 2. Gather Data
# 3. Preprocess Data
# 4. Perform EDA
# 5. Feature Engineering & Selection
# 6. Train, Evaluate, and Select Model
# 7. Deploy Model
# 8. Test Model
# 9. Optimize and Maintain Model
