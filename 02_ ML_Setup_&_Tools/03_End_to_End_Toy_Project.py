# # ============================================================
# # End-to-End Machine Learning Toy Project
# # ============================================================
# # Dataset: placement.csv (contains cgpa, iq, and placement label)
# # Goal: Predict whether a student gets placement based on CGPA & IQ
# # ============================================================

# # Import required libraries
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# # ============================================================
# # Load Dataset
# # ============================================================
# df = pd.read_csv('/content/placement.csv')   # Load data
# print(df.head())                             # View first 5 rows
# print("Shape of dataset:", df.shape)         # Rows × Columns
# print(df.info())                             # Data summary

# # Drop first column if unnecessary (like ID or serial number)
# df = df.iloc[:, 1:]

# # ============================================================
# # Exploratory Data Analysis (EDA)
# # ============================================================
# # Scatter plot: CGPA vs IQ, color-coded by placement
# plt.scatter(df['cgpa'], df['iq'], c=df['placement'])
# plt.xlabel("CGPA")
# plt.ylabel("IQ")
# plt.title("CGPA vs IQ (placement)")
# plt.show()

# # ============================================================
# # Extract Input (X) and Output (Y)
# # ============================================================
# X = df.iloc[:, 0:2]    # Features = [cgpa, iq]
# Y = df.iloc[:, -1]     # Target = placement (0 or 1)

# # ============================================================
# # Train-Test Split
# # ============================================================
# from sklearn.model_selection import train_test_split

# X_train, X_test, Y_train, Y_test = train_test_split(
#     X, Y, test_size=0.1, random_state=42
# )

# # ============================================================
# # Feature Scaling
# # ============================================================
# from sklearn.preprocessing import StandardScaler

# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)   # Fit + Transform training data
# X_test = scaler.transform(X_test)         # Only transform test data

# # ============================================================
# # Train Logistic Regression Model
# # ============================================================
# from sklearn.linear_model import LogisticRegression

# clf = LogisticRegression()
# clf.fit(X_train, Y_train)   # Train model

# # Predict on test data
# y_pred = clf.predict(X_test)

# # ============================================================
# # Model Evaluation
# # ============================================================
# from sklearn.metrics import accuracy_score

# acc = accuracy_score(Y_test, y_pred)
# print("Model Accuracy:", acc)

# # Plot decision boundary
# from mlxtend.plotting import plot_decision_regions

# plot_decision_regions(X_train, Y_train.values, clf=clf, legend=2)
# plt.title("Decision Boundary (Training Data)")
# plt.show()

# # ============================================================
# # Save Trained Model (Deployment Step)
# # ============================================================
# import pickle

# pickle.dump(clf, open('model.pkl', 'wb'))  # Save model as pkl file
# print("Model saved as model.pkl")

# # ============================================================
# # Summary of Steps:
# # 1. Data Loading
# # 2. Preprocessing + EDA
# # 3. Feature Extraction (X, Y)
# # 4. Feature Scaling
# # 5. Train-Test Split
# # 6. Model Training (Logistic Regression)
# # 7. Model Evaluation
# # 8. Model Deployment (Pickle)
# # ============================================================
