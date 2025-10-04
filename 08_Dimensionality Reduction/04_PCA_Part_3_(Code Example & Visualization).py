# PCA Part 3: Wine Classification Example (With & Without PCA)
# -------------------------------------------------------------
# This example demonstrates how to apply PCA for dimensionality reduction
# and compares KNN classifier performance with and without PCA.

# -----------------------
# Step 1: Data Loading
# -----------------------
import pandas as pd

wine_data_path = "v"  # Replace with your dataset path
wine_data = pd.read_csv(wine_data_path)
print("First row of dataset:\n", wine_data.head(1))

# -----------------------
# Step 2: Data Exploration
# -----------------------
# Check dataset info
wine_data.info()

# Check for missing values
print("Missing values per column:\n", wine_data.isna().sum())

# Check for duplicate rows
print("Number of duplicate rows:", wine_data.duplicated().sum())

# -----------------------
# Step 3: Data Cleaning
# -----------------------
# Drop rows with missing values
wine_data.dropna(inplace=True)

# Drop duplicate rows
wine_data.drop_duplicates(inplace=True)

print("Shape of dataset after cleaning:", wine_data.shape)

# -----------------------
# Step 4: Feature & Target Split
# -----------------------
# Features: physicochemical properties
# Target: wine type
X = wine_data.drop('type', axis=1)
y = wine_data['type']

# -----------------------
# Step 5: Train-Test Split
# -----------------------
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------
# Step 6: Feature Scaling
# -----------------------
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------
# Step 7: KNN Classification WITHOUT PCA
# -----------------------
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Train KNN classifier
knn_no_pca = KNeighborsClassifier(n_neighbors=5)
knn_no_pca.fit(X_train_scaled, y_train)

# Predict and evaluate
y_pred_no_pca = knn_no_pca.predict(X_test_scaled)
accuracy_no_pca = accuracy_score(y_test, y_pred_no_pca)

print(f"KNN (without PCA) - Number of features: {X_train_scaled.shape[1]}")
print(f"Classification Accuracy without PCA: {accuracy_no_pca * 100:.2f}%")

# -----------------------
# Step 8: Apply PCA
# -----------------------
from sklearn.decomposition import PCA

# Reduce to 5 principal components
pca = PCA(n_components=5)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# -----------------------
# Step 9: KNN Classification WITH PCA
# -----------------------
knn_with_pca = KNeighborsClassifier(n_neighbors=5)
knn_with_pca.fit(X_train_pca, y_train)

# Predict and evaluate
y_pred_pca = knn_with_pca.predict(X_test_pca)
accuracy_pca = accuracy_score(y_test, y_pred_pca)

print(f"KNN (with PCA) - Number of features: {X_train_pca.shape[1]}")
print(f"Classification Accuracy with PCA: {accuracy_pca * 100:.2f}%")

# -----------------------
# Optional: PCA Explained Variance
# -----------------------
print("Explained variance ratio by each principal component:", pca.explained_variance_ratio_)
print("Total variance explained by 5 components:", sum(pca.explained_variance_ratio_))
