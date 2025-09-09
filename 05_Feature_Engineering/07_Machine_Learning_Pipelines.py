# ============================================================
# 🚀 Machine Learning Pipelines A–Z (Titanic Example)
# ============================================================
# A pipeline chains multiple preprocessing + modeling steps
# so the same transformations are consistently applied on
# training, testing, and real-world user input data.
# ============================================================

# -----------------------------
# 1. Import Required Libraries
# -----------------------------
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import set_config
import pickle

# -----------------------------
# 2. Load and Prepare Dataset
# -----------------------------
df = pd.read_csv("train.csv")

# Drop columns that are not useful for prediction
df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], inplace=True)

# -----------------------------
# 3. Train/Test Split
# -----------------------------
X = df.drop(columns=["Survived"])
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 4. Define Transformations
# -----------------------------

# Step 4.1 → Imputation (fill missing values)
# - Age → fill with mean
# - Embarked → fill with most frequent value
trf1 = ColumnTransformer([
    ("impute_age", SimpleImputer(), [2]),  # Age column index
    ("impute_embarked", SimpleImputer(strategy="most_frequent"), [6])  # Embarked column index
], remainder="passthrough")

# Step 4.2 → Encoding (categorical → numerical)
# - Encode Sex and Embarked columns with OneHotEncoder
trf2 = ColumnTransformer([
    ("ohe_sex_embarked", OneHotEncoder(sparse=False, handle_unknown="ignore"), [1, 6])
], remainder="passthrough")

# Step 4.3 → Scaling (normalize numerical features)
# - Scale first 8 columns into range [0,1]
trf3 = ColumnTransformer([
    ("scale", MinMaxScaler(), slice(0, 8))
])

# Step 4.4 → Feature Selection
# - Select top 5 features using Chi-square test
trf4 = SelectKBest(score_func=chi2, k=5)

# Step 4.5 → Model
# - Use Decision Tree Classifier as the ML model
trf5 = DecisionTreeClassifier(random_state=42)

# -----------------------------
# 5. Build Pipeline
# -----------------------------

# Option 1: Explicitly name each step
pipe = Pipeline([
    ("trf1", trf1),
    ("trf2", trf2),
    ("trf3", trf3),
    ("trf4", trf4),
    ("trf5", trf5)
])

# Option 2: Cleaner syntax with make_pipeline (no naming needed)
# pipe = make_pipeline(trf1, trf2, trf3, trf4, trf5)

# -----------------------------
# 6. Train Pipeline
# -----------------------------
pipe.fit(X_train, y_train)

# -----------------------------
# 7. Evaluate Pipeline
# -----------------------------
y_pred = pipe.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred))

# Display pipeline visually
set_config(display="diagram")
pipe

# -----------------------------
# 8. Cross Validation
# -----------------------------
cv_score = cross_val_score(pipe, X_train, y_train, cv=5, scoring="accuracy").mean()
print("Cross Validation Score:", cv_score)

# -----------------------------
# 9. Hyperparameter Tuning with GridSearchCV
# -----------------------------
params = {
    "trf5__max_depth": [1, 2, 3, 4, 5, None]  # Tune max_depth of Decision Tree
}

grid = GridSearchCV(pipe, params, cv=5, scoring="accuracy")
grid.fit(X_train, y_train)

print("Best CV Score:", grid.best_score_)
print("Best Params:", grid.best_params_)

# -----------------------------
# 10. Save Final Pipeline
# -----------------------------
# Save best pipeline (ready for production)
pickle.dump(grid.best_estimator_, open("titanic_pipeline.pkl", "wb"))

# -----------------------------
# 11. Load Pipeline & Test on New Data
# -----------------------------
loaded_pipe = pickle.load(open("titanic_pipeline.pkl", "rb"))

# Example passenger input:
# Pclass, Sex, Age, SibSp, Parch, Fare, Embarked
test_input = np.array([2, "male", 31.0, 0, 0, 10.5, "S"], dtype=object).reshape(1, 7)

print("New Prediction:", loaded_pipe.predict(test_input))
