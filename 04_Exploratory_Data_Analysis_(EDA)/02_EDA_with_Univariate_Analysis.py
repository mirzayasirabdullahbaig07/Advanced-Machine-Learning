# EDA with Univariate Analysis

# Why Univariate Analysis?
# - "Uni" means single, "variate" means variable.
# - When we analyze one variable at a time, it's called univariate analysis.
# - Helps us understand the distribution, central tendency, and spread of data.

# Types of Data:
# 1. Numerical Data -> numbers (e.g., Age, Salary, Fare)
# 2. Categorical Data -> categories (e.g., Country, Gender, Survived)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('train.csv')

# Display first few rows
print(df.head())

# -------------------------------
# CATEGORICAL DATA ANALYSIS
# -------------------------------
# Common plots: Countplot, Bar chart, Pie chart

# Example: Target variable "Survived"
plt.figure(figsize=(6,4))
sns.countplot(x='Survived', data=df)
plt.title("Count of Survived vs Not Survived")
plt.show()

# Bar plot with actual value counts
print("\nValue counts of Survived:")
print(df['Survived'].value_counts())

df['Survived'].value_counts().plot(kind='bar', title="Survival Counts (Bar Plot)")
plt.show()

# Pie chart for Survived
df['Survived'].value_counts().plot(kind='pie', autopct='%.2f%%', title="Survival Distribution")
plt.show()

# Passenger Class distribution
plt.figure(figsize=(6,4))
sns.countplot(x='Pclass', data=df)
plt.title("Passenger Class Distribution")
plt.show()

# -------------------------------
# NUMERICAL DATA ANALYSIS
# -------------------------------
# Common plots: Histogram, KDE/Distplot, Boxplot

# Histogram for Age
plt.figure(figsize=(6,4))
plt.hist(df['Age'].dropna(), bins=50, color='skyblue', edgecolor='black')
plt.title("Age Distribution (Histogram)")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

# Improved histogram: Distribution Plot with KDE
sns.displot(df['Age'].dropna(), kde=True, bins=50)
plt.title("Age Distribution with KDE")
plt.show()

# Box Plot for Fare
plt.figure(figsize=(6,4))
sns.boxplot(x=df['Fare'])
plt.title("Fare Distribution (Boxplot)")
plt.show()

# Statistical summary of numerical columns
print("\nSummary statistics for numerical data:")
print(df.describe())

# Specific descriptive stats for Age
print("\nAge Statistics:")
print("Min Age:", df['Age'].min())
print("Max Age:", df['Age'].max())
print("Median Age:", df['Age'].median())
print("Q1 (25%):", df['Age'].quantile(0.25))
print("Q3 (75%):", df['Age'].quantile(0.75))
print("IQR (Q3 - Q1):", df['Age'].quantile(0.75) - df['Age'].quantile(0.25))

# Thats enoung for the Univarite Analysis