# ------------------------------------------------------------
# Exploratory Data Analysis (EDA) - Bivariate & Multivariate
# ------------------------------------------------------------

#  What is Bivariate Analysis?
# - "Bi" means two, "variate" means variables.
# - Analysis of the relationship between TWO variables.
# - Example: How Age affects Survival in Titanic dataset.

#  What is Multivariate Analysis?
# - "Multi" means many.
# - Analysis of the relationship between THREE or MORE variables.
# - Example: How Age, Gender, and Class together affect Survival.

#  Why we use it?
# - To uncover hidden relationships between variables.
# - To answer real-life questions like:
#    Do smokers tip less than non-smokers? (tips dataset)
#    Do women survive more than men on Titanic? (titanic dataset)
#    How do petal/sepal measurements classify iris species? (iris dataset)
#    How air travel grows over time? (flights dataset)

# ------------------------------------------------------------
# Import Libraries & Datasets
# ------------------------------------------------------------
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Sample datasets
tips = sns.load_dataset('tips')
titanic = pd.read_csv('titanic.csv')
flights = sns.load_dataset('flights')
iris = sns.load_dataset('iris')

# Preview data
print(tips.head())
print(titanic.head())
print(flights.head())
print(iris.head())

# ------------------------------------------------------------
# BIVARIATE ANALYSIS
# ------------------------------------------------------------

# 1. Scatter Plot (Numerical vs Numerical)
# Example: Do bigger bills lead to bigger tips? Does gender/smoking matter?
plt.figure(figsize=(6,4))
sns.scatterplot(x='total_bill', y='tip', data=tips, hue='sex', style='smoker', size='size')
plt.title("Total Bill vs Tip (with Gender & Smoker)")
plt.show()

# 2. Bar Plot (Categorical vs Numerical)
# Example: Average Age per Passenger Class on Titanic
plt.figure(figsize=(6,4))
sns.barplot(x='Pclass', y='Age', data=titanic, hue='Sex')
plt.title("Average Age by Passenger Class & Gender")
plt.show()

# 3. Box Plot (Categorical vs Numerical)
# Example: Age distribution by Gender and Survival
plt.figure(figsize=(6,4))
sns.boxplot(x='Sex', y='Age', hue='Survived', data=titanic)
plt.title("Age Distribution by Gender & Survival")
plt.show()

# 4. Distribution Plot (Numerical vs Categorical comparison)
# Example: Age distribution of survivors vs non-survivors
sns.displot(titanic[titanic['Survived']==0]['Age'].dropna(), kind="kde", label="Not Survived")
sns.displot(titanic[titanic['Survived']==1]['Age'].dropna(), kind="kde", label="Survived")
plt.title("Age Distribution by Survival")
plt.legend()
plt.show()

# 5. Heatmap (Categorical vs Categorical)
# Example: Relationship between Pclass & Survival
plt.figure(figsize=(6,4))
sns.heatmap(pd.crosstab(titanic['Pclass'], titanic['Survived']), annot=True, cmap="YlGnBu")
plt.title("Pclass vs Survival (Heatmap)")
plt.show()

# ------------------------------------------------------------
# MULTIVARIATE ANALYSIS
# ------------------------------------------------------------

# 1. Cluster Map (Categorical vs Categorical)
# Example: SibSp (siblings/spouses) vs Survival
sns.clustermap(pd.crosstab(titanic['SibSp'], titanic['Survived']), cmap="coolwarm", annot=True)
plt.title("Clustermap of SibSp vs Survival")
plt.show()

# 2. Pair Plot (Multiple Numerical Variables)
# Example: Iris dataset - how flower measurements classify species
sns.pairplot(iris, hue='species')
plt.show()

# 3. Line Plot (Numerical vs Numerical across time)
# Example: Number of passengers over years (Flights dataset)
new = flights.groupby('year').sum().reset_index()
plt.figure(figsize=(8,4))
sns.lineplot(x='year', y='passengers', data=new, marker="o")
plt.title("Yearly Airline Passengers Growth")
plt.show()

# 4. Heatmap (Multivariate across time & category)
# Example: Passengers by Month & Year
plt.figure(figsize=(10,6))
sns.heatmap(flights.pivot_table(values='passengers', index='month', columns='year'), cmap="YlOrBr")
plt.title("Passengers Heatmap (Month vs Year)")
plt.show()

# 5. Cluster Map (Multivariate pattern detection)
sns.clustermap(flights.pivot_table(values='passengers', index='month', columns='year'), cmap="coolwarm")
plt.title("Clustered Passenger Trends")
plt.show()
