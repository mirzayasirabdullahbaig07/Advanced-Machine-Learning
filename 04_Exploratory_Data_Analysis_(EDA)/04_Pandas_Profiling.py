"""
------------------------------------------------------------
Pandas Profiling (ydata-profiling)
------------------------------------------------------------

What is it?
--------------
- Pandas Profiling (now ydata-profiling) is an automatic EDA library.
- It generates a full Exploratory Data Analysis (EDA) report in just one line of code.

Why use it?
--------------
- Saves time → no need to manually write multiple EDA commands.
- Quickly checks data quality, correlations, missing values, duplicates, and distributions.
- Produces an interactive HTML report → easy to share with team.

Advantages:
--------------
1. Data Overview → shape, data types, memory usage.
2. Variable Analysis → histograms, frequency tables, categorical & numerical summaries.
3. Missing Values → counts + heatmaps.
4. Correlation Analysis → Pearson, Spearman, Kendall, Phik correlations with heatmaps.
5. Duplicate Detection → find duplicate rows.
6. Alerts → highlights skewed data, high cardinality, outliers, etc.
7. Interactive Report → collapsible sections in HTML.
8. Great for ML preprocessing → shows issues before model training.

Limitations:
--------------
- Can be slow for very large datasets.
- Report may be too heavy to open with millions of rows.
- Best used as a starting point, not as a replacement for manual EDA.

Alternatives:
--------------
- Sweetviz → auto-EDA with dataset comparison.
- D-Tale → dataframe explorer as web-app.
- AutoViz → automatic visualization tool.
- Lux → Jupyter extension for visualization.

------------------------------------------------------------
"""

# -------- Step 1: Install Library (run only once) --------
# !pip install ydata-profiling

# -------- Step 2: Import Libraries --------
import pandas as pd
from ydata_profiling import ProfileReport

# -------- Step 3: Load Dataset --------
df = pd.read_csv("train.csv")   # Example: Titanic dataset

# -------- Step 4: Generate Profile Report --------
profile = ProfileReport(df, title="Titanic Dataset Report", explorative=True)

# -------- Step 5: Export Report --------
profile.to_file("output.html")    # Save as HTML
# profile.to_notebook_iframe()    # If using Jupyter Notebook

"""
------------------------------------------------------------
How to Use in ML Projects?
------------------------------------------------------------
1. Run profiling → get insights.
2. Handle missing values (from missing value section).
3. Remove duplicates (from duplicate check).
4. Fix skewed data (from distribution analysis).
5. Drop highly correlated features (from correlation heatmap).
6. Then start preprocessing & ML model training.

In short: Pandas Profiling = Instant EDA + Data Quality Report
------------------------------------------------------------
"""
