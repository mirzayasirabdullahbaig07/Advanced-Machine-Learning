# ============================================================
# DATA GATHERING IN MACHINE LEARNING
# ============================================================
# Before building models, we first need to GATHER data.
# Data gathering means collecting data from different sources.
# Common formats/sources of data in ML:
# 
# 1. CSV  -> Comma Separated Values
# 2. TSV  -> Tab Separated Values
# 3. JSON -> JavaScript Object Notation
# 4. APIs -> Fetching data from online APIs
# 5. Web Scraping -> Extracting data from websites
# ============================================================


# ============================================================
# CSV (Comma Separated Values)
# ============================================================
# A plain text file where values are separated by commas.
# Example:
# name, age, country
# John, 25, USA
# Mary, 30, UK
# ============================================================

# Opening a CSV file
import pandas as pd
df = pd.read_csv("aug_train.csv")   # Reads local CSV file
print(df.head())


# ============================================================
# Opening a CSV file from URL
# ============================================================
import requests
from io import StringIO

url = "https://raw.githubusercontent.com/cs109/2014_data/master/countries.csv"
headers = {"User-Agent": "Mozilla/5.0"}

req = requests.get(url, headers=headers)
data = StringIO(req.text)
df = pd.read_csv(data)
print(df.head())


# ============================================================
# TSV (Tab Separated Values)
# ============================================================
# Similar to CSV but values are separated by TAB (\t)
# Example:
# name    age   country
# John    25    USA
# Mary    30    UK
# ============================================================

pd.read_csv('movie_titles_metadata.tsv', sep='\t',
            names=['sno', 'name', 'release_year', 'rating', 'votes', 'genres'])


# ============================================================
# Important Parameters in pd.read_csv()
# ============================================================

# 1. index_col -> set a column as index
pd.read_csv("aug_train.csv", index_col="enrollee_id")

# 2. header -> specify which row to use as column headers
pd.read_csv("aug_train.csv", header=1)  # 2nd row as header

# 3. usecols -> select only certain columns
pd.read_csv("aug_train.csv", usecols=['enrollee_id', 'gender'])

# 4. squeeze (deprecated in new pandas) -> returns Series if single column
pd.read_csv("aug_train.csv", usecols=['gender']).squeeze()

# 5. skiprows -> skip certain rows while reading
pd.read_csv("aug_train.csv", skiprows=10)   # skip first 10 rows

# 6. nrows -> read only limited number of rows
pd.read_csv("aug_train.csv", nrows=100)   # first 100 rows only

# 7. encoding -> handle encoding issues (like utf-8, latin1, etc.)
pd.read_csv("aug_train.csv", encoding='utf-8')

# 8. on_bad_lines -> skip bad rows (with wrong columns)
pd.read_csv("aug_train.csv", on_bad_lines='skip')

# 9. dtype -> set datatypes manually
pd.read_csv("aug_train.csv", dtype={'gender': 'category'})

# 10. parse_dates -> automatically parse dates
pd.read_csv("date_data.csv", parse_dates=['joining_date'])

# 11. converters -> apply custom function to a column while loading
pd.read_csv("aug_train.csv", converters={'gender': lambda x: x.upper()})

# 12. na_values -> define extra missing values
pd.read_csv("aug_train.csv", na_values=['NA', 'missing'])

# 13. chunksize -> load huge datasets in smaller parts
for chunk in pd.read_csv("aug_train.csv", chunksize=500):
    print("Chunk size:", chunk.shape)
    # process each chunk separately (useful for big data)

# ============================================================
# SUMMARY:
# - CSV: Comma separated
# - TSV: Tab separated
# - JSON/API/Web scraping: Other sources
# - pd.read_csv() has powerful options for real-world messy data
# ============================================================
