"""
Topic: Working with SQL and JSON in Python
------------------------------------------
This script demonstrates:
1. What SQL and JSON are.
2. Why JSON is important in Machine Learning.
3. How to load JSON data using Pandas.
4. How to connect to a MySQL database using Python.
5. How to query SQL tables and fetch results with Pandas.
"""

# ===============================
# 1. What is SQL?
# ===============================
# SQL (Structured Query Language) is used to manage and query data 
# stored in relational databases such as MySQL, PostgreSQL, and SQLite.

# ===============================
# 2. What is JSON?
# ===============================
# JSON (JavaScript Object Notation) is a lightweight data format 
# used for storing and exchanging structured data. 
# Example: {"name": "Alice", "age": 25}

# ===============================
# 3. Why is JSON important in ML?
# ===============================
# JSON is important in Machine Learning because:
# - Datasets are often shared in JSON format.
# - APIs return ML-related data (like text, images, predictions) in JSON.
# - JSON is lightweight, human-readable, and easy to parse in Python.

# ================================================================
# Part A: Loading JSON file into Pandas
# ================================================================

import pandas as pd

# Example: Reading a JSON dataset
try:
    df_json = pd.read_json("train.json")
    print(" JSON data loaded successfully")
    print(df_json.head())   # Show first 5 rows
except Exception as e:
    print(" Error loading JSON:", e)

# ================================================================
# Part B: Working with SQL in Python using Pandas
# ================================================================

# Install mysql-connector-python (correct package name)
# Run this once in terminal or notebook
# !pip install mysql-connector-python

import mysql.connector

# Establish MySQL connection
try:
    conn = mysql.connector.connect(
        host="localhost",
        user="root",        # username (default: root)
        password="",        # enter your MySQL password
        database="world"    # sample database "world"
    )
    print(" MySQL connection successful")
except Exception as e:
    print(" Error connecting to MySQL:", e)

# Example Queries
try:
    # 1. Get cities where country code is 'IND'
    query1 = "SELECT * FROM city WHERE CountryCode LIKE 'IND'"
    df_city = pd.read_sql_query(query1, conn)
    print("\nCities in India:")
    print(df_city.head())

    # 2. Get countries where life expectancy > 60
    query2 = "SELECT * FROM Country WHERE LifeExpectancy > 60"
    df_country = pd.read_sql_query(query2, conn)
    print("\nCountries with Life Expectancy > 60:")
    print(df_country.head())

    # 3. Get all country languages
    query3 = "SELECT * FROM CountryLanguage"
    df_lang = pd.read_sql_query(query3, conn)
    print("\nCountry Languages:")
    print(df_lang.head())

except Exception as e:
    print(" SQL Query Error:", e)

# Close connection
if conn.is_connected():
    conn.close()
    print("\n MySQL connection closed")
