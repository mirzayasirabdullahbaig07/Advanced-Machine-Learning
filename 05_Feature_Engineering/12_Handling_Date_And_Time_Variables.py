
# Handling Date & Time Variables

# Definition:
# - Date and time variables contain rich hidden information like:
#   * Year, Month, Day
#   * Day of Week (Mon–Sun)
#   * Quarter, Season
#   * Hour, Minute, Second
# - These features are often very useful for:
#   * Time-series analysis
#   * Customer behavior analysis
#   * Trend detection
#   * Feature engineering for ML models

# Example Datasets:
# - orders.csv   -> Contains date information for customer orders
# - message.csv  -> Contains timestamp of messages


import numpy as np
import pandas as pd
import datetime

# -------------------------------
# Load example datasets
# -------------------------------
date = pd.read_csv('orders.csv')     # dataset with order dates
time = pd.read_csv('message.csv')    # dataset with message timestamps

print(date.head())
print(time.head())

date.info()
time.info()

# -------------------------------
# Working with Dates
# -------------------------------
# Convert to datetime format
date['date'] = pd.to_datetime(date['date'])

# Extract today’s datetime
today = datetime.datetime.today()
print("Today:", today)

# Time elapsed between dates
date['days_elapsed'] = (today - date['date']).dt.days
print(date[['date', 'days_elapsed']].head())

# Months passed since each date
date['months_elapsed'] = np.round(
    (today - date['date']) / np.timedelta64(1, 'M'),
    0
)
print(date[['date', 'months_elapsed']].head())

# -------------------------------
# Working with Time (hour, min, sec)
# -------------------------------
time['date'] = pd.to_datetime(time['date'])
time.info()

# Extract components
time['hour'] = time['date'].dt.hour
time['minute'] = time['date'].dt.minute
time['second'] = time['date'].dt.second
time['time_only'] = time['date'].dt.time

print(time.head())

# -------------------------------
# Time differences
# -------------------------------
# Difference between now and each message timestamp
time['seconds_elapsed'] = (today - time['date']) / np.timedelta64(1, 's')
time['minutes_elapsed'] = (today - time['date']) / np.timedelta64(1, 'm')
time['hours_elapsed']   = (today - time['date']) / np.timedelta64(1, 'h')

print(time[['date', 'seconds_elapsed', 'minutes_elapsed', 'hours_elapsed']].head())
