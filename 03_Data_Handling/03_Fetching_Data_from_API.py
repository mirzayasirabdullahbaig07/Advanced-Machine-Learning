"""
What is API?
------------
API (Application Programming Interface) is a way for two software systems 
to communicate with each other. You can think of it as a data pipeline 
that sends and receives structured information.
"""

import pandas as pd
import requests

# Example: calling an API
url = "API_URL_HERE"

# Make the request
response = requests.get(url)

# Check if the response is OK (status code 200)
if response.status_code == 200:
    data = response.json()  # Convert JSON response to Python dict
    # Extract 'results' and load into DataFrame
    df_sample = pd.DataFrame(data.get('results', []))[['id', 'title', 'overview', 'release_date']].head(2)
    print(df_sample)
else:
    print("Error:", response.status_code)


# Collecting data across multiple pages
all_data = []

for page in range(1, 430):   # Assuming 429 pages
    response = requests.get(f"{url}?page={page}")
    
    if response.status_code == 200:
        data = response.json().get('results', [])
        if data:  # Only process if results exist
            temp_df = pd.DataFrame(data)[['id', 'title', 'overview', 'release_date']]
            all_data.append(temp_df)
    else:
        print(f"Failed at page {page}, Status:", response.status_code)

# Combine all pages into a single DataFrame
df = pd.concat(all_data, ignore_index=True)

# Show shape of final dataset
print("Final shape:", df.shape)

# Save to CSV
df.to_csv("movies_data.csv", index=False)
print("Data saved to movies_data.csv")
