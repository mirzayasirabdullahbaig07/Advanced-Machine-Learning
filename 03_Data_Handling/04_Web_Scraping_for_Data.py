"""
Web Scraping with BeautifulSoup
-------------------------------
Web scraping is used to extract data from websites by parsing HTML content.
It is widely used for data collection, automation, and analysis in web
development and machine learning.
"""

import pandas as pd
import requests
from bs4 import BeautifulSoup

# Define headers (mimic a browser request)
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/115.0.0.0 Safari/537.36"
}

# URL to scrape
url = "YOUR_TARGET_URL"

# Fetch webpage content
response = requests.get(url, headers=headers)

# Check response
if response.status_code == 200:
    soup = BeautifulSoup(response.content, "lxml")
    
    # Print formatted HTML (for inspection)
    print(soup.prettify()[:500])   # print only first 500 chars
    
    # Example: Extract first h1 tag
    first_h1 = soup.find("h1")
    if first_h1:
        print("H1 Tag:", first_h1.text.strip())
    
    # Example: Extract all h2 tags
    print("\n--- H2 Tags ---")
    for tag in soup.find_all("h2"):
        print(tag.text.strip())
    
    # Example: Extract all paragraph tags
    print("\n--- Paragraphs ---")
    for p in soup.find_all("p"):
        print(p.text.strip())
    
    # Extract companies with ratings
    companies = soup.find_all("div", class_="company-content-wrapper")
    print(f"\nFound {len(companies)} companies")
    
    company_names = []
    company_ratings = []
    
    for comp in companies:
        name_tag = comp.find("h2")
        rating_tag = comp.find("p", class_="rating")
        
        if name_tag and rating_tag:
            company_names.append(name_tag.text.strip())
            company_ratings.append(rating_tag.text.strip())
    
    # Save into DataFrame
    df = pd.DataFrame({
        "Company": company_names,
        "Rating": company_ratings
    })
    
    print("\n--- Extracted Data ---")
    print(df.head())
    
    # Save to CSV
    df.to_csv("company_data.csv", index=False)
    print("\nData saved to company_data.csv")

else:
    print("Failed to retrieve webpage. Status code:", response.status_code)
