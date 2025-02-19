import time
import json
import pandas as pd
from search import search_claims

def scrape_publishers(publishers):
    all_claims = []  # List to hold claims from all publishers

    for publisher in publishers:
        print(f"Searching claims for publisher: {publisher}")
        claims = search_claims(publisher=publisher)
        if claims:
            all_claims.extend(claims)  # Add claims to the list

    return all_claims

def main():
    # List of publishers to search
    publishers = [
        "politifact.com",
        "washingtonpost.com",
        "nytimes.com",
        "apnews.com",
        "usatoday.com",
        "cbsnews.com",
        "fullfact.org",
        "science.feedback.org",
        "snopes.com",
        "factcheck.org",
        "factcheck.afp.com",
        "cnn.com",
    ]

    # Scrape claims for the list of publishers
    claims = scrape_publishers(publishers)

    if claims:
        current_timestamp = int(time.time())  # Get current timestamp
        output_file = f'fact_claims_{current_timestamp}.json'  # Append timestamp to the filename
        with open(output_file, 'w') as json_file:
            json.dump(claims, json_file, indent=4)
        print(f"Combined claims saved to {output_file}")
    else:
        print("No claims found for the specified publishers.")

if __name__ == "__main__":

    main()
