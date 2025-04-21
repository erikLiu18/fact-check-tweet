import os
import time
import json
import datetime
import re
from dateutil import parser
import pandas as pd
from fetch_fact_checks_by_topic import search_claims

def find_latest_json_file():
    """Find the most recent fact_claims JSON file in the data/raw directory."""
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/raw"))
    json_files = [f for f in os.listdir(data_dir) if f.startswith('fact_claims_') and f.endswith('.json')]
    
    if not json_files:
        return None
    
    # Function to safely extract timestamp from filenames
    def extract_timestamp(filename):
        # Try to extract timestamp using regex pattern
        match = re.search(r'fact_claims_(\d+)\.json', filename)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                # If conversion fails, use file modification time as fallback
                file_path = os.path.join(data_dir, filename)
                return int(os.path.getmtime(file_path))
        else:
            # If no timestamp found in filename, use file modification time
            file_path = os.path.join(data_dir, filename)
            return int(os.path.getmtime(file_path))
    
    # Extract timestamps and find the latest
    latest_file = max(json_files, key=extract_timestamp)
    return os.path.join(data_dir, latest_file)

def get_days_since_last_run():
    """Calculate the number of days since the last run based on the timestamp in the filename."""
    latest_file = find_latest_json_file()
    
    if not latest_file:
        # If no previous file found, return None (will fetch all facts)
        return None
    
    # Get the file's modification time as a fallback
    file_mtime = int(os.path.getmtime(latest_file))
    
    # Try to extract timestamp from filename
    match = re.search(r'fact_claims_(\d+)\.json', os.path.basename(latest_file))
    timestamp = int(match.group(1)) if match else file_mtime
    
    # Calculate days since the timestamp
    now = int(time.time())
    days_diff = (now - timestamp) // (24 * 60 * 60)
    
    # Ensure we have at least 1 day to fetch new data
    return max(1, days_diff)

def load_existing_claims():
    """Load claims from the most recent JSON file."""
    latest_file = find_latest_json_file()
    
    if not latest_file:
        return []
    
    with open(latest_file, 'r') as file:
        return json.load(file)

def scrape_publishers(publishers, max_age_days=None):
    all_claims = []  # List to hold claims from all publishers

    for publisher in publishers:
        print(f"Searching claims for publisher: {publisher}")
        claims = search_claims(publisher=publisher, maxAgeDays=max_age_days)
        if claims:
            all_claims.extend(claims)  # Add claims to the list

    return all_claims

def merge_claims(existing_claims, new_claims):
    """Merge existing and new claims, removing duplicates based on the review URL."""
    # Create a set of existing claim URLs to avoid duplicates
    existing_urls = set()
    for claim in existing_claims:
        for review in claim.get('claimReview', []):
            if 'url' in review:
                existing_urls.add(review['url'])
    
    # Add only new, unique claims
    unique_new_claims = []
    for claim in new_claims:
        is_new = True
        for review in claim.get('claimReview', []):
            if 'url' in review and review['url'] in existing_urls:
                is_new = False
                break
        if is_new:
            unique_new_claims.append(claim)
    
    # Combine existing and unique new claims
    merged_claims = existing_claims + unique_new_claims
    
    return merged_claims

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
    ]

    # Load existing claims
    existing_claims = load_existing_claims()
    print(f"Loaded {len(existing_claims)} existing claims.")
    
    # Get the days since the last run based on file timestamp
    days_since_last_run = get_days_since_last_run()
    
    if days_since_last_run:
        print(f"Last run was {days_since_last_run} days ago. Fetching claims from the last {days_since_last_run} days.")
        new_claims = scrape_publishers(publishers, max_age_days=days_since_last_run)
    else:
        print("No previous run detected. Fetching all available claims.")
        new_claims = scrape_publishers(publishers)

    if new_claims:
        print(f"Found {len(new_claims)} new claims.")
        
        # Merge existing and new claims
        merged_claims = merge_claims(existing_claims, new_claims)
        print(f"Total claims after merging: {len(merged_claims)}")
        
        # Save the merged claims
        current_timestamp = int(time.time())  # Get current timestamp
        output_file = f'{os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/raw"))}/fact_claims_{current_timestamp}.json'
        with open(output_file, 'w') as json_file:
            json.dump(merged_claims, json_file, indent=4)
        print(f"Combined claims saved to {output_file}")
    else:
        print("No new claims found for the specified publishers.")

if __name__ == "__main__":
    main()
