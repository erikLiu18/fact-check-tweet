import json
import time
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import os

def save_claims_to_file(claims, query):
    filename = f'fact_claims_{query}.json'  # Use query as part of the filename
    filepath = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data', filename))
    with open(filepath, 'w') as json_file:
        json.dump(claims, json_file, indent=4)
    print(f"Claims saved to {filepath}")

def search_claims(query):
    api_key = 'AIzaSyCbKpZvfiVEDQGrA6tZap5E0e4-yNb4SNg'
    all_claims = []  # List to hold all claims
    page_count = 0  # Counter for the number of pages processed
    page_size = 100  # Number of claims per page
    delay_between_requests = 1  # Delay in seconds between requests
    max_retries = 7  # Maximum number of retries for API requests

    try:
        # Build the service object
        service = build('factchecktools', 'v1alpha1', 
                       developerKey=api_key,
                       discoveryServiceUrl='https://factchecktools.googleapis.com/$discovery/rest?version=v1alpha1')
        
        # Initialize the request
        request = service.claims().search(query=query, languageCode='en', pageSize=page_size)
        
        while request is not None:
            for attempt in range(max_retries):
                try:
                    response = request.execute()
                    claims = response.get('claims', [])
                    all_claims.extend(claims)  # Add claims to the list
                    
                    # Increment the page count and show progress
                    page_count += 1
                    print(f"Processed page {page_count}: {len(claims)} claims found.")
                    
                    # Check for nextPageToken and create a new request if it exists
                    next_page_token = response.get('nextPageToken')
                    if next_page_token:
                        request = service.claims().search(query=query, languageCode='en', pageToken=next_page_token, pageSize=page_size)
                    else:
                        request = None
                    
                    # Delay before the next request
                    time.sleep(delay_between_requests)
                    break  # Exit the retry loop if successful
                except HttpError as e:
                    print(f"An API error occurred: {e}")
                    if attempt < max_retries - 1:  # If not the last attempt
                        wait_time = 2 ** attempt  # Exponential backoff
                        print(f"Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)  # Wait before retrying
                    else:
                        print("Max retries reached. Moving on to the next request.")
                        request = None  # Exit the loop if max retries reached
        
        return all_claims
        
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def main():
    query = input("Enter a claim to fact check: ")
    results = search_claims(query)
    
    if results:
        save_claims_to_file(results, query)  # Pass query to save_claims_to_file

if __name__ == "__main__":
    main() 