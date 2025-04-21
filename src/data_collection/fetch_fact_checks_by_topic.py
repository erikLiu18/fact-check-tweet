import argparse
import json
import time
import os
import logging
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Set up logging to a file
log_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../logs/search.log'))
logging.basicConfig(filename=log_file_path, level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

def search_claims(query=None, publisher=None):
    if not query and not publisher:
        raise ValueError("Either 'query' or 'publisher' must be provided.")

    api_key = 'AIzaSyCbKpZvfiVEDQGrA6tZap5E0e4-yNb4SNg'
    all_claims = []  # List to hold all claims
    page_count = 0  # Counter for the number of pages processed
    page_size = 100  # Number of claims per page
    delay_between_requests = 1  # Delay in seconds between requests
    max_retries = 8  # Maximum number of retries for API requests

    try:
        # Build the service object
        service = build('factchecktools', 'v1alpha1', 
                       developerKey=api_key,
                       discoveryServiceUrl='https://factchecktools.googleapis.com/$discovery/rest?version=v1alpha1')
        
        # Initialize the request with the appropriate filters
        request_params = {
            'languageCode': 'en',
            'pageSize': page_size
        }
        
        if query:
            request_params['query'] = query
        if publisher:
            request_params['reviewPublisherSiteFilter'] = publisher
        
        request = service.claims().search(**request_params)
        
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
                        request_params['pageToken'] = next_page_token
                        request = service.claims().search(**request_params)
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
                        # Log the request parameters if all retries fail
                        logging.error(f"Max retries reached. Request parameters: {request_params}")
                        print("Max retries reached. Moving on to the next request.")
                        request = None  # Exit the loop if max retries reached
        
        return all_claims
        
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def save_claims_to_file(claims, identifier):
    current_timestamp = int(time.time())  # Get current timestamp
    filename = f'fact_claims_{identifier}_{current_timestamp}.json'  # Append timestamp to the filename
    filepath = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data', filename))
    with open(filepath, 'w') as json_file:
        json.dump(claims, json_file, indent=4)
    print(f"Claims saved to {filepath}")

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Search for fact-checked claims.')
    parser.add_argument('--query', type=str, help='The claim to fact check.')
    parser.add_argument('--publisher', type=str, help='The publisher to filter claims by.')

    args = parser.parse_args()

    # Call search_claims with the provided arguments
    results = search_claims(query=args.query, publisher=args.publisher)
    
    if results:
        # Use both query and publisher for the filename if both are present
        identifier = f"{args.query}_{args.publisher}" if args.query and args.publisher else args.query or args.publisher
        save_claims_to_file(results, identifier)  # Pass identifier to save_claims_to_file

if __name__ == "__main__":
    main() 