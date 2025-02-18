from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

def search_claims(query):
    api_key = 'AIzaSyCbKpZvfiVEDQGrA6tZap5E0e4-yNb4SNg'
    
    try:
        # Build the service object
        service = build('factchecktools', 'v1alpha1', 
                       developerKey=api_key,
                       discoveryServiceUrl='https://factchecktools.googleapis.com/$discovery/rest?version=v1alpha1')
        
        # Make the API request
        request = service.claims().search(query=query, languageCode='en')
        response = request.execute()
        return response
        
    except HttpError as e:
        print(f"An API error occurred: {e}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def print_claim_results(results):
    if not results:
        print("No results found or error occurred")
        return
        
    claims = results.get('claims', [])
    if not claims:
        print("No claims found for this query")
        return
        
    for i, claim in enumerate(claims, 1):
        print(f"\nClaim {i}:")
        print(f"Text: {claim.get('text', 'N/A')}")
        print(f"Claimant: {claim.get('claimant', 'N/A')}")
        
        # Print claim review if available
        claim_review = claim.get('claimReview', [])
        if claim_review:
            review = claim_review[0]  # Get the first review
            print(f"Publisher: {review.get('publisher', {}).get('name', 'N/A')}")
            print(f"Rating: {review.get('textualRating', 'N/A')}")
            print(f"Review URL: {review.get('url', 'N/A')}")

def main():
    query = input("Enter a claim to fact check: ")
    results = search_claims(query)
    print_claim_results(results)

if __name__ == "__main__":
    main() 