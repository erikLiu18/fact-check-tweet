import argparse
from search import search_claims, save_claims_to_file

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