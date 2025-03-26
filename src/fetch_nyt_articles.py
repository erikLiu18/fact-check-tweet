import requests
import json
from pathlib import Path
import argparse
from datetime import datetime
import time
from typing import Dict, Any

def fetch_nyt_articles(year: int, month: int, api_key: str) -> Dict[str, Any]:
    """
    Fetch articles from NYT Archive API for a specific year and month.
    
    Args:
        year: Year to fetch articles for
        month: Month to fetch articles for (1-12)
        api_key: NYT API key
        
    Returns:
        Dictionary containing the API response
    """
    base_url = "https://api.nytimes.com/svc/archive/v1"
    url = f"{base_url}/{year}/{month}.json"
    
    params = {
        'api-key': api_key
    }
    
    print(f"Fetching articles for {year}/{month:02d}...")
    response = requests.get(url, params=params)
    
    if response.status_code != 200:
        raise Exception(f"API request failed with status code {response.status_code}: {response.text}")
    
    return response.json()

def save_articles(data: Dict[str, Any], output_dir: str, year: int, month: int) -> str:
    """
    Save articles to a JSON file.
    
    Args:
        data: Dictionary containing the API response
        output_dir: Directory to save the file
        year: Year of the articles
        month: Month of the articles
        
    Returns:
        Path to the saved file
    """
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate filename with year and month
    output_file = output_dir / f"nyt_articles_{year}_{month:02d}.json"
    
    # Save the data
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Articles saved to {output_file}")
    return str(output_file)

def main():
    parser = argparse.ArgumentParser(description='Fetch NYT articles from Archive API')
    parser.add_argument('--api-key', type=str, required=True,
                        help='NYT API key')
    parser.add_argument('--year', type=int, default=2024,
                        help='Year to fetch articles for')
    parser.add_argument('--month', type=int, default=1,
                        help='Month to fetch articles for (1-12)')
    parser.add_argument('--output-dir', type=str, default='data/raw/nyt',
                        help='Directory to save the articles')
    args = parser.parse_args()
    
    try:
        # Fetch articles
        data = fetch_nyt_articles(args.year, args.month, args.api_key)
        
        # Print summary
        num_articles = len(data.get('response', {}).get('docs', []))
        print(f"\nFound {num_articles} articles")
        
        # Save articles with year and month
        output_file = save_articles(data, args.output_dir, args.year, args.month)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main()) 