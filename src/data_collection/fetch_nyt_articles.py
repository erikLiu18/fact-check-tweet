import requests
import json
from pathlib import Path
import argparse
from datetime import datetime
import time
from typing import Dict, Any

# Rate limit: 5 requests per minute
RATE_LIMIT = 5
RATE_WINDOW = 60  # seconds

# Global variables for rate limiting
request_count = 0
last_reset_time = time.time()

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
    global request_count, last_reset_time

    # If we've reached the rate limit, wait until the minute is up
    if request_count >= RATE_LIMIT:
        wait_time = RATE_WINDOW - (time.time() - last_reset_time)
        if wait_time > 0:
            print(f"Rate limit reached. Waiting {wait_time:.2f} seconds...")
            time.sleep(wait_time)
            # Reset after waiting
            request_count = 0
            last_reset_time = time.time()
    
    base_url = "https://api.nytimes.com/svc/archive/v1"
    url = f"{base_url}/{year}/{month}.json"
    
    params = {
        'api-key': api_key
    }
    
    print(f"Fetching articles for {year}/{month:02d}...")
    response = requests.get(url, params=params)
    
    # Increment the request count
    request_count += 1
    
    if response.status_code != 200:
        raise Exception(f"API request failed with status code {response.status_code}: {response.text}")
    
    return response.json()

def save_articles(data: Dict[str, Any], output_dir: str, year: int) -> str:
    """
    Save filtered articles to a JSON file.
    
    Args:
        data: Dictionary containing the API response
        output_dir: Directory to save the file
        year: Year of the articles
        
    Returns:
        Path to the saved file
    """
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate filename with year
    output_file = output_dir / f"nyt_articles_{year}.json"
    
    # Filter articles with type_of_material as "News" and extract only abstract and pub_date
    filtered_articles = []
    for article in data.get('response', {}).get('docs', []):
        if article.get('type_of_material') == 'News':
            filtered_article = {
                'abstract': article.get('abstract', ''),
                'pub_date': article.get('pub_date', '')
            }
            filtered_articles.append(filtered_article)
    
    # Save the filtered data
    with open(output_file, 'w') as f:
        json.dump(filtered_articles, f, indent=2)
    
    print(f"Filtered articles saved to {output_file}")
    return str(output_file)

def main():
    parser = argparse.ArgumentParser(description='Fetch NYT articles from Archive API')
    parser.add_argument('--api-key', type=str, required=True,
                        help='NYT API key')
    parser.add_argument('--year', type=int, required=True,
                        help='Year to fetch articles for')
    parser.add_argument('--output-dir', type=str, default='data/raw/nyt',
                        help='Directory to save the articles')
    args = parser.parse_args()
    
    # Initialize a list to store all filtered articles
    all_filtered_articles = []
    
    # Get current date to handle future months in the current year
    current_date = datetime.now()
    max_month = 12
    
    # If the requested year is the current year, only process months up to the current month
    if args.year == current_date.year:
        max_month = current_date.month
    
    # Fetch articles for all valid months in the year
    for month in range(1, max_month + 1):
        print(f"\nProcessing {args.year}/{month:02d}...")
        try:
            data = fetch_nyt_articles(args.year, month, args.api_key)
            
            # Count total articles and News articles
            all_articles = data.get('response', {}).get('docs', [])
            news_articles = [article for article in all_articles if article.get('type_of_material') == 'News']
            
            print(f"Found {len(all_articles)} total articles")
            print(f"Of which {len(news_articles)} are News articles")
            
            # Filter articles with type_of_material as "News" and extract only abstract and pub_date
            for article in news_articles:
                filtered_article = {
                    'abstract': article.get('abstract', ''),
                    'pub_date': article.get('pub_date', '')
                }
                all_filtered_articles.append(filtered_article)
        except Exception as e:
            print(f"Error fetching articles for {args.year}/{month:02d}: {str(e)}")
            print("Continuing with the next month...")
    
    # Save all filtered articles to a single file
    try:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"nyt_articles_{args.year}.json"
        
        with open(output_file, 'w') as f:
            json.dump(all_filtered_articles, f, indent=2)
        
        print(f"\nAll filtered articles for {args.year} saved to {output_file}")
        print(f"Total News articles saved: {len(all_filtered_articles)}")
    except Exception as e:
        print(f"Error saving articles: {str(e)}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main()) 