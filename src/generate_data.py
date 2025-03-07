import pandas as pd
import json
import os
import argparse
from tqdm import tqdm
from llm import QwenModel

def load_preprocessed_data(file_path, subset_size=None):
    """
    Load preprocessed data from a JSON file.
    
    Args:
        file_path (str): Path to the preprocessed JSON file
        subset_size (int, optional): Number of rows to load (for testing)
        
    Returns:
        list: List of dictionaries containing the data
    """
    # Read the JSON file line by line
    data = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if subset_size is not None and i >= subset_size:
                break
            data.append(json.loads(line))
    
    print(f"Loaded {len(data)} rows from {file_path}")
    
    # Verify that required columns exist
    required_columns = ['text', 'date', 'mergedTextualRating']
    for column in required_columns:
        if column not in data[0]:
            print(f"Warning: Required column '{column}' not found in the data")
    
    return data

def generate_tweets(model, fact_checks, tweets_per_check=10, batch_size=4, max_tokens=100):
    """
    Generate tweets for each fact check using the LLM.
    
    Args:
        model: The LLM model to use
        fact_checks (list): List of fact check data
        tweets_per_check (int): Number of tweets to generate per fact check
        batch_size (int): Batch size for processing
        max_tokens (int): Maximum number of tokens to generate per tweet
        
    Returns:
        list: List of dictionaries with original fact check and generated tweets
    """
    system_prompt = """You are a helpful assistant that generates short, engaging tweets (maximum 280 characters) 
    based on given claims. Your task is to create tweets that spread the information in the claim. 
    Make the tweets sound natural, as if written by a regular Twitter user. 
    Do not include hashtags, @mentions, or URLs unless they are part of the original claim.
    Only respond with the tweet, no other text."""
    
    # Extract text from fact checks
    texts = [check.get('text', '') for check in fact_checks]
    user_prompts = [f"Generate a short tweet (max 280 chars) spreading the information in this claim: {text}" for text in texts]
    
    # Generate tweets using batch processing
    print(f"Generating {tweets_per_check} tweets for each of the {len(fact_checks)} fact checks...")
    batch_responses = model.batch_generate_responses(
        user_prompts=user_prompts,
        system_prompt=system_prompt,
        responses_per_prompt=tweets_per_check,
        temperature=0.8,
        batch_size=batch_size,
        max_new_tokens=max_tokens
    )
    
    # Combine original fact checks with generated tweets
    results = []
    for i, fact_check in enumerate(fact_checks):
        results.append({
            'original_fact_check': fact_check,
            'generated_tweets': batch_responses[i]
        })
    
    return results

def save_results(results, output_file):
    """
    Save the generated tweets to a JSON file.
    Each row will contain the original claim text, date, rating, and a generated tweet.
    
    Args:
        results (list): List of dictionaries with original fact check and generated tweets
        output_file (str): Path to save the results
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Restructure the data for output
    structured_results = []
    for result in results:
        fact_check = result['original_fact_check']
        tweets = result['generated_tweets']
        
        # Extract the relevant fields
        for tweet in tweets:
            structured_results.append({
                'text': fact_check.get('text', ''),
                'date': fact_check.get('date', ''),
                'mergedTextualRating': fact_check.get('mergedTextualRating', ''),
                'generated_tweet': tweet
            })
    
    # Save to JSON file - each line is a separate JSON object
    with open(output_file, 'w') as f:
        for item in structured_results:
            f.write(json.dumps(item) + '\n')
    
    print(f"Results saved to {output_file}")
    print(f"Total generated tweets: {len(structured_results)}")

def main():
    parser = argparse.ArgumentParser(description='Generate tweets from fact checks using an LLM')
    parser.add_argument('--input', type=str, default='data/processed/processed_fact_claims_1739933287.json',
                        help='Path to the preprocessed JSON file')
    parser.add_argument('--output', type=str, default='data/generated/generated_tweets.jsonl',
                        help='Path to save the generated tweets (JSONL format)')
    parser.add_argument('--subset', type=int, default=10,
                        help='Number of fact checks to process (for testing)')
    parser.add_argument('--tweets-per-check', type=int, default=10,
                        help='Number of tweets to generate per fact check')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size for processing')
    parser.add_argument('--max-tokens', type=int, default=100,
                        help='Maximum number of tokens to generate per tweet')
    args = parser.parse_args()
    
    # Load the preprocessed data
    fact_checks = load_preprocessed_data(args.input, subset_size=args.subset)
    
    # Initialize the LLM model
    model = QwenModel()
    
    # Generate tweets
    results = generate_tweets(
        model=model,
        fact_checks=fact_checks,
        tweets_per_check=args.tweets_per_check,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens
    )
    
    # Save the results
    save_results(results, args.output)

if __name__ == "__main__":
    main()
