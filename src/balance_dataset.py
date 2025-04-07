import json
import pandas as pd
import os
import argparse
import random
from sklearn.model_selection import train_test_split
from datetime import datetime

def load_fact_claims(file_path, year):
    """
    Load fact claims from the processed JSON file and filter for a specific year.
    
    Args:
        file_path (str): Path to the processed fact claims JSON file
        year (int): Year to filter the data for
        
    Returns:
        pd.DataFrame: DataFrame containing the filtered fact claims
    """
    df_filtered = pd.read_json(file_path, orient='records', lines=True)
    
    # Convert date column to datetime
    df_filtered['date'] = pd.to_datetime(df_filtered['date'])
    
    # Filter for the specified year
    df_filtered = df_filtered[df_filtered['date'].dt.year == year].copy()
    
    # Select only the required columns
    df_filtered = df_filtered[['text', 'mergedTextualRating']].copy()
    
    # Rename the label column
    # df_filtered = df_filtered.rename(columns={'mergedTextualRating': 'label'})
    
    return df_filtered

def load_nyt_articles(file_path):
    """
    Load NYT articles from the JSON file.
    
    Args:
        file_path (str): Path to the NYT articles JSON file
        
    Returns:
        pd.DataFrame: DataFrame containing the NYT articles
    """
    # Read the JSON file
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Filter out articles with empty abstract fields
    original_count = len(df)
    df = df.dropna(subset=['abstract'])
    df = df[df['abstract'].str.strip() != '']
    filtered_count = len(df)
    
    if filtered_count < original_count:
        print(f"Filtered out {original_count - filtered_count} articles with empty abstract fields")
    
    # Add a label column with 'true' for all articles
    df['label'] = 'true'
    
    # Select only the required columns
    df = df[['abstract', 'label']].copy()
    
    # Rename the text column
    df = df.rename(columns={'abstract': 'text'})
    
    # Rename the label column
    df = df.rename(columns={'label': 'mergedTextualRating'})
    
    return df

def balance_dataset(fact_claims_df, nyt_articles_df):
    """
    Balance the dataset by adding NYT articles to the fact claims.
    
    Args:
        fact_claims_df (pd.DataFrame): DataFrame containing the fact claims
        nyt_articles_df (pd.DataFrame): DataFrame containing the NYT articles
        
    Returns:
        pd.DataFrame: Balanced DataFrame
    """
    # Count the number of false and true labels in fact claims
    false_count = len(fact_claims_df[fact_claims_df['mergedTextualRating'] == 'false'])
    true_count = len(fact_claims_df[fact_claims_df['mergedTextualRating'] == 'true'])
    
    # Calculate how many true samples we need to add
    samples_to_add = false_count - true_count
    
    if samples_to_add <= 0:
        print("Dataset is already balanced or has more true than false samples.")
        return fact_claims_df
    
    # Randomly sample from NYT articles
    if samples_to_add > len(nyt_articles_df):
        print(f"Warning: Not enough NYT articles to balance the dataset. Using all available articles.")
        samples_to_add = len(nyt_articles_df)
    
    sampled_nyt = nyt_articles_df.sample(n=samples_to_add, random_state=42)
    
    # Combine the fact claims with the sampled NYT articles
    balanced_df = pd.concat([fact_claims_df, sampled_nyt], ignore_index=True)
    
    # Remove both regular and tilted double quotes at the beginning or end of strings in the text column
    balanced_df['text'] = balanced_df['text'].apply(lambda x: x.strip('"\'“”') if isinstance(x, str) else x)
    
    # Shuffle the combined dataset
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return balanced_df

def split_dataset(df, year, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42):
    """
    Split the dataset into training, validation, and test sets.
    
    Args:
        df (pd.DataFrame): The full dataset
        year (int): Year of the data
        train_size (float): Proportion of data for training
        val_size (float): Proportion of data for validation
        test_size (float): Proportion of data for testing
        random_state (int): Random seed for reproducibility
    """
    # First split: separate training set
    train_df, temp_df = train_test_split(
        df, 
        train_size=train_size, 
        random_state=random_state,
        stratify=df['mergedTextualRating']
    )
    
    # Second split: separate validation and test sets
    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_size/(val_size + test_size),
        random_state=random_state,
        stratify=temp_df['mergedTextualRating']
    )
    
    # Create the output directory if it doesn't exist
    os.makedirs('data/processed/balanced', exist_ok=True)
    
    # Save the splits with year in the filename
    train_df.to_csv(f'data/processed/balanced/train_set_{year}.csv', index=False)
    val_df.to_csv(f'data/processed/balanced/val_set_{year}.csv', index=False)
    test_df.to_csv(f'data/processed/balanced/test_set_{year}.csv', index=False)
    
    print(f'Dataset split into:')
    print(f'  Training set: {len(train_df)} rows')
    print(f'  Validation set: {len(val_df)} rows')
    print(f'  Test set: {len(test_df)} rows')
    
    # Print class distribution for each split
    for name, split_df in [('Training', train_df), ('Validation', val_df), ('Test', test_df)]:
        dist = split_df['mergedTextualRating'].value_counts(normalize=True)
        print(f'\n{name} set class distribution:')
        for label, prop in dist.items():
            print(f'  {label}: {prop:.2%}')

def main():
    parser = argparse.ArgumentParser(description='Balance the dataset by combining fact claims with NYT articles.')
    parser.add_argument('--year', type=int, required=True,
                        help='Year to filter the data for')
    args = parser.parse_args()
    
    # Load fact claims
    fact_claims_file = 'data/processed/processed_fact_claims_1739933287.json'
    fact_claims_df = load_fact_claims(fact_claims_file, args.year)
    
    print(f"Loaded {len(fact_claims_df)} fact claims for year {args.year}")
    print(f"Class distribution in fact claims:")
    print(fact_claims_df['mergedTextualRating'].value_counts(normalize=True))
    
    # Load NYT articles
    nyt_articles_file = f'data/raw/nyt/nyt_articles_{args.year}.json'
    nyt_articles_df = load_nyt_articles(nyt_articles_file)
    
    print(f"Loaded {len(nyt_articles_df)} NYT articles for year {args.year}")
    
    # Balance the dataset
    balanced_df = balance_dataset(fact_claims_df, nyt_articles_df)
    
    print(f"\nBalanced dataset created with {len(balanced_df)} rows")
    print(f"Class distribution in balanced dataset:")
    print(balanced_df['mergedTextualRating'].value_counts(normalize=True))
    
    # Split the dataset
    split_dataset(balanced_df, args.year)
    
    # Save the full balanced dataset
    os.makedirs('data/processed/balanced', exist_ok=True)
    balanced_df.to_csv(f'data/processed/balanced/full_set_{args.year}.csv', index=False)
    print(f"\nFull balanced dataset saved to data/processed/balanced/full_set_{args.year}.csv")

if __name__ == "__main__":
    main()
