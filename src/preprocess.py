import pandas as pd
import os
from sklearn.model_selection import train_test_split

from config import RATING_MERGE_DICT
from util import load_claims_from_file, convert_claims_to_dataframe


def preprocess_claims(file_path):
    # Load and convert claims
    df = convert_claims_to_dataframe(load_claims_from_file(file_path))

    # Filter dataframe
    df_filtered = df.dropna(subset=['claimDate', 'reviewDate'], how='all').copy()

    # Create a new date column and drop unnecessary columns
    df_filtered.loc[:, 'date'] = df_filtered[['claimDate', 'reviewDate']].min(axis=1)
    df_filtered = df_filtered.drop(columns=['claimDate', 'reviewDate', 'publisherSite', 'reviewUrl'])

    # Merge textual ratings
    df_filtered['mergedTextualRating'] = df_filtered['textualRating'].str.lower().map(RATING_MERGE_DICT).fillna(df_filtered['textualRating'].str.lower())

    df_filtered = df_filtered[df_filtered['mergedTextualRating'].isin(['false', 'true'])]
    
    return df_filtered

def create_full_dataset(df):
    """
    Create a CSV file containing only the text and mergedTextualRating columns.
    
    Args:
        df (pd.DataFrame): The preprocessed dataframe
    """
    # Select only the required columns
    full_set = df[['text', 'mergedTextualRating']].copy()
    
    # Create the output directory if it doesn't exist
    os.makedirs('data/processed', exist_ok=True)
    
    # Save the full dataset
    full_set.to_csv('data/processed/full_set.csv', index=False)
    print(f'Full dataset saved with {len(full_set)} rows')
    
    return full_set

def split_dataset(df, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42):
    """
    Split the dataset into training, validation, and test sets.
    
    Args:
        df (pd.DataFrame): The full dataset
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
    
    # Save the splits
    train_df.to_csv('data/processed/train_set.csv', index=False)
    val_df.to_csv('data/processed/val_set.csv', index=False)
    test_df.to_csv('data/processed/test_set.csv', index=False)
    
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

if __name__ == "__main__":
    # Process the claims
    # df_filtered = preprocess_claims('data/raw/fact_claims_1739933287.json')
    # print(f'Number of rows in df_filtered: {df_filtered.shape[0]}')

    # Save df_filtered to a JSON file in the data/processed folder
    output_file_path = 'data/processed/processed_fact_claims_1739933287.json'
    # df_filtered.to_json(output_file_path, orient='records', lines=True)
    # print(f'Data saved to {output_file_path}')

    df_filtered = pd.read_json(output_file_path, orient='records', lines=True)
    
    # Create and save the full dataset
    full_set = create_full_dataset(df_filtered)
    
    # Split and save the dataset
    split_dataset(full_set)