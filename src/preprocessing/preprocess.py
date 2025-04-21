import pandas as pd
import os
import argparse
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

def create_full_dataset(df, output_dir):
    """
    Create a CSV file containing only the text and mergedTextualRating columns.
    
    Args:
        df (pd.DataFrame): The preprocessed dataframe
    """
    # Select only the required columns
    full_set = df[['text', 'mergedTextualRating']].copy()
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the full dataset
    full_set.to_csv(os.path.join(output_dir, 'full_set.csv'), index=False)
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

def split_dataset_by_month(df, month, output_dir):
    """
    Create train, validation, and test sets for a specific month.
    
    Args:
        df (pd.DataFrame): The full dataset
        month (str): Month in format 'YYYY-MM'
        output_dir (str): Directory to save the splits
    """
    # Filter data for the specified month
    df['month'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m')
    month_df = df[df['month'] == month].copy()
    
    if len(month_df) == 0:
        print(f"No data found for month {month}")
        return
    
    # Create month directory
    month_dir = os.path.join(output_dir, month)
    os.makedirs(month_dir, exist_ok=True)

    full_set = create_full_dataset(month_df, month_dir)
    
    # Split the dataset
    train_df, temp_df = train_test_split(
        full_set, 
        train_size=0.7, 
        random_state=42,
        stratify=full_set['mergedTextualRating']
    )
    
    val_df, test_df = train_test_split(
        temp_df,
        train_size=0.15/(0.15 + 0.15),
        random_state=42,
        stratify=temp_df['mergedTextualRating']
    )
    
    # Save the splits
    train_df.to_csv(os.path.join(month_dir, 'train_set.csv'), index=False)
    val_df.to_csv(os.path.join(month_dir, 'val_set.csv'), index=False)
    test_df.to_csv(os.path.join(month_dir, 'test_set.csv'), index=False)
    
    print(f'Dataset for {month} split into:')
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
    parser = argparse.ArgumentParser(description='Preprocess the fact-checked claims.')
    parser.add_argument('--file', required=True, type=str, default='data/raw/fact_claims_1745200721.json', help='The file to preprocess.')
    parser.add_argument('--task', required=True, type=str, choices=['process_raw', 'split', 'split_month'], help='Preprocessing task to perform')
    args = parser.parse_args()

    if args.task == 'process_raw':
        # Process the claims
        df_filtered = preprocess_claims(args.file)
        print(f'Number of rows in df_filtered: {df_filtered.shape[0]}')

        # Save df_filtered to a JSON file in the data/processed folder
        # Extract the original filename from the path and prepend "processed_"
        original_filename = os.path.basename(args.file)
        output_file_path = f'data/processed/processed_{original_filename}'

        df_filtered.to_json(output_file_path, orient='records', lines=True)
        print(f'Data saved to {output_file_path}')
    elif args.task == 'split':
        # Save df_filtered to a JSON file in the data/processed folder
        output_file_path = args.file
        
        df_filtered = pd.read_json(output_file_path, orient='records', lines=True)
        
        # Create and save the full dataset
        full_set = create_full_dataset(df_filtered, output_dir='data/processed/')
        
        # Split and save the dataset
        split_dataset(full_set)
    elif args.task == 'split_month':
        # Save df_filtered to a JSON file in the data/processed folder
        output_file_path = args.file
        
        df_filtered = pd.read_json(output_file_path, orient='records', lines=True)
        
        # Create monthly splits for January, February, and March 2022
        months = ['2024-10', '2024-11', '2024-12']
        output_dir = 'data/processed/months'
        os.makedirs(output_dir, exist_ok=True)
        
        for month in months:
            print(f"\nProcessing month: {month}")
            split_dataset_by_month(df_filtered, month, output_dir)