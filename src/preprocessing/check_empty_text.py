import pandas as pd
import argparse
import os

def check_empty_text(csv_file):
    """
    Check if a CSV file has any empty text fields.
    
    Args:
        csv_file (str): Path to the CSV file
        
    Returns:
        tuple: (bool, int) - (True if empty fields found, number of empty fields)
    """
    # Check if file exists
    if not os.path.exists(csv_file):
        print(f"Error: File '{csv_file}' does not exist.")
        return True, 0
    
    # Read the CSV file
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Error reading CSV file: {str(e)}")
        return True, 0
    
    # Check if 'text' column exists
    if 'text' not in df.columns:
        print(f"Error: 'text' column not found in '{csv_file}'.")
        return True, 0
    
    # Count empty text fields
    empty_count = df['text'].isna().sum()
    
    # Count empty strings
    empty_string_count = (df['text'] == '').sum()
    
    # Count whitespace-only strings
    whitespace_count = df['text'].str.strip().eq('').sum()
    
    total_empty = empty_count + empty_string_count + whitespace_count
    
    if total_empty > 0:
        print(f"Found {total_empty} empty text fields in '{csv_file}':")
        print(f"  - {empty_count} NaN values")
        print(f"  - {empty_string_count} empty strings")
        print(f"  - {whitespace_count} whitespace-only strings")
        
        # Print the first few rows with empty text fields
        empty_rows = df[df['text'].isna() | (df['text'] == '') | (df['text'].str.strip() == '')]
        if not empty_rows.empty:
            print("\nFirst few rows with empty text fields:")
            print(empty_rows[['text', 'mergedTextualRating']].head())
        
        return True, total_empty
    else:
        print(f"No empty text fields found in '{csv_file}'.")
        return False, 0

def main():
    parser = argparse.ArgumentParser(description='Check for empty text fields in a CSV file.')
    parser.add_argument('csv_file', type=str, help='Path to the CSV file to check')
    args = parser.parse_args()
    
    has_empty, count = check_empty_text(args.csv_file)
    
    if has_empty:
        print(f"\nSummary: File '{args.csv_file}' has {count} empty text fields.")
        return 1
    else:
        print(f"\nSummary: File '{args.csv_file}' has no empty text fields.")
        return 0

if __name__ == "__main__":
    exit(main()) 