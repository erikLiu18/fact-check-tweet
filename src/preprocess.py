import pandas as pd

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

if __name__ == "__main__":
    df_filtered = preprocess_claims('data/raw/fact_claims_1739933287.json')
    print(f'Number of rows in df_filtered: {df_filtered.shape[0]}')

    # Save df_filtered to a JSON file in the data/processed folder
    output_file_path = 'data/processed/processed_fact_claims_1739933287.json'
    df_filtered.to_json(output_file_path, orient='records', lines=True)
    print(f'Data saved to {output_file_path}')