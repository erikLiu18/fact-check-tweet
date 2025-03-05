import pandas as pd
import json

def load_claims_from_file(filename):
    with open(filename, 'r') as json_file:
        return json.load(json_file)

def convert_claims_to_dataframe(claims):
    # Normalize the claimReview field to flatten the structure
    claims_data = []
    for claim in claims:
        for review in claim.get('claimReview', []):
            claims_data.append({
                'text': claim.get('text'),
                'claimant': claim.get('claimant'),
                'claimDate': pd.to_datetime(claim.get('claimDate')),
                'publisherName': review.get('publisher', {}).get('name'),
                'publisherSite': review.get('publisher', {}).get('site'),
                'reviewUrl': review.get('url'),
                'reviewTitle': review.get('title'), 
                'reviewDate': pd.to_datetime(review.get('reviewDate')),
                'textualRating': review.get('textualRating')
            })
    return pd.DataFrame(claims_data)