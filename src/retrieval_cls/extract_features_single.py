#!/usr/bin/env python3
"""
extract_features_single.py

This script reads a single CSV file containing claims with 'text' and 'mergedTextualRating' columns,
uses an existing FAISS index to retrieve similar fact-check articles, and computes interaction-based
features for classification. It's a simplified version of extract_features.py without temporal management.
"""

import argparse
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from vector_search import FactCheckSearch
import os
from pathlib import Path

def load_nli_model(model_name: str, device: str):
    """
    Load a pretrained NLI model for entailment/contradiction scoring.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir='/home/hice1/yliu3390/scratch/.cache/huggingface')
    model = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir='/home/hice1/yliu3390/scratch/.cache/huggingface').to(device)
    model.eval()
    return tokenizer, model

def compute_features(claim: str, retrieved: list, tokenizer, nli_model, device: str):
    """
    Compute aggregate and interaction features for a claim based on retrieved articles.
    """
    sims, entails, contradicts = [], [], []
    for title, _, sim in retrieved:
        sims.append(sim)
        # NLI prediction
        inputs = tokenizer(claim, title,
                           return_tensors='pt',
                           truncation=True,
                           padding=True).to(device)
        with torch.no_grad():
            logits = nli_model(**inputs).logits
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        contradicts.append(probs[0])  # class 0: contradiction
        entails.append(probs[1])      # class 1: entailment

    sims_arr = np.array(sims)
    entail_arr = np.array(entails)
    contradict_arr = np.array(contradicts)

    # Base aggregations
    feats = {
        'sim_max': sims_arr.max() if len(sims_arr) > 0 else 0,
        'sim_mean': sims_arr.mean() if len(sims_arr) > 0 else 0,
        'sim_std': sims_arr.std() if len(sims_arr) > 0 else 0,
        'sim_min': sims_arr.min() if len(sims_arr) > 0 else 0,
        'entail_max': entail_arr.max() if len(entail_arr) > 0 else 0,
        'entail_mean': entail_arr.mean() if len(entail_arr) > 0 else 0,
        'entail_std': entail_arr.std() if len(entail_arr) > 0 else 0,
        'contradict_max': contradict_arr.max() if len(contradict_arr) > 0 else 0,
        'contradict_mean': contradict_arr.mean() if len(contradict_arr) > 0 else 0,
    }

    # Interaction features
    sim_ent = sims_arr * entail_arr
    sim_contr = sims_arr * contradict_arr

    feats.update({
        'sim_ent_mean': sim_ent.mean() if len(sim_ent) > 0 else 0,
        'sim_ent_max': sim_ent.max() if len(sim_ent) > 0 else 0,
        'sim_contr_mean': sim_contr.mean() if len(sim_contr) > 0 else 0,
        'support_score': sim_ent.sum() / len(sim_ent) if len(sim_ent) > 0 else 0,
        'strong_support_count': int((sim_ent > 0.8).sum()) if len(sim_ent) > 0 else 0,
        'strong_contradict_count': int((sim_contr > 0.8).sum()) if len(sim_contr) > 0 else 0,
    })

    return feats

def main():
    parser = argparse.ArgumentParser(description="Extract features for a single CSV file using an existing index")
    parser.add_argument('--input-file', type=str, required=True,
                        help='Path to input CSV file with text and mergedTextualRating columns')
    parser.add_argument('--output-file', type=str, required=True,
                        help='Path to save extracted features')
    parser.add_argument('--index-dir', type=str, required=True,
                        help='Directory containing the existing search index')
    parser.add_argument('--top-k', type=int, default=5,
                        help='Number of top similar articles to retrieve')
    parser.add_argument('--nli-model', default='cross-encoder/nli-deberta-v3-large',
                        help='HuggingFace NLI model name')
    parser.add_argument('--model', type=str, default='all-MiniLM-L6-v2',
                        help='Sentence transformer model to use')
    parser.add_argument('--device', default=None,
                        help='Device for model inference (cuda or cpu)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for processing')
    args = parser.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # Initialize vector search and load existing index
    print(f"Loading search index from {args.index_dir}...")
    searcher = FactCheckSearch(model_name=args.model, device=device)
    searcher.load_index(args.index_dir)

    # Load NLI model
    print(f"Loading NLI model: {args.nli_model}...")
    tokenizer, nli_model = load_nli_model(args.nli_model, device)

    # Load input data
    print(f"Reading input file: {args.input_file}...")
    df = pd.read_csv(args.input_file)
    
    # Check that required columns exist
    if 'text' not in df.columns or 'mergedTextualRating' not in df.columns:
        raise ValueError("Input CSV must contain 'text' and 'mergedTextualRating' columns")

    # Process claims
    features_list = []
    print(f"Processing {len(df)} claims...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        claim = row['text']
        label = 1 if str(row['mergedTextualRating']).lower() == 'true' else 0
        
        # Get similar articles
        retrieved = searcher.search(claim, args.top_k)
        
        # Compute features
        feats = compute_features(claim, retrieved, tokenizer, nli_model, device)
        feats['label'] = label
        
        # Include original claim for reference
        # feats['claim'] = claim
        
        features_list.append(feats)
        
        # Process in batches to avoid memory issues
        if (idx + 1) % args.batch_size == 0:
            print(f"Processed {idx + 1}/{len(df)} claims")

    # Create features DataFrame
    features_df = pd.DataFrame(features_list)
    
    # Save to CSV
    features_df.to_csv(args.output_file, index=False)
    print(f"Features extracted and saved to {args.output_file}")

if __name__ == "__main__":
    main() 

# python src/retrieval_cls/extract_features_single.py --input-file data/processed/months/2024-10/train_set.csv --output-file data/features/train_features.csv --index-dir models/search_indices --top-k 5