#!/usr/bin/env python3
"""
extract_features.py

This script reads a CSV of claims, retrieves the top-k most similar fact-check articles
from a pre-built FAISS index, computes interaction-based features (similarity + entailment),
and outputs a feature table suitable for training an XGBoost classifier.

The script now supports time-based data management for training, validation, and testing.
"""

import argparse
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from vector_search import FactCheckSearch
from temporal_data_manager import TemporalDataManager, TimeRange
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

def parse_time_range(time_range_str: str) -> TimeRange:
    """Parse a time range string in format YYYY-MM:YYYY-MM into a TimeRange object."""
    start, end = time_range_str.split(':')
    start_year, start_month = map(int, start.split('-'))
    end_year, end_month = map(int, end.split('-'))
    return TimeRange(start_year, start_month, end_year, end_month)

def main():
    parser = argparse.ArgumentParser(description="Extract features for fake news classification with time-based data management")
    parser.add_argument('--data-root', type=str, default='data/processed/balanced',
                        help='Root directory containing yearly subdirectories with data')
    
    # Time range arguments
    parser.add_argument('--index-time-range', type=str, required=True,
                        help='Time range for building the vector DB index in format YYYY-MM:YYYY-MM')
    parser.add_argument('--train-time-range', type=str, required=True,
                        help='Time range for training data in format YYYY-MM:YYYY-MM')
    parser.add_argument('--val-time-range', type=str, required=True,
                        help='Time range for validation data in format YYYY-MM:YYYY-MM')
    parser.add_argument('--test-time-range', type=str, required=True,
                        help='Time range for test data in format YYYY-MM:YYYY-MM')
    
    # Other arguments
    parser.add_argument('--index-dir', type=str, default='models/search_indices',
                        help='Directory to store vector search indices')
    parser.add_argument('--output-dir', type=str, default='features',
                        help='Output directory for feature files')
    parser.add_argument('--top-k', type=int, default=5,
                        help='Number of top similar articles to retrieve')
    parser.add_argument('--nli-model', default='cross-encoder/nli-deberta-v3-large',
                        help='HuggingFace NLI model name')
    parser.add_argument('--batch-size', type=int, default=512,
                        help='Batch size for generating embeddings')
    parser.add_argument('--model', type=str, default='all-MiniLM-L6-v2',
                        help='Sentence transformer model to use')
    parser.add_argument('--device', default=None,
                        help='Device for model inference (cuda or cpu)')
    args = parser.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.index_dir, exist_ok=True)

    # Initialize data manager and vector search
    data_manager = TemporalDataManager(args.data_root)
    searcher = FactCheckSearch(model_name=args.model, device=device)

    # Load NLI model
    tokenizer, nli_model = load_nli_model(args.nli_model, device)

    # Parse time ranges
    index_time_range = parse_time_range(args.index_time_range)
    train_time_range = parse_time_range(args.train_time_range)
    val_time_range = parse_time_range(args.val_time_range)
    test_time_range = parse_time_range(args.test_time_range)

    # Process training data
    print(f"Processing training data for time range {train_time_range}...")
    data_manager.extract_features_with_temporal_awareness(
        data_time_range=train_time_range,
        index_time_range=index_time_range,
        searcher=searcher,
        index_dir=args.index_dir,
        output_file=os.path.join(args.output_dir, "train_features.csv"),
        tokenizer=tokenizer,
        nli_model=nli_model,
        device=device,
        top_k=args.top_k,
        compute_features_fn=compute_features,
        batch_size=args.batch_size
    )

    # Process validation data
    print(f"Processing validation data for time range {val_time_range}...")
    data_manager.extract_features_with_temporal_awareness(
        data_time_range=val_time_range,
        index_time_range=train_time_range,  # Use training time range as starting point for validation
        searcher=searcher,
        index_dir=args.index_dir,
        output_file=os.path.join(args.output_dir, "val_features.csv"),
        tokenizer=tokenizer,
        nli_model=nli_model,
        device=device,
        top_k=args.top_k,
        compute_features_fn=compute_features,
        batch_size=args.batch_size
    )

    # Process test data
    print(f"Processing test data for time range {test_time_range}...")
    data_manager.extract_features_with_temporal_awareness(
        data_time_range=test_time_range,
        index_time_range=val_time_range,  # Use validation time range as starting point for test
        searcher=searcher,
        index_dir=args.index_dir,
        output_file=os.path.join(args.output_dir, "test_features.csv"),
        tokenizer=tokenizer,
        nli_model=nli_model,
        device=device,
        top_k=args.top_k,
        compute_features_fn=compute_features,
        batch_size=args.batch_size
    )

    print(f"All features extracted and saved to {args.output_dir}")

if __name__ == "__main__":
    main()
