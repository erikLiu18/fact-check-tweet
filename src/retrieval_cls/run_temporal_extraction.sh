#!/bin/bash
# This script demonstrates how to use the time-based feature extraction.

# Example usage for extracting features with temporal awareness
# In this example:
# - We use data from 2021-01 to 2023-12 as the initial vector database
# - Training data is from 2024-01 to 2024-06
# - Validation data is from 2024-07 to 2024-09
# - Test data is from 2024-10 to 2024-12

# Ensure the output directories exist
mkdir -p models/search_indices
mkdir -p data/features

# Run the feature extraction
python src/retrieval_cls/extract_features.py \
  --data-root data/processed/balanced \
  --index-time-range 2021-01:2023-12 \
  --train-time-range 2024-01:2024-08 \
  --val-time-range 2024-09:2024-10 \
  --test-time-range 2024-11:2024-12 \
  --index-dir models/search_indices \
  --output-dir data/features \
  --top-k 5 \
  --batch-size 512 \
  --device cuda

echo "Feature extraction complete. Results saved to data/features/ directory." 