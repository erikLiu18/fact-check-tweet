# Time-Based Feature Extraction for Fact-Checking

This document describes the time-based feature extraction system for the fact-checking pipeline. The system allows you to specify different time ranges for vector database creation, training, validation, and testing.

## Overview

The system implements a temporal approach to fact-checking where:
1. An initial vector database is built using historical data as a knowledge base
2. As time progresses, each month's data is incrementally added to the index
3. For each claim, only information available up to that claim's time is used for feature extraction
4. Different time periods can be specified for training, validation, and testing

## Files

- `temporal_data_manager.py`: Core module for handling time-based data loading and processing
- `extract_features.py`: Updated to support time-based feature extraction
- `vector_search.py`: Vector search functionality (used for index management)
- `run_temporal_extraction.sh`: Example script demonstrating usage

## Time Range Format

Time ranges are specified in the format `YYYY-MM:YYYY-MM`, where:
- `YYYY`: 4-digit year
- `MM`: 2-digit month (1-12)

Example: `2021-01:2023-12` represents the period from January 2021 to December 2023.

## Usage

### Command Line Arguments

```bash
python src/retrieval_cls/extract_features.py \
  --data-root data/processed/balanced \
  --index-time-range 2021-01:2023-12 \
  --train-time-range 2024-01:2024-06 \
  --val-time-range 2024-07:2024-09 \
  --test-time-range 2024-10:2024-12 \
  --index-dir models/search_indices \
  --output-dir features \
  --top-k 5 \
  --batch-size 512 \
  --device cuda
```

### Arguments Explained

- `--data-root`: Root directory containing yearly subdirectories with data
- `--index-time-range`: Initial time range for vector database construction
- `--train-time-range`: Time range for training data
- `--val-time-range`: Time range for validation data
- `--test-time-range`: Time range for test data
- `--index-dir`: Directory to store vector search indices
- `--output-dir`: Directory for output feature files
- `--top-k`: Number of top similar articles to retrieve
- `--batch-size`: Batch size for generating embeddings
- `--model`: Sentence transformer model to use
- `--device`: Device for model inference (cuda or cpu)

## Implementation Details

### Incremental Vector Database Growth

The system uses a progressive approach to building the vector database:

1. An initial index is built with all data from the `index-time-range`
2. When processing data chronologically:
   - Before processing month N, all data up to month N-1 is added to the index
   - The system efficiently extends the existing index rather than rebuilding it each time
   - Each version of the index is saved for reuse

This ensures both temporal integrity and computational efficiency.

### Temporal Data Management

For each claim in the training, validation, or test sets:
1. The system identifies the claim's month and year
2. It uses a vector index built with all data up to the previous month
3. Features are computed using only this temporally constrained knowledge

### Output

The system produces three feature files:
1. `train_features.csv`: Features for training data
2. `val_features.csv`: Features for validation data
3. `test_features.csv`: Features for test data

Each feature file includes all the original features plus:
- `year`: Year of the claim
- `month`: Month of the claim

## Example Walkthrough

Here's how the system processes data over time:

1. **Initial Setup**: 
   - Build index with all data from 2021-01 to 2023-12

2. **Training Data Processing**:
   - For claims from 2024-01: Use the initial index (2021-01 to 2023-12)
   - For claims from 2024-02: Add 2024-01 data to the index first
   - For claims from 2024-03: Add 2024-02 data to the index first
   - ... and so on

3. **Validation Data Processing**:
   - Continues where training left off, extending the same indices

4. **Test Data Processing**:
   - Continues where validation left off, extending the same indices

This ensures that each claim is evaluated using only the knowledge that would have been available at the time it was made. 