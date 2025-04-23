# Fake News Classification

## Introduction
This project aims to develop and evaluate automated fact-checking systems. It implements multiple approaches including:

* RoBERTa-based text classification models for direct fact-checking
* Retrieval-augmented classification that leverages existing fact-checks
* LLM-based fact verification with retrieval context
* Temporal analysis of misinformation patterns

The codebase supports the entire fact-checking pipeline from data collection through preprocessing, feature extraction, model training, and evaluation. Our goal is to analyze the feasibility of automated fact-checking and understand the patterns of misinformation spread across social media platforms.

## Setup
```bash
module load anaconda3
conda create -n iec -f environment.yml
conda activate iec
```

## Repository Organization
* `src/` - Source code for the project
  * `preprocessing/` - Scripts for data preprocessing and balancing
  * `data_collection/` - Scripts for collecting the dataset
  * `simple_cls/` - Simple classification models
  * `retrieval_cls/` - Retrieval-based classification
* `data/` - Data storage
  * `raw/` - Raw collected data
  * `processed/` - Processed and balanced datasets
  * `features/` - Extracted features for XGBoost
  * `fnc/` - Previous fact-checking datasets from the course
* `models/` - Trained models and evaluation results
  * `roberta_classifier/` - RoBERTa-based classification model
  * `classifier_eval/` - Evaluation results for classifiers
  * `search_indices/` - Indices for retrieval-based approaches
* `past_ko_project/` - Reference materials from previous knowledge organization project
* `analysis/` - Analysis scripts and notebooks
* `logs/` - Log files from training and evaluation

## Usage

### Data Collection
* `src/data_collection/fetch_all_fact_checks.py` - Fetches fact-check claims from multiple publishers
  * Usage: `python src/data_collection/fetch_all_fact_checks.py`
  * Automatically finds the latest data file and fetches only new data
* `src/data_collection/fetch_fact_checks_by_topic.py` - Fetches fact-checks for specific topics or publishers
  * Usage: `python src/data_collection/fetch_fact_checks_by_topic.py --publisher politifact.com --max-age 30`
* `src/data_collection/fetch_nyt_articles.py` - Fetches articles from the New York Times API
  * Usage: `python src/data_collection/fetch_nyt_articles.py --query "climate change" --begin-date 2022-01-01`

### Data Preprocessing
* `src/preprocessing/preprocess.py` - Preprocesses raw fact-check data
  * Process raw data: `python src/preprocessing/preprocess.py --file data/raw/fact_claims_1234567890.json --task process_raw`
  * Split dataset: `python src/preprocessing/preprocess.py --file data/processed/processed_fact_claims.json --task split`
  * Create monthly splits: `python src/preprocessing/preprocess.py --file data/processed/processed_fact_claims.json --task split_month`
* `src/preprocessing/balance_dataset.py` - Creates balanced datasets with equal true/false distributions
  * Usage: `python src/preprocessing/balance_dataset.py --input data/processed/full_set.csv --output data/processed/balanced`
* `src/preprocessing/generate_synthetic_data.py` - Generates synthetic data for training
  * Usage: `python src/preprocessing/generate_synthetic_data.py --input data/processed/imbalanced.csv --output data/generated/`
* `src/preprocessing/check_empty_text.py` - Identifies and filters out claims with empty or invalid text
  * Usage: `python src/preprocessing/check_empty_text.py --input data/processed/raw_set.csv`

### Simple Classification
* `src/simple_cls/train_classifier.py` - Trains and evaluates a RoBERTa classifier
  * Train a new model: `python src/simple_cls/train_classifier.py --train`
  * Evaluate an existing model: `python src/simple_cls/train_classifier.py --evaluate --model-dir models/roberta_classifier`
  * Train and evaluate in one go: `python src/simple_cls/train_classifier.py --train --evaluate`
* `src/simple_cls/classify_text.py` - Uses a trained model to classify new text
  * Usage: `python src/simple_cls/classify_text.py --model-dir models/roberta_classifier --text "This is a claim to classify"`

### Retrieval-based Classification
* `src/retrieval_cls/vector_search.py` - Creates and manages vector search indices for claims
  * Build index: `python src/retrieval_cls/vector_search.py --build --data data/processed/full_set.csv --output models/search_indices`
  * Search: `python src/retrieval_cls/vector_search.py --search --index models/search_indices --query "climate change"`
* `src/retrieval_cls/extract_features.py` - Extracts features from claims for retrieval
  * Usage: `python src/retrieval_cls/extract_features.py --input data/processed/full_set.csv --output data/features/`
* `src/retrieval_cls/llm_classifier.py` - LLM-based fact-checking using retrieval augmentation
  * Usage: `python src/retrieval_cls/llm_classifier.py --index models/search_indices --test data/processed/test_set.csv --output models/llm_eval`
* `src/retrieval_cls/train_xgboost.py` - Trains an XGBoost classifier on extracted features
  * Usage: `python src/retrieval_cls/train_xgboost.py --train-data data/features/train_features.csv --test-data data/features/test_features.csv`
* `src/retrieval_cls/predict.py` - Predicts fact-check ratings using trained models
  * Usage: `python src/retrieval_cls/predict.py --model models/xgboost/model.pkl --data data/features/new_features.csv`
* `src/retrieval_cls/temporal_data_manager.py` - Manages temporal aspect of data for time-aware models
  * Usage: `python src/retrieval_cls/temporal_data_manager.py --input data/processed/full_set.csv --output data/processed/temporal`
* `src/retrieval_cls/run_temporal_extraction.sh` - Bash script to extract temporal features in parallel
  * Usage: `bash src/retrieval_cls/run_temporal_extraction.sh`

## Notes
To update conda environment file, run the following command:
```bash
conda env export > environment.yml
```

To allocate GPU on PACE ICE:
```
salloc --gres=gpu:H100:1 --ntasks-per-node=1
```

## Important links to prior datasets and KO project
* Workflow of the KO project and tutorial: [link](https://drive.google.com/file/d/1FQ-ZDHSC4dq0d38EIF1J92_zNFdSYoDo/view?usp=sharing)
* [NELA Dataset](https://gtvault-my.sharepoint.com/:f:/g/personal/khu83_gatech_edu/EpLrHHhqikxKmNnffXBvD30BufXfZsfUMYNzOGj5FFm6Cw?e=7hSyvO)
* [Annotated Dataset - OLD FNC](https://gtvault-my.sharepoint.com/:f:/g/personal/khu83_gatech_edu/En-VZMxCJSpAlJoHwthr5-sBVjSehHCytZICund8S5Zx3Q?e=2vdvYR)
* [Spreadsheet for raw FNC datasets](https://gtvault-my.sharepoint.com/:x:/g/personal/khu83_gatech_edu/ERro17H5Qv9JrcgRJV50g30Bp3W0pQO7uVHYGdFfl8SROw?e=qBP8va)
    * Spring 25 represents unused datasets

## Related Literature
https://misinforeview.hks.harvard.edu/article/fact-checking-fact-checkers-a-data-driven-approach/