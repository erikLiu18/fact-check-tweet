#!/usr/bin/env python3
"""
train_xgboost.py

This script trains an XGBoost classifier on the features extracted by extract_features.py.
It loads the train, validation, and test datasets, trains the model with hyperparameter 
tuning, evaluates performance, and saves the trained model for later use.
"""

import argparse
import pandas as pd
import numpy as np
import os
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import xgboost as xgb
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(features_dir):
    """
    Load train, validation, and test features from CSV files.
    
    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    train_df = pd.read_csv(os.path.join(features_dir, "train_features.csv"))
    val_df = pd.read_csv(os.path.join(features_dir, "val_features.csv"))
    test_df = pd.read_csv(os.path.join(features_dir, "test_features.csv"))
    
    # Separate features and labels, excluding year and month
    feature_cols = [col for col in train_df.columns if col not in ['label', 'year', 'month']]
    
    X_train = train_df[feature_cols]
    y_train = train_df['label']
    
    X_val = val_df[feature_cols]
    y_val = val_df['label']
    
    X_test = test_df[feature_cols]
    y_test = test_df['label']
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def train_model(X_train, y_train, X_val, y_val, params=None):
    """
    Train an XGBoost model with hyperparameter tuning if no params are provided.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        params: Optional hyperparameters dict
        
    Returns:
        Trained XGBoost model and best parameters
    """
    # Create DMatrix objects for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    # If params are not provided, perform hyperparameter tuning
    if params is None:
        print("Performing hyperparameter tuning...")
        best_params = {}
        best_score = 0
        
        # Grid search for hyperparameters
        for max_depth in [3, 5, 7, 9]:
            for learning_rate in [0.01, 0.05, 0.1, 0.2]:
                for n_estimators in [50, 100, 200]:
                    for subsample in [0.8, 1.0]:
                        for colsample_bytree in [0.8, 1.0]:
                            params = {
                                'objective': 'binary:logistic',
                                'eval_metric': 'logloss',
                                'max_depth': max_depth,
                                'learning_rate': learning_rate,
                                'n_estimators': n_estimators,
                                'subsample': subsample,
                                'colsample_bytree': colsample_bytree,
                                'tree_method': 'hist'
                            }
                            
                            # Train with early stopping
                            model = xgb.train(
                                params,
                                dtrain,
                                params['n_estimators'],
                                evals=[(dval, 'validation')],
                                early_stopping_rounds=10,
                                verbose_eval=False
                            )
                            
                            # Get validation score
                            y_pred = model.predict(dval)
                            y_pred_binary = (y_pred > 0.5).astype(int)
                            score = f1_score(y_val, y_pred_binary)
                            
                            if score > best_score:
                                best_score = score
                                best_params = params
                                best_params['best_iteration'] = model.best_iteration
        
        print(f"Best parameters: {best_params}")
        print(f"Best validation F1 score: {best_score:.4f}")
        
        # Train final model with best parameters
        final_params = best_params.copy()
        final_params['n_estimators'] = best_params['best_iteration']
        
    else:
        final_params = params
    
    # Train the final model
    print("Training final model...")
    model = xgb.train(
        final_params,
        dtrain,
        final_params['n_estimators'],
        evals=[(dtrain, 'train'), (dval, 'validation')],
        verbose_eval=100
    )
    
    return model, final_params

def evaluate_model(model, X, y, dataset_name="Test"):
    """
    Evaluate model performance on the given dataset.
    
    Args:
        model: Trained XGBoost model
        X: Features
        y: Labels
        dataset_name: Name of the dataset for printing
    
    Returns:
        Dictionary of evaluation metrics
    """
    dmatrix = xgb.DMatrix(X)
    y_prob = model.predict(dmatrix)
    y_pred = (y_prob > 0.5).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average='binary')
    recall = recall_score(y, y_pred, average='binary')
    f1 = f1_score(y, y_pred, average='binary')
    cm = confusion_matrix(y, y_pred)
    
    # Print results
    print(f"\n{dataset_name} Set Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    
    # Return metrics
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm.tolist()
    }
    
    return metrics

def plot_feature_importance(model, X_train, output_dir):
    """
    Plot and save feature importance graph.
    
    Args:
        model: Trained XGBoost model
        X_train: Training features
        output_dir: Directory to save the plot
    """
    # Get feature importance
    importance = model.get_score(importance_type='gain')
    
    # Convert to DataFrame for plotting
    importance_df = pd.DataFrame({
        'Feature': list(importance.keys()),
        'Importance': list(importance.values())
    })
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    # Plot
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title('Feature Importance (Gain)')
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, 'feature_importance.png')
    plt.savefig(plot_path)
    print(f"Feature importance plot saved to {plot_path}")

def main():
    parser = argparse.ArgumentParser(description="Train XGBoost model for fake news classification")
    parser.add_argument('--features-dir', type=str, default='data/features',
                        help='Directory containing feature CSV files')
    parser.add_argument('--output-dir', type=str, default='models/xgboost',
                        help='Output directory for model and results')
    parser.add_argument('--params-file', type=str, default=None,
                        help='JSON file containing XGBoost parameters (optional)')
    parser.add_argument('--skip-tuning', action='store_true',
                        help='Skip hyperparameter tuning and use default parameters')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(args.features_dir)
    
    # Load parameters if provided
    params = None
    if args.params_file:
        with open(args.params_file, 'r') as f:
            params = json.load(f)
    elif args.skip_tuning:
        # Default parameters
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 5,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'tree_method': 'hist'
        }
    
    # Train model
    model, best_params = train_model(X_train, y_train, X_val, y_val, params)
    
    # Evaluate on all datasets
    train_metrics = evaluate_model(model, X_train, y_train, "Training")
    val_metrics = evaluate_model(model, X_val, y_val, "Validation")
    test_metrics = evaluate_model(model, X_test, y_test, "Test")
    
    # Save metrics
    metrics = {
        'train': train_metrics,
        'validation': val_metrics,
        'test': test_metrics,
        'parameters': best_params
    }
    
    with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Plot feature importance
    plot_feature_importance(model, X_train, args.output_dir)
    
    # Save model
    model_path = os.path.join(args.output_dir, 'model.json')
    model.save_model(model_path)
    print(f"Model saved to {model_path}")
    
    # Save parameters
    with open(os.path.join(args.output_dir, 'best_params.json'), 'w') as f:
        json.dump(best_params, f, indent=2)
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main() 

# python src/retrieval_cls/train_xgboost.py --features-dir data/features --output-dir models/xgboost