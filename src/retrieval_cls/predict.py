#!/usr/bin/env python3
"""
predict.py

This script loads a trained XGBoost model and makes predictions on new feature data.
It can be used to predict whether a claim is fake news based on the retrieval-based 
features extracted using extract_features.py.
"""

import argparse
import pandas as pd
import numpy as np
import os
import json
import xgboost as xgb
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def load_model(model_path):
    """
    Load a trained XGBoost model from file.
    
    Args:
        model_path: Path to the saved model file
        
    Returns:
        Loaded XGBoost model
    """
    model = xgb.Booster()
    model.load_model(model_path)
    return model

def save_confusion_matrix(cm, labels, output_path):
    """
    Create and save a confusion matrix visualization.
    
    Args:
        cm: Confusion matrix array
        labels: Class labels
        output_path: Path to save the visualization
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def predict(model, features_file, output_file=None, eval_dir=None):
    """
    Make predictions using a trained model on the given features.
    
    Args:
        model: Trained XGBoost model
        features_file: Path to CSV file containing features
        output_file: Optional path to save predictions
        eval_dir: Directory to save evaluation metrics
        
    Returns:
        DataFrame with original data and predictions
    """
    # Load features
    df = pd.read_csv(features_file)
    
    # Extract feature columns (excluding label, year, month, and claim)
    feature_cols = [col for col in df.columns if col not in ['label', 'year', 'month', 'claim']]
    X = df[feature_cols]
    
    # Create DMatrix
    dmatrix = xgb.DMatrix(X)
    
    # Make predictions
    probabilities = model.predict(dmatrix)
    predictions = (probabilities > 0.5).astype(int)
    
    # Add predictions to the dataframe
    result_df = df.copy()
    result_df['probability'] = probabilities
    result_df['prediction'] = predictions
    
    # Print prediction statistics
    print(f"Predictions summary:")
    print(f"Total samples: {len(result_df)}")
    print(f"Predicted real (1): {sum(predictions)}")
    print(f"Predicted fake (0): {len(predictions) - sum(predictions)}")
    
    # Create evaluation directory if needed
    if eval_dir:
        os.makedirs(eval_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        eval_prefix = os.path.join(eval_dir, f"eval_{timestamp}")
    
    # If ground truth is available, calculate and save metrics
    if 'label' in result_df.columns:
        y_true = result_df['label'].values
        y_pred = result_df['prediction'].values
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average='binary')
        cm = confusion_matrix(y_true, y_pred)
        
        # Print metrics
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("\nConfusion Matrix:")
        print(cm)
        
        # Get detailed classification report
        report = classification_report(y_true, y_pred, target_names=['Fake', 'Real'])
        print("\nClassification Report:")
        print(report)
        
        # Save metrics if evaluation directory is specified
        if eval_dir:
            # Save metrics to JSON
            metrics = {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'confusion_matrix': cm.tolist(),
                'total_samples': int(len(result_df)),
                'predicted_real': int(sum(predictions)),
                'predicted_fake': int(len(predictions) - sum(predictions)),
                'timestamp': timestamp,
                'features_file': features_file
            }
            
            with open(f"{eval_prefix}_metrics.json", 'w') as f:
                json.dump(metrics, f, indent=2)
            
            # Save confusion matrix visualization
            save_confusion_matrix(cm, ['Fake', 'Real'], f"{eval_prefix}_confusion_matrix.png")
            
            # Save classification report
            with open(f"{eval_prefix}_classification_report.txt", 'w') as f:
                f.write(f"Feature file: {features_file}\n")
                f.write(f"Timestamp: {timestamp}\n\n")
                f.write("Classification Report:\n")
                f.write(report)
                f.write("\n\nConfusion Matrix:\n")
                f.write(str(cm))
            
            print(f"Evaluation metrics saved to {eval_dir}")
    
    # Save predictions if output file is specified
    if output_file:
        result_df.to_csv(output_file, index=False)
        print(f"Predictions saved to {output_file}")
    
    return result_df

def main():
    parser = argparse.ArgumentParser(description="Make predictions using a trained XGBoost model")
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to the trained XGBoost model file')
    parser.add_argument('--features-file', type=str, required=True,
                        help='Path to CSV file containing features')
    parser.add_argument('--output-file', type=str, default=None,
                        help='Path to save predictions (optional)')
    parser.add_argument('--eval-dir', type=str, default='models/xgboost/eval',
                        help='Directory to save evaluation metrics')
    
    args = parser.parse_args()
    
    # Load model
    model = load_model(args.model_path)
    
    # Make predictions and evaluate
    predict(model, args.features_file, args.output_file, args.eval_dir)

if __name__ == "__main__":
    main() 

# Example usage:
# python src/retrieval_cls/predict.py --model-path models/xgboost/model.json --features-file data/features/test_features.csv --output-file predictions.csv