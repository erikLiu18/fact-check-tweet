import argparse
import os
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import sys
import time
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder

# Add parent directory to system path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval_cls.vector_search import FactCheckSearch
from preprocessing.llm import QwenModel

class RetrievalAugmentedClassifier:
    def __init__(self, index_dir, model_name="Qwen/Qwen2.5-7B-Instruct", 
                 cache_dir="/home/hice1/yliu3390/scratch/.cache/huggingface",
                 device=None):
        """
        Initialize the retrieval-augmented classifier.
        
        Args:
            index_dir (str): Directory containing the vector search index
            model_name (str): The name of the LLM to use
            cache_dir (str): Directory to store the model cache
            device (str): Device to run on ('cuda' or 'cpu')
        """
        # Initialize vector search
        self.searcher = FactCheckSearch(device=device)
        self.searcher.load_index(index_dir)
        
        # Initialize LLM
        self.llm = QwenModel(model_name=model_name, cache_dir=cache_dir)
        
        # Define system prompt
        self.system_prompt = """You are a helpful assistant specializing in fact-checking. 
Your task is to determine if a given claim is REAL (true) or FAKE (false).
You will be given a claim to classify and several similar claims with their known classifications.
Use these similar examples to guide your judgment on the new claim.
Respond with a single word: either "REAL" or "FAKE".
"""
    
    def _construct_prompt(self, claim, similar_results):
        """
        Construct a prompt for the LLM based on the claim and similar results.
        
        Args:
            claim (str): The claim to classify
            similar_results (list): List of tuples (text, rating, score) from vector search
            
        Returns:
            str: The constructed prompt
        """
        prompt = f"I need to determine if the following claim is real or fake:\n\n"
        prompt += f"CLAIM TO CLASSIFY: {claim}\n\n"
        prompt += f"Here are similar claims with known classifications:\n\n"
        
        for i, (text, rating, score) in enumerate(similar_results, 1):
            # Convert rating to REAL/FAKE format - handle both string and boolean
            if isinstance(rating, str):
                classification = "REAL" if rating.lower() == "true" else "FAKE"
            else:  # boolean type
                classification = "REAL" if rating else "FAKE"
            prompt += f"SIMILAR CLAIM {i} (Classification: {classification}, Similarity: {score:.4f}):\n{text}\n\n"
        
        prompt += "Based on these similar examples, classify the given claim as either REAL or FAKE."
        return prompt
    
    def classify(self, claim, k=5, threshold=None, temperature=0.2):
        """
        Classify a claim using retrieval-augmented generation.
        
        Args:
            claim (str): The claim to classify
            k (int): Number of similar examples to retrieve
            threshold (float): Similarity threshold for retrieval
            temperature (float): LLM temperature parameter
            
        Returns:
            str: Classification result ("REAL" or "FAKE")
        """
        # Retrieve similar examples
        similar_results = self.searcher.search(claim, k=k, threshold=threshold)
        
        # Construct prompt
        prompt = self._construct_prompt(claim, similar_results)
        
        # Generate classification
        response = self.llm.generate_response(
            user_prompt=prompt, 
            system_prompt=self.system_prompt,
            temperature=temperature,
            max_new_tokens=10  # Short response
        )
        
        # Parse response to get classification
        if "REAL" in response.upper():
            return "REAL"
        elif "FAKE" in response.upper():
            return "FAKE"
        else:
            # If response is unclear, default to the most common classification in similar results
            ratings = [result[1] for result in similar_results]
            # Handle both string and boolean types
            if all(isinstance(r, str) for r in ratings):
                real_count = sum(1 for r in ratings if r.lower() == "true")
            else:
                real_count = sum(1 for r in ratings if r is True)
            fake_count = len(ratings) - real_count
            return "REAL" if real_count >= fake_count else "FAKE"
    
    def classify_dataset(self, csv_path, output_dir, batch_size=1, k=5, threshold=None, temperature=0.2):
        """
        Classify all claims in a dataset and evaluate performance.
        
        Args:
            csv_path (str): Path to CSV file with claims
            output_dir (str): Directory to save results
            batch_size (int): Batch size for processing
            k (int): Number of similar examples to retrieve
            threshold (float): Similarity threshold for retrieval
            temperature (float): LLM temperature parameter
            
        Returns:
            dict: Classification report
        """
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load dataset
        df = pd.read_csv(csv_path)
        
        # Initialize results
        results = []
        
        # Start timing
        start_time = time.time()
        
        # Process claims
        for i, row in tqdm(df.iterrows(), total=len(df), desc="Classifying claims"):
            claim = row['text']
            # Handle boolean type directly
            true_label = "REAL" if row['mergedTextualRating'] is True else "FAKE"
            
            # Classify
            pred_label = self.classify(claim, k=k, threshold=threshold, temperature=temperature)
            
            # Store result
            results.append({
                'claim': claim,
                'true_label': true_label,
                'pred_label': pred_label,
                'correct': true_label == pred_label
            })
        
        # Calculate total time
        total_time = time.time() - start_time
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Calculate metrics
        y_true = results_df['true_label'].values
        y_pred = results_df['pred_label'].values
        
        # Convert string labels to numeric for metrics calculation
        le = LabelEncoder()
        le.fit(np.concatenate([y_true, y_pred]))
        y_true_encoded = le.transform(y_true)
        y_pred_encoded = le.transform(y_pred)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_encoded, 
            y_pred_encoded, 
            average='weighted',
            zero_division=0
        )
        
        # Generate classification report
        report = classification_report(
            y_true, 
            y_pred, 
            output_dict=True,
            zero_division=0
        )
        
        # Add overall metrics
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'total_time_seconds': float(total_time),
            'avg_time_per_claim_seconds': float(total_time / len(df))
        }
        
        # Add label mapping to report
        report['label_mapping'] = {i: label for i, label in enumerate(le.classes_)}
        
        # Save results
        results_df.to_csv(output_dir / "classification_results.csv", index=False)
        
        # Save report
        with open(output_dir / "classification_report.json", 'w') as f:
            json.dump({'metrics': metrics, 'report': report}, f, indent=2)
        
        # Print summary
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Total classification time: {total_time:.2f} seconds")
        print(f"Average time per claim: {total_time / len(df):.2f} seconds")
        
        return {'metrics': metrics, 'report': report}

def main():
    parser = argparse.ArgumentParser(description='Retrieval-augmented claim classification')
    parser.add_argument('--index-dir', type=str, required=True,
                        help='Directory containing the vector search index')
    parser.add_argument('--input-csv', type=str, required=True,
                        help='Path to CSV file with claims to classify')
    parser.add_argument('--output-dir', type=str, default='models/llm',
                        help='Directory to save results')
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-7B-Instruct',
                        help='LLM model to use')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to run on (cuda/cpu)')
    parser.add_argument('--k', type=int, default=5,
                        help='Number of similar examples to retrieve')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Similarity threshold for retrieval')
    parser.add_argument('--temperature', type=float, default=0.2,
                        help='LLM temperature parameter')
    args = parser.parse_args()
    
    # Initialize classifier
    classifier = RetrievalAugmentedClassifier(
        index_dir=args.index_dir,
        model_name=args.model,
        device=args.device
    )
    
    # Classify dataset
    classifier.classify_dataset(
        csv_path=args.input_csv,
        output_dir=args.output_dir,
        k=args.k,
        threshold=args.threshold,
        temperature=args.temperature
    )

if __name__ == '__main__':
    main() 