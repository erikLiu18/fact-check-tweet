import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import argparse
import torch
from pathlib import Path
import json
from typing import List, Tuple
import sys
from tqdm import tqdm

class FactCheckSearch:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', device: str = None):
        """
        Initialize the search system.
        
        Args:
            model_name: Name of the sentence transformer model to use
            device: Device to run the model on ('cuda' or 'cpu')
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print(f"Loading model {model_name} on {device}...")
        self.model = SentenceTransformer(model_name, device=device)
        self.device = device
        self.index = None
        self.texts = None
        self.ratings = None
        
    def build_index(self, csv_path: str, batch_size: int = 32) -> None:
        """
        Build FAISS index from the CSV file.
        
        Args:
            csv_path: Path to the CSV file
            batch_size: Batch size for generating embeddings
        """
        print(f"Loading data from {csv_path}...")
        df = pd.read_csv(csv_path)
        self.texts = df['text'].tolist()
        self.ratings = df['mergedTextualRating'].tolist()
        
        print("Generating embeddings...")
        embeddings = []
        num_batches = (len(self.texts) + batch_size - 1) // batch_size
        
        for i in tqdm(range(0, len(self.texts), batch_size), total=num_batches, desc="Processing batches"):
            batch_texts = self.texts[i:i + batch_size]
            batch_embeddings = self.model.encode(
                batch_texts,
                show_progress_bar=False,  # Disable internal progress bar
                convert_to_numpy=True
            )
            embeddings.append(batch_embeddings)
        
        embeddings = np.vstack(embeddings)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Create FAISS index
        print("Building FAISS index...")
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings.astype('float32'))
        
        print(f"Index built with {len(self.texts)} documents")
    
    def save_index(self, output_dir: str) -> None:
        """
        Save the index and metadata to disk.
        
        Args:
            output_dir: Directory to save the index and metadata
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(output_dir / "index.faiss"))
        
        # Save metadata
        metadata = {
            'texts': self.texts,
            'ratings': self.ratings
        }
        with open(output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f)
        
        print(f"Index and metadata saved to {output_dir}")
    
    def load_index(self, index_dir: str) -> None:
        """
        Load the index and metadata from disk.
        
        Args:
            index_dir: Directory containing the index and metadata
        """
        index_dir = Path(index_dir)
        
        # Load FAISS index
        self.index = faiss.read_index(str(index_dir / "index.faiss"))
        
        # Load metadata
        with open(index_dir / "metadata.json", 'r') as f:
            metadata = json.load(f)
            self.texts = metadata['texts']
            self.ratings = metadata['ratings']
        
        print(f"Index loaded with {len(self.texts)} documents")
    
    def search(self, query: str, k: int = 5, threshold: float = 0.5) -> List[Tuple[str, str, float]]:
        """
        Search for similar texts.
        
        Args:
            query: Search query
            k: Number of results to return
            threshold: Similarity score threshold
            
        Returns:
            List of tuples (text, rating, score)
        """
        # Generate query embedding
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        
        # Filter by threshold and format results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if score >= threshold:
                results.append((self.texts[idx], self.ratings[idx], float(score)))
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Build and query FAISS index for fact checks')
    parser.add_argument('--input', type=str, required=True,
                        help='Input CSV file path')
    parser.add_argument('--model', type=str, default='all-MiniLM-L6-v2',
                        help='Sentence transformer model to use')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to run the model on (cuda/cpu)')
    parser.add_argument('--batch-size', type=int, default=512,
                        help='Batch size for generating embeddings')
    parser.add_argument('--output-dir', type=str, default='models/search_index',
                        help='Directory to save the index')
    parser.add_argument('--load-index', action='store_true',
                        help='Load existing index instead of building new one')
    parser.add_argument('--top-k', type=int, default=5,
                        help='Number of results to return')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Similarity score threshold')
    args = parser.parse_args()
    
    # Initialize search system
    searcher = FactCheckSearch(model_name=args.model, device=args.device)
    
    if args.load_index:
        searcher.load_index(args.output_dir)
    else:
        searcher.build_index(args.input, batch_size=args.batch_size)
        searcher.save_index(args.output_dir)
    
    # Interactive search
    print("\nEnter your search query (or 'quit' to exit):")
    while True:
        query = input("\nQuery: ").strip()
        if query.lower() == 'quit':
            break
        
        results = searcher.search(query, k=args.top_k, threshold=args.threshold)
        
        if not results:
            print("No results found above the threshold.")
            continue
        
        print("\nResults:")
        for i, (text, rating, score) in enumerate(results, 1):
            print(f"\n{i}. Score: {score:.4f}")
            print(f"Rating: {rating}")
            print(f"Text: {text}")

if __name__ == '__main__':
    main() 