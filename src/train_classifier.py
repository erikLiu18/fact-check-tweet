import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import numpy as np
from pathlib import Path

class FactCheckDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length, return_tensors='pt')
        self.labels = labels
        
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.labels)

def load_data(file_path):
    """Load data from CSV file."""
    df = pd.read_csv(file_path)
    labels = df['mergedTextualRating'].astype(int).tolist()
    # Add debug prints
    print(f"\nLoading {file_path}")
    print("Label distribution:", np.bincount(labels))
    return df['text'].tolist(), labels

def train_model(model, train_loader, val_loader, device, num_epochs=3, learning_rate=2e-5):
    """Train the model and return the best model."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    best_val_acc = 0
    best_model = None
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} - Training'):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Validation
        model.eval()
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'Epoch {epoch + 1}/{num_epochs} - Validation'):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(outputs.logits, dim=-1)
                
                val_preds.extend(predictions.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        val_acc = accuracy_score(val_labels, val_preds)
        print(f'Epoch {epoch + 1}:')
        print(f'  Prediction distribution: {np.bincount(val_preds)}')
        print(f'  True label distribution: {np.bincount(val_labels)}')
        print(f'  Average training loss: {total_loss / len(train_loader):.4f}')
        print(f'  Validation accuracy: {val_acc:.4f}')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model.state_dict()
    
    return best_model

def evaluate_model(model, test_loader, device):
    """Evaluate the model on the test set."""
    model.eval()
    test_preds = []
    test_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1)
            
            test_preds.extend(predictions.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
    
    print('\nTest Set Results:')
    print(classification_report(test_labels, test_preds, target_names=['False', 'True']))

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load data
    print('Loading data...')
    train_texts, train_labels = load_data('data/processed/train_set.csv')
    val_texts, val_labels = load_data('data/processed/val_set.csv')
    test_texts, test_labels = load_data('data/processed/test_set.csv')
    
    # Initialize tokenizer and model
    print('Initializing model and tokenizer...')
    model_name = 'roberta-base'
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)
    
    # Create datasets
    print('Creating datasets...')
    train_dataset = FactCheckDataset(train_texts, train_labels, tokenizer)
    val_dataset = FactCheckDataset(val_texts, val_labels, tokenizer)
    test_dataset = FactCheckDataset(test_texts, test_labels, tokenizer)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    test_loader = DataLoader(test_dataset, batch_size=16)
    
    # Train the model
    print('Starting training...')
    best_model = train_model(model, train_loader, val_loader, device)
    
    # Load best model and evaluate
    print('Loading best model and evaluating...')
    model.load_state_dict(best_model)
    evaluate_model(model, test_loader, device)
    
    # Save the model
    print('Saving model...')
    output_dir = Path('models/roberta_classifier')
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f'Model saved to {output_dir}')

if __name__ == '__main__':
    main()