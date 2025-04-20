import argparse
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification

def classify_text(text, model_dir):
    """
    Classify a single text using the trained model.
    
    Args:
        text: The text to classify
        model_dir: Directory containing the trained model
        
    Returns:
        prediction: 0 for False, 1 for True
        confidence: Confidence score for the prediction
    """
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the model and tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(model_dir)
    model = RobertaForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    
    # Tokenize the input text
    inputs = tokenizer(text, truncation=True, padding=True, return_tensors='pt')
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        prediction = torch.argmax(probabilities, dim=-1).item()
        confidence = probabilities[0][prediction].item()
    
    return prediction, confidence

def main():
    parser = argparse.ArgumentParser(description='Classify text using trained RoBERTa classifier')
    parser.add_argument('--model-dir', type=str, default='models/roberta_classifier',
                        help='Directory containing the trained model')
    parser.add_argument('--text', type=str,
                        help='Text to classify')
    parser.add_argument('--interactive', action='store_true',
                        help='Run in interactive mode to classify multiple texts')
    args = parser.parse_args()
    
    if not args.text and not args.interactive:
        parser.error("Either --text or --interactive must be specified")
    
    # Print device information
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    if args.text:
        # Classify the provided text
        prediction, confidence = classify_text(args.text, args.model_dir)
        print_classification_result(args.text, prediction, confidence)
    
    if args.interactive:
        print("Interactive mode. Enter text to classify, or 'q' to quit.")
        while True:
            text = input("\nEnter text: ")
            if text.lower() == 'q':
                break
                
            if not text.strip():
                print("Please enter some text.")
                continue
                
            prediction, confidence = classify_text(text, args.model_dir)
            print_classification_result(text, prediction, confidence)

def print_classification_result(text, prediction, confidence):
    """Print the classification result in a readable format."""
    label = "True" if prediction == 1 else "False"
    print(f"\nText: {text}")
    print(f"Classification: {label}")
    print(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")

if __name__ == '__main__':
    main()
