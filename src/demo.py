# demo.py
import torch
from src.models.baseline import BaselineModel
from src.models.cnn_model import EmotionCNN
from src.dataset import EmotionDataset   # for vocab and label mapping
from src.preprocessing import tokenize   # or your clean + tokenize function

def load_models_and_vocab():
    """Load vocabulary, Baseline and CNN models"""
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load vocabulary
    vocab = torch.load("outputs/models/vocab.pth", weights_only=False)
    
    vocab_size = len(vocab)
    
    # Initialize models
    baseline_model = BaselineModel(vocab_size=vocab_size, freeze_embeddings=False).to(DEVICE)
    cnn_model = EmotionCNN(vocab_size=vocab_size, freeze_embeddings=False).to(DEVICE)
    
    # Load trained weights
    baseline_model.load_state_dict(torch.load('outputs/models/baseline_final.pth', map_location=DEVICE))
    cnn_model.load_state_dict(torch.load('outputs/models/cnn_final.pth', map_location=DEVICE))
    
    baseline_model.eval()
    cnn_model.eval()
    
    return baseline_model, cnn_model, vocab, DEVICE


def predict_emotion(text: str, model, vocab, device, max_seq_length=50):
    """Predict emotion for a single text"""
    # 1. Tokenize and encode
    encoded = vocab.encode(text)
    
    # 2. Pad sequence
    if len(encoded) < max_seq_length:
        encoded = encoded + [0] * (max_seq_length - len(encoded))
    else:
        encoded = encoded[:max_seq_length]
    
    # 3. Convert to tensor
    input_ids = torch.tensor([encoded]).to(device)   # shape: (1, seq_len)
    
    # 4. Forward pass
    with torch.no_grad():
        logits = model(input_ids)
        predicted_class = torch.argmax(logits, dim=1).item()
    
    return predicted_class


def main():
    print("Loading models...")
    baseline_model, cnn_model, vocab, device = load_models_and_vocab()
    
    # Create label mapping (reverse)
    label_names = list(vocab.label2idx.keys()) if hasattr(vocab, 'label2idx') else \
                  ["anger", "fear", "joy", "love", "sadness", "surprise"]   # adjust if needed
    
    print("\n=== Emotion Classification Demo ===\n")
    print("Enter a sentence/tweet (type 'quit' to exit)\n")
    
    while True:
        text = input("You: ").strip()
        
        if text.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
            
        if not text:
            continue
        
        # Get predictions from both models
        baseline_pred_idx = predict_emotion(text, baseline_model, vocab, device)
        cnn_pred_idx = predict_emotion(text, cnn_model, vocab, device)
        
        baseline_emotion = label_names[baseline_pred_idx]
        cnn_emotion = label_names[cnn_pred_idx]
        
        print(f"\nBaseline Model  →  {baseline_emotion.upper()}")
        print(f"CNN Model       →  {cnn_emotion.upper()}  ⭐")
        print("-" * 60)


if __name__ == "__main__":
    main()