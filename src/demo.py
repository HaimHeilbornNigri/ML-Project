import torch
import torch.nn.functional as F

from src.dataset import EmotionDataset
from src.models.cnn_model import EmotionCNN
from src.models.baseline import BaselineModel


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_SEQ_LENGTH = 50
EMBED_DIM = 100
HIDDEN_DIM = 128
NUM_FILTERS = 100
DROPOUT = 0.5


def load_models(vocab_size):
    cnn = EmotionCNN(
        vocab_size=vocab_size,
        embed_dim=EMBED_DIM,
        num_filters=NUM_FILTERS,
        hidden_dim=HIDDEN_DIM,
        dropout_rate=DROPOUT
    ).to(DEVICE)

    baseline = BaselineModel(
        vocab_size=vocab_size,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM
    ).to(DEVICE)

    cnn.load_state_dict(torch.load("outputs/models/cnn_final.pth", map_location=DEVICE))
    baseline.load_state_dict(torch.load("outputs/models/baseline_final.pth", map_location=DEVICE))

    cnn.eval()
    baseline.eval()

    return cnn, baseline


def encode_text(text, vocab):
    tokens = vocab.encode(text)
    tokens = tokens[:MAX_SEQ_LENGTH]
    tokens += [0] * (MAX_SEQ_LENGTH - len(tokens))
    return torch.tensor(tokens).unsqueeze(0).to(DEVICE)


def predict(model, x):
    with torch.no_grad():
        out = model(x)
        probs = F.softmax(out, dim=1)
        pred = torch.argmax(probs, dim=1).item()
    return pred, probs.squeeze().cpu().numpy()


def main():
    print("\nEmotion Demo (type 'quit' to exit)\n")

    vocab = torch.load("outputs/models/vocab.pth")
    dataset = EmotionDataset(
        file_path="data/kaggle/train.txt",
        vocab=vocab,
        max_seq_length=MAX_SEQ_LENGTH,
        is_train=False
    )

    label_names = dataset.label_names
    vocab_size = len(vocab.word2idx)

    cnn, baseline = load_models(vocab_size)

    while True:
        text = input("Enter text: ")
        if text.lower() == "quit":
            break

        x = encode_text(text, dataset.vocab)

        cnn_pred, cnn_probs = predict(cnn, x)
        base_pred, base_probs = predict(baseline, x)

        print("\nCNN:")
        print(f"  {label_names[cnn_pred]} ({cnn_probs[cnn_pred]:.3f})")

        print("Baseline:")
        print(f"  {label_names[base_pred]} ({base_probs[base_pred]:.3f})")
        print("-" * 40)


if __name__ == "__main__":
    main()