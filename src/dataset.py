import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from src.vocab import Vocabulary


# =================
# DATASET
# =================

class EmotionDataset(Dataset):
    def __init__(self, file_path, vocab=None, max_seq_length=50, is_train=True, label2idx=None):
        self.data = load_kaggle_file(file_path)
        self.max_seq_length = max_seq_length

        texts = self.data["text"].tolist()
        labels = self.data["label"].tolist()

        # ======================
        # LABEL HANDLING (FIXED)
        # ======================
        if label2idx is None:
            # only build from training set
            label_names = sorted(set(labels))
            self.label2idx = {label: i for i, label in enumerate(label_names)}
        else:
            # use shared mapping (IMPORTANT FIX)
            self.label2idx = label2idx

        # vocab handling
        if is_train:
            self.vocab = Vocabulary()
            self.vocab.build(texts)
        else:
            assert vocab is not None, "Must pass vocab for validation/test"
            self.vocab = vocab

        # encode
        self.encoded_texts = [self.vocab.encode(t) for t in texts]

        # safe label encoding (prevents crash)
        self.labels = []
        for l in labels:
            if l in self.label2idx:
                self.labels.append(self.label2idx[l])
            else:
                raise ValueError(f"Unknown label found: {l}")


    def pad(self, seq):
        if len(seq) < self.max_seq_length:
            seq = seq + [0] * (self.max_seq_length - len(seq))
        else:
            seq = seq[:self.max_seq_length]
        return seq

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.pad(self.encoded_texts[idx])
        return torch.tensor(seq), torch.tensor(self.labels[idx])


# ===========================
# DATALOADERS
# ===========================

def create_dataloaders(train_csv, val_csv, test_csv, batch_size, max_seq_length):

    # TRAIN
    train_dataset = EmotionDataset(
        train_csv,
        max_seq_length=max_seq_length,
        is_train=True
    )

    vocab = train_dataset.vocab
    label2idx = train_dataset.label2idx #Fix for labeling issues (now global labels)

    # VAL
    val_dataset = EmotionDataset(
        val_csv,
        vocab=vocab,
        max_seq_length=max_seq_length,
        is_train=False,
        label2idx=label2idx
    )

    # TEST
    test_dataset = EmotionDataset(
        test_csv,
        vocab=vocab,
        max_seq_length=max_seq_length,
        is_train=False,
        label2idx=label2idx
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    print("Vocabulary size:", len(vocab.word2idx))

    return train_loader, val_loader, test_loader, vocab


# ============================
# FILE LOADER (ROBUST)
# ============================

def load_kaggle_file(path):
    import pandas as pd

    with open(path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    # ALWAYS assume last delimiter split is label
    split_data = []

    for line in lines:
        # split ONLY on last occurrence
        if "\t" in line:
            parts = line.rsplit("\t", 1)
        elif "," in line:
            parts = line.rsplit(",", 1)
        else:
            parts = line.rsplit(";", 1)  # IMPORTANT FIX

        if len(parts) == 2:
            text, label = parts
            split_data.append([text.strip(), label.strip()])

    df = pd.DataFrame(split_data, columns=["text", "label"])

    df = df.dropna()
    df = df[df["text"].str.strip() != ""]
    df = df[df["label"].str.strip() != ""]

    print(f"Loaded dataset size: {len(df)}")

    return df