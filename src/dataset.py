import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from vocab import Vocabulary

class EmotionDataset(Dataset):
    def __init__(self, csv_file, vocab=None, max_seq_length=50, is_train=True):
        self.data = pd.read_csv(csv_file)
        self.max_seq_length = max_seq_length

        self.label_names = sorted(self.data["label"].unique())
        self.label2idx = {label: i for i, label in enumerate(self.label_names)}

        texts = self.data["text"].tolist()

        if is_train:
            self.vocab = Vocabulary()
            self.vocab.build(texts)
        else:
            assert vocab is not None, "Must pass vocab for validation/test"
            self.vocab = vocab

        self.encoded_texts = [self.vocab.encode(t) for t in texts]
        self.labels = [self.label2idx[l] for l in self.data["label"]]

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


def create_dataloaders(train_csv, test_csv, batch_size=64, max_seq_length=50):

    train_dataset = EmotionDataset(
        train_csv,
        max_seq_length=max_seq_length,
        is_train=True
    )

    vocab = train_dataset.vocab

    val_dataset = EmotionDataset(
        test_csv,
        vocab=vocab,
        max_seq_length=max_seq_length,
        is_train=False
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader, vocab