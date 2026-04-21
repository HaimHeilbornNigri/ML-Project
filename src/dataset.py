import pandas as pd
import torch
from torch.utils.data import Dataset
from preprocessing import clean_text, tokenize

class EmotionDataset(Dataset):
    def __init__(self, file_path, vocab, max_len=50):
        self.data = pd.read_csv(file_path)
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = clean_text(self.data.iloc[idx]['text'])
        tokens = tokenize(text)
        indices = self.vocab.numericalize(tokens)

        # padding
        if len(indices) < self.max_len:
            indices += [0] * (self.max_len - len(indices))
        else:
            indices = indices[:self.max_len]

        label = self.data.iloc[idx]['label']

        return torch.tensor(indices), torch.tensor(label)