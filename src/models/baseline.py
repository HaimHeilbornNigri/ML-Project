import torch.nn as nn

class BaselineModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)          # (batch, seq, embed)
        x = x.permute(0, 2, 1)         # (batch, embed, seq)
        x = self.pool(x).squeeze(2)    # average pooling
        return self.fc(x)