import torch
import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, 100, kernel_size=k)
            for k in [2, 3, 4]
        ])

        self.fc = nn.Linear(300, num_classes)

    def forward(self, x):
        x = self.embedding(x)              # (batch, seq, embed)
        x = x.permute(0, 2, 1)             # (batch, embed, seq)

        convs = [torch.relu(conv(x)) for conv in self.convs]
        pools = [torch.max(c, dim=2)[0] for c in convs]

        x = torch.cat(pools, dim=1)
        return self.fc(x)