import torch
import torch.nn as nn
import torch.nn.functional as F

"""
This main model will have the following layers:
    1. The embedding layer will convert word indices into pre-trained GloVe vectors. First, frozen embeddings will be used and then fine-tuning will be done as an experiment.
    2. Three parallel convolutional branches of kernel sizes 2, 3, and 4 will be used to implement a one-dimensional convolution. The varying kernel sizes is so the model can interpret emotional patterns of different lengths. 
    3. Batch-normalization will be done on each branch after each convolution.
    4. ReLU activation will be done on each branch after batch-norm.
    5. One-dimensional max pooling is done on each branch to keep the strongest activation from each branch (aka the strongest emotional signal).
    6. Concatenation is done on all three branches to combine them into one vector.
    7. Dropout at 0.4 or 0.5 is done to prevent overfitting.
    8. One or two linear layers with ReLU in between output on of six emotional classes.
"""

class EmotionCNN(nn.Module):
    
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int = 100,
                 num_filters: int = 100,
                 hidden_dim: int = 128,
                 num_classes: int = 6,
                 dropout_rate: float = 0.5,
                 padding_idx: int = 0,
                 freeze_embeddings: bool = True):
        
        super().__init__()

        # ================================================
        # 1. Embedding Layer (reused from baseline model)
        # ================================================

        self.embedding = nn.Embedding( #This will convert word indices to vectors so we can actually use 'em
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=padding_idx
        )

        # ====== Bonus =======
        # Freezing Logic
        # ====================

        self.freeze_embeddings = freeze_embeddings
        if freeze_embeddings:
            self.embedding.weight.requires_grad = False

        # =======================================================
        # 2. Convolutional Branches (With Kernel Sizes 2, 3, 4)
        # =======================================================

        #Branch Numero Uno (Kernel Size 2)

        self.conv2 = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=num_filters,
            kernel_size=2,
            padding=1
        )

        self.bn2 = nn.BatchNorm1d(num_filters)

        #Branch Triforce (Kernel Size 3)

        self.conv3 = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=num_filters,
            kernel_size=3,
            padding=1
        )

        self.bn3 = nn.BatchNorm1d(num_filters)

        #Branch Fourth Wall (Kernel Size 4)

        self.conv4 = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=num_filters,
            kernel_size=4,
            padding=2 #Kernel size four apparently needs this ig
        )

        self.bn4 = nn.BatchNorm1d(num_filters)

        # ==================================================
        # The Classifier (adjusted from the baseline model)
        # ==================================================

        self.dropout = nn.Dropout(p=dropout_rate)

        self.classifier = nn.Sequential(
            nn.Linear(num_filters * 3, hidden_dim), #These are fully connected layers btw. embed_dim replaced with num_filters * 3
            nn.ReLU(), #Activation stufff. Same as baseline model
            nn.Dropout(p=dropout_rate), #This variable is finally well-adjusted. *Silence* Are you not entertained? (insert gladiator meme)
            nn.Linear(hidden_dim, num_classes) #Same as baseline model
        )

    #Forward pass incoming... Bam!
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        
        # =========================
        # Step 1: Embedding Lookup
        # =========================

        embedded = self.embedding(input_ids) #The input_ids are batch_size by seq_len

        embedded = embedded.transpose(1, 2) #Transpose to fit conv1d expectations of batch_size by seq_len by embed_dim

        # ==============================================
        # Step 2: Three Parallel Convolutional Branches
        # ==============================================

        #Branch 1 (Numero Uno): Kernel Size 2 Electric Bugaloo
        x2 = self.conv2(embedded)
        x2 = self.bn2(x2) #Batch normalization
        x2 = F.relu(x2) #ReLU
        x2 = F.max_pool1d(x2, kernel_size=x2.shape[2]) #Global max pooling
        x2 = x2.squeeze(2) #Basically removes the last dimension

        #Branch 2 (Triforce): Kernel Size 3
        x3 = self.conv3(embedded)
        x3 = self.bn3(x3) #Batch normalization
        x3 = F.relu(x3) #ReLU
        x3 = F.max_pool1d(x3, kernel_size=x3.shape[2]) #Global max pooling
        x3 = x3.squeeze(2) #Basically removes the last dimension

        #Branch 3 (Fourth Wall): Kernel Size 4
        x4 = self.conv4(embedded)
        x4 = self.bn4(x4) #Batch normalization
        x4 = F.relu(x4) #ReLU
        x4 = F.max_pool1d(x4, kernel_size=x4.shape[2]) #Global max pooling
        x4 = x4.squeeze(2) #Basically removes the last dimension

        # ====================================================
        # Step 3: It's Concatenation Time (Combine Branches)
        # ====================================================

        combined = torch.cat([x2, x3, x4,], dim=1) #Shape: batch_size by num_filters * 3

        # ============================================
        # Step 4: (Highschool) Dropout and Classifier
        # ============================================

        combined = self.dropout(combined)
        logits = self.classifier(combined)

        return logits
    
def load_glove_weights(model, glove_weights: torch.Tensor, freeze: bool = True):
    
    with torch.no_grad():
        model.embedding.weight.copy_(glove_weights)

    model.embedding.weight.requires_grad = not freeze
    model.freeze_embeddings = freeze
    




