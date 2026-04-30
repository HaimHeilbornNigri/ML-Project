import torch
import torch.nn as nn

"""
The baseline model will have the following steps:
    The tweet is converted into a sequence of word indices.
    Pre-trained GloVe embeddings convert each word index into a dense vector.
    Global average pooling is applied to average all of the word vectors in the tweet, producing a single fixed-sized vector.
    The vector is passed through one or two fully connected linear layers.
    The final layer outputs logits for the six emotion classes.
The baseline model ignores most word order and focuses on the overall average meaning of the tweet.
"""

class BaselineModel(nn.Module):

    def __init__(self, 
                 vocab_size: int, #This is the size of the vocabulary (number of unique words)
                 embed_dim: int = 100, #This is the dimension of the GloVe embeddings
                 hidden_dim: int = 128, #This is the size of the hidden layer in the classifier
                 num_classes: int = 6, #This is the number of emotions that the tweets will be classified into
                 padding_idx: int = 0): #This is the index used for padding tokens
        
        super().__init__() #Initialize model
        
        # ====================
        # 1. Embedding Layer
        # ====================

        self.embedding = nn.Embedding( #V
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=padding_idx
        )

        #Todo: load pre-trained GloVe weights here
        
        # ==============
        # 2. Classifier
        # ==============

        self.classifier == nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.3), #Maybe make this adjustable later?
            nn.Linear(hidden_dim, num_classes)
        )

    #Forward pass stuff

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor: #input_ids is the tensor of the shape containing the word indices btw. The function should return logits hopefully.
        
        #Step 1: Embedding Lookup

        embedded = self.embedding(input_ids)

        #Step 2: Global Average Pooling

        pooled = torch.mean(embedded, dim=1)

        #Step 3: Here Comes the Classifier (dun-dun-da!)

        logits = self.classifer(pooled)

        return logits 
    
    #Helper function for loading the GloVe weights:


def load_glove_weights(model: BaselineModel, glove_weights: torch.Tensor):
    
    with torch.no_grad():
        model.embedding.weight.copy_(glove_weights)

    

        

