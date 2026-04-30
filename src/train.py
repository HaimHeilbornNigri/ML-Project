#Libraries
import torch #PyTorch stuffff
import torch.nn as nn
import torch.optim as optim #Optimizers over here
from torch.utils.data import DataLoader #This one is for efficient batch loading btw
import matplotlib.pyplot as plt #Using this to plot stuff
from pathlib import Path #This one is for creating directories
import time #Need this one to measure how long an epoch takes

#Import models and dataset
from models.baseline import BaselineModel, load_glove_weights
from models.cnn_model import EmotionCNN, load_glove_weights as load_glove_cnn
from dataset import create_dataloaders

"""
This training script trains both models, performs validation, logs metrics, and saves the final trained models.
"""

def main():
    
    BATCH_SIZE = 64
    MAX_SEQ_LENGTH = 50
    EMBED_DIM = 100
    NUM_FILTERS = 100
    HIDDEN_DIM = 128
    DROPOUT_RATE = 0.5
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 12
    FREEZE_EMBEDDINGS = True #Start with frozen embeddings

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #Chooses device and prioritizes choosing cuda
    print(f"Using Device: {DEVICE}") #Print indication of whether cuda or cpu is being used

    # =============================================
    # 1. Loading Datasets and Creating Dataloaders
    # =============================================

    print("Processing: Loading datasets and creating dataloaders")

    train_loader, val_loader, vocab, = create_dataloaders(
        train_csv='data/train.csv'
        test_csv='data/test.csv'
        batch_size=BATCH_SIZE,
        max_seq_length=MAX_SEQ_LENGTH
    )








    