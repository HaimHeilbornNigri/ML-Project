#Libraries
import torch #PyTorch stuffff
import torch.nn as nn
import torch.optim as optim #Optimizers over here
from torch.utils.data import DataLoader #This one is for efficient batch loading btw
import matplotlib.pyplot as plt #Using this to plot stuff
from pathlib import Path #This one is for creating directories
import time #Need this one to measure how long an epoch takes
import numpy as np

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

    print("Process: Loading datasets and creating dataloaders...")

    train_loader, val_loader, vocab, = create_dataloaders(
        train_csv='data/train.csv',
        test_csv='data/test.csv',
        batch_size=BATCH_SIZE,
        max_seq_length=MAX_SEQ_LENGTH
    )

    vocab_size = len(vocab)
    print(f'Vocabulary size: {vocab_size}')

    #Saving vocabulary for evaluation consistency
    Path("outputs/models").mkdir(parents=True, exist_ok=True)
    torch.save(vocab, "outputs/models/vocab.pth")

    print("Vocabulary has been SAVED to outputs/models/vocab.pth")

    # =======================
    # 2. Initializing Models
    # =======================

    print("Process: Initializing models...")

    baseline_model = BaselineModel( #Initializing the baseline model
        vocab_size=vocab_size,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        freeze_embeddings=FREEZE_EMBEDDINGS
    ).to(DEVICE)

    cnn_model = EmotionCNN( #Initializing the cnn model
        vocab_size=vocab_size,
        embed_dim=EMBED_DIM,
        num_filters=NUM_FILTERS,
        hidden_dim=HIDDEN_DIM,
        dropout_rate=DROPOUT_RATE,
        freeze_embeddings=FREEZE_EMBEDDINGS
    ).to(DEVICE)

    # ========================================
    # 3. Loading Pre-Trained GloVe Embeddings
    # ========================================

    print("Process: Loading GloVe Embeddings")

    glove_path = "data/glove.6B.100d.txt"

    glove_weights = build_glove_matrix(
        glove_path=glove_path,
        vocab=vocab,
        embed_dim=EMBED_DIM
    )

    baseline_model.embedding.weight.data = glove_weights.clone()
    cnn_model.embedding.weight.data = glove_weights.clone()

    #To freeze or not to freeze, that is the question:

    if FREEZE_EMBEDDINGS:
        baseline_model.embedding.weight.requires_grad = False
        cnn_model.embedding.weight.requires_grad = False

    print("GloVe embeddings loaded successfully into both models!!!")

    # ====================================================
    # 4. Defining the Loss Function and the Optimizers
    # ====================================================

    criterion = nn.CrossEntropyLoss() #This is the loss function

    #Optimizers over here (using Adam)
    optimizer_baseline = optim.Adam(baseline_model.parameters(), lr=LEARNING_RATE)
    optimizer_cnn = optim.Adam(cnn_model.parameters(), lr=LEARNING_RATE)

    # =================
    # 5. Training Loop
    # =================
    print("Process: Initiating training loop...")

    for epoch in range(NUM_EPOCHS): #Looping for the number of epochs stated
        epoch_start = time.time() #Record the start time

        #Training the baseline model
        train_loss_b, train_acc_b = train_one_epoch(
            baseline_model, 
            train_loader,
            optimizer_baseline,
            criterion,
            DEVICE)
        
        #Training the cnn model
        train_loss_c, train_acc_c = train_one_epoch(
            cnn_model, 
            train_loader,
            optimizer_cnn,
            criterion,
            DEVICE)
        
        #Evaluating the baseline model
        val_loss_b, val_acc_b = evaluate(baseline_model, val_loader, criterion, DEVICE)

        #Evaluating the cnn model
        val_loss_c, val_acc_c = evaluate(cnn_model, val_loader, criterion, DEVICE)

        #Print epoch summary 
        epoch_time = time.time() - epoch_start
        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}] - {epoch_time:.1f}s")
        
        print(f"================================ Baseline ====================================")
        print(f"Train Loss: {train_loss_b:.4f} | Train Accuracy: {train_acc_b:.3f}")
        print(f"Val Loss: {val_loss_b:.4f} | Val Accuracy: {val_acc_b:.3f}")

        print(f"=================================== CNN ======================================")
        print(f"Train Loss: {train_loss_c:.4f} | Train Accuracy: {train_acc_c:.3f}")
        print(f"Val Loss: {val_loss_c:.4f} | Val Accuracy: {val_acc_c:.3f}")

        print("-" * 90)

    # =====================================================
    # 6. Saving Trained Models (pretty self-explanatory)
    # =====================================================

    print("Process: Saving models...")
    
    Path("outputs/models").mkdir(parents=True, exist_ok=True) #Making the directory to save them in

    torch.save(baseline_model.state_dict(), "outputs/models/baseline_final.pth")
    torch.save(cnn_model.state_dict(), "outputs/models/cnn_final.pth")

    print("\n Training completed! Models saved successfully! Letsss goooooo")

#Training epoch function:

def train_one_epoch(model, dataloader, optimizer, criterion, device):

    model.train() #The model is now in training mode
    total_loss = 0.0 #Setting variable that tracks total loss
    correct = 0 #Correctness tracker
    total_samples = 0 #Sample tracker

    for input_ids, labels in dataloader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        optimizer.zero_grad() #So this clears the previous, older gradients
        outputs = model(input_ids) #This is the forward pass
        loss = criterion(outputs, labels) #Computing the loss

        loss.backward() #This is the backward pass to compute the gradients
        optimizer.step() #Updating model weights

        #Computing statistics
        total_loss += loss.item() * input_ids.size(0) 
        _, predicted = torch.max(outputs, dim=1)
        correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)

    #Even more statistics yayay
    avg_loss = total_loss / total_samples
    accuracy = correct / total_samples

    return avg_loss, accuracy


#Evaluation function

def evaluate(model, dataloader, criterion, device):

    model.eval() #Setting model to the evaluation mode
    total_loss = 0.0 #Setting loss tracker to zero
    correct = 0 #Setting correct tracker to zero
    total_samples = 0 #Setting sample tracker to zero

    with torch.no_grad():
        for input_ids, labels in dataloader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            outputs = model(input_ids)
            loss = criterion(outputs, labels)

            #The following is the same as in the training epoch function

            #Computing statistics
            total_loss += loss.item() * input_ids.size(0) 
            _, predicted = torch.max(outputs, dim=1)
            correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    #Even more statistics yayay
    avg_loss = total_loss / total_samples
    accuracy = correct / total_samples

    return avg_loss, accuracy

def build_glove_matrix(glove_path, vocab, embed_dim=100):
    print("Process: Loading GloVe...")

    glove = {}

    with open(glove_path, "r", encoding="utf8") as f:
        for line in f:
            parts = line.split()
            word = parts[0]
            vec = np.asarray(parts[1:], dtype=np.float32)
            glove[word] = vec

    vocab_size = len(vocab.word2idx)
    matrix = np.random.normal(scale=0.6, size=(vocab_size, embed_dim))

    matrix[0] = np.zeros(embed_dim)

    for word, idx in vocab.word2idx.items():
        if word in glove:
            matrix[idx] = glove[word]

    return torch.tensor(matrix, dtype=torch.float32)

if __name__ == "__main__":
    main()

























    