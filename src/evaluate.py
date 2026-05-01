import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path
import numpy as np

#Import models
from src.models.baseline import BaselineModel
from src.models.cnn_model import EmotionCNN

def evaluate_models():

    DEVICE = torch.device('cuda'if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")

    MAX_SEQ_LENGTH = 50
    EMBED_DIM = 100
    NUM_FILTERS = 100
    HIDDEN_DIM = 128
    DROPOUT_RATE = 0.5

    # =======================
    # 1. Loading the Dataset
    # =======================

    print("Process: Loading dataset...")
    from dataset import EmotionDataset
    vocab = torch.load("outputs/models/vocab.pth")

    test_dataset = EmotionDataset(
        csv_file='data/test.csv',
        vocab=vocab,
        max_seq_length=MAX_SEQ_LENGTH,
        is_train = False
    )
    
    #Dataloader with batch size of 1 for an analysis of a singular sample
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False
    )

    vocab_size = len(test_dataset.vocab)

    # =========================
    # 2. Load Trained Models
    # =========================

    print("Process: Loading in trained models...")

    baseline_model = BaselineModel(
        vocab_size=vocab_size,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        freeze_embeddings=False
    ).to(DEVICE)

    cnn_model = EmotionCNN(
        vocab_size=vocab_size,
        embed_dim=EMBED_DIM,
        num_filters = NUM_FILTERS,
        hidden_dim=HIDDEN_DIM,
        dropout_rate=DROPOUT_RATE,
        freeze_embeddings=False
    ).to(DEVICE)

    #Loading in the saved weights
    
    baseline_model.load_state_dict(torch.load('outputs/models/baseline_final.pth', map_location=DEVICE))
    cnn_model.load_state_dict(torch.load('outputs/models/cnn_final.pth', map_location=DEVICE))

    baseline_model.eval()
    cnn_model.eval()

    print("Models have been loaded in successfully")

    # ==========================
    # 3. Evaluating the Models
    # ==========================

    print("Evaluating the baseline model...")
    baseline_preds, baseline_labels = get_predictions(baseline_model, test_loader, DEVICE)

    print("Evaluating the baseline model...")
    cnn_preds, cnn_labels = get_predictions(cnn_model, test_loader, DEVICE)

    # =========================================
    # 4. Generating Metrics and Visualizations
    # =========================================

    #Baseline model
    save_evaluation_results(
        preds=baseline_preds,
        labels=baseline_labels,
        dataset=test_dataset,
        model_name="Baseline"
    )

    #CNN model
    save_evaluation_results(
        preds=cnn_preds,
        labels=cnn_labels,
        dataset=test_dataset,
        model_name="CNN "
    )

    print("Evaluation completed! Check out the 'outputs/' directory for results!")


def get_predictions(model, dataloader, device):

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for input_ids, labels in dataloader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            
            outputs = model(input_ids)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_preds), np.array(all_labels)
    
def save_evaluation_results(preds, labels, dataset, model_name):

    #Create the output folder
    Path("outputs/evaluation").mkdir(parents=True, exist_ok=True)

    #The Classificatin Report (top secret [classified])
    report = classification_report(
        labels, 
        preds,
        target_names=dataset.label_names,
        digits=4)
    
    print(f"\n============== {model_name} Results =============")
    print(report)

    with open(f"outputs/evaluation/{model_name.lower()}_report.txt", "w") as f: f.write(report)

    #Confusion Matrix
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=dataset.label_names,
        yticklabels=dataset.label_names)
    
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(f"outputs/evaluation/{model_name.lower()}_cm.png")
    plt.close()


