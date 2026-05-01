import matplotlib.pyplot as plt
from pathlib import Path
import torch

def save_model(model, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def plot_curves(train_vals, val_vals, title, save_path):
    plt.figure()
    plt.plot(train_vals, label="Train")
    plt.plot(val_vals, label="Validation")
    plt.title(title)
    plt.legend()
    plt.savefig(save_path)
    plt.close()