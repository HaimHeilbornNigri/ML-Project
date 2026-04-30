import torch

def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for X, y in dataloader:
            preds = model(X)
            predicted = preds.argmax(dim=1)
            correct += (predicted == y).sum().item()
            total += y.size(0)

    return correct / total