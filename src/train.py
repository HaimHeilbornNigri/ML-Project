import torch

def train(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0

    for X, y in dataloader:
        optimizer.zero_grad()
        preds = model(X)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)