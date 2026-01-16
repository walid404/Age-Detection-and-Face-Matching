import torch
from model.utils.metrics import mae, mse, rmse, r2_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0

    for x, y in dataloader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        preds = model(x).squeeze()
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


def eval_epoch(model, dataloader, criterion):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            preds = model(x).squeeze()
            loss = criterion(preds, y)
            total_loss += loss.item()

    return total_loss / len(dataloader)

def evaluate_full(model, dataloader):
    model.eval()
    preds_all, targets_all = [], []

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            preds = model(x).squeeze()
            preds_all.append(preds)
            targets_all.append(y)

    preds_all = torch.cat(preds_all)
    targets_all = torch.cat(targets_all)

    return {
        "mae": mae(preds_all, targets_all),
        "mse": mse(preds_all, targets_all),
        "rmse": rmse(preds_all, targets_all),
        "r2": r2_score(preds_all, targets_all),
    }
