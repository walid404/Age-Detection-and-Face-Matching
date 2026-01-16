import torch
import numpy as np


def mae(preds, targets):
    return torch.mean(torch.abs(preds - targets)).item()


def mse(preds, targets):
    return torch.mean((preds - targets) ** 2).item()


def rmse(preds, targets):
    return torch.sqrt(torch.mean((preds - targets) ** 2)).item()


def r2_score(preds, targets):
    preds = preds.cpu().numpy()
    targets = targets.cpu().numpy()
    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    return 1 - ss_res / ss_tot
