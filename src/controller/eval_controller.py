import torch
from torch.utils.data import DataLoader
from model.training.trainer import eval_epoch
from model.training.losses import get_loss

def run_evaluation(model, test_set, batch_size=32, config):
    loader = DataLoader(test_set, batch_size=batch_size)
    criterion = get_loss(config["training"]["loss"])
    return eval_epoch(model, loader, criterion)
