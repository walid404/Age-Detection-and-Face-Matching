import torch.nn as nn

def get_loss(name: str):
    if name == "mae":
        return nn.L1Loss()
    elif name == "mse":
        return nn.MSELoss()
    else:
        raise ValueError("Unsupported loss")
