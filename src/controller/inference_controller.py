import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def predict_age(model, image_tensor):
    model.eval()
    with torch.no_grad():
        pred = model(image_tensor.unsqueeze(0).to(DEVICE))
    return pred.item()
