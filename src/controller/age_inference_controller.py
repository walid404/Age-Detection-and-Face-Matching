import torch
from torchvision import transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def predict_age(model, image_tensor, img_size=224):
    transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),  # values in [0, 1]
        ])
    model.eval()
    with torch.no_grad():
        pred = model(transform(image_tensor).unsqueeze(0).to(DEVICE))
    return int(round(pred.item(), 0))
