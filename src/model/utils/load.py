import torch
from PIL import Image
from src.model.networks.age_models import get_age_model


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(model_name, checkpoint_path):
    model = get_age_model(model_name)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.to(DEVICE)
    return model

def load_image(image_path: str):
        img = Image.open(image_path).convert("RGB")
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        return img
