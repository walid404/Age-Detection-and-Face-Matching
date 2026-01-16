import torch
from torchvision import transforms
from PIL import Image
import argparse
from model.networks.age_models import get_age_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(model_name, checkpoint_path):
    model = get_age_model(model_name)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def predict(image_path, model_name, checkpoint_path, img_size=224):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),  # [0,1]
    ])

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)

    model = load_model(model_name, checkpoint_path)

    with torch.no_grad():
        prediction = model(image).item()

    return prediction


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_name", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    age = predict(args.image_name, args.model, args.checkpoint)
    print(f"Predicted Age: {age:int}")
