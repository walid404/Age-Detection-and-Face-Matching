import torch
import argparse
from src.controller.age_inference_controller import predict_age
from src.model.utils.load import load_model, load_image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def predict(image_path, model_name, checkpoint_path, img_size=224):
    
    image = load_image(image_path)
    model = load_model(model_name, checkpoint_path)
    prediction = predict_age(model, image, img_size)

    return prediction

def parse_args():
    parser = argparse.ArgumentParser(description="Age Prediction Inference")
    parser.add_argument(
        "--image_path", 
        type=str, 
        required=True, 
        help="Path to the input image"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        required=True, 
        help="Model architecture name"
    )
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        required=True, 
        help="Path to the model checkpoint"
    )
    parser.add_argument(
        "--img_size", 
        type=int, 
        default=224, 
        help="Input image size for the model"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    age = predict(
        args.image_path, 
        args.model, 
        args.checkpoint, 
        args.img_size
    )
    print(f"Predicted Age: {age}")
