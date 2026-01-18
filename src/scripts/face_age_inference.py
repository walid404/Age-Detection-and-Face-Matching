import argparse
from src.controller.age_face_inference_controller import AgeFaceMatchingInference
from src.model.utils.load import load_image, load_model
from src.model.networks.arcface_model import ArcFaceExtractor
from src.model.networks.face_matcher import FaceMatcher
from typing import Dict
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def age_face_match_inference(
    image_path_1: str,
    image_path_2: str,
    age_model_name: str = "mobilenet",
    age_model_weights: str = "src/saved_models/mobilenet_identity_bs16_lr0.0005_ep60.pt",
    match_threshold: float = 0.45,
    img_size: int = 224,
) -> Dict:
    """
    Perform age-aware face matching between two images.

    Parameters
    ----------
    image_path_1 : str
        Path to the first image.
    image_path_2 : str
        Path to the second image.
    age_model_name : str, optional
        Name of the age prediction model architecture (default "mobilenet").
    age_model_weights : str, optional
        Path to the pretrained weights of the age model.
    match_threshold : float, optional
        Threshold for determining a face match (default 0.45).

    Returns
    -------
    dict
        Dictionary containing the matching result and similarity score.
    """
    pipeline = AgeFaceMatchingInference(
        age_model=load_model(age_model_name, age_model_weights),
        extractor=ArcFaceExtractor(device=DEVICE),
        matcher=FaceMatcher(match_threshold),
        img_size=img_size,
    )

    img1 = load_image(image_path_1)
    img2 = load_image(image_path_2)
    result = pipeline.infer(img1, img2)
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Age-aware Face Matching CLI")
    parser.add_argument(
        "--image_path1", 
        type=str, 
        required=True, 
        help="Path to the first image"
    )
    parser.add_argument(
        "--image_path2", 
        type=str, 
        required=True, 
        help="Path to the second image"
    )
    parser.add_argument(
        "--model", 
        type=str,
        default="mobilenet", 
        help="Age model architecture (default: mobilenet)"
    )
    parser.add_argument(
        "--weights", 
        type=str,
        default="src/saved_models/mobilenet_identity_bs16_lr0.0005_ep60.pt",
        help="Path to age model weights"
    )
    parser.add_argument(
        "--threshold", 
        type=float, 
        default=0.45, 
        help="Matching threshold (default: 0.45)"
    )
    parser.add_argument(
        "--img_size", 
        type=int, 
        default=224, 
        help="Input image size for the age model (default: 224)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    result = age_face_match_inference(
        image_path_1=args.image_path1,
        image_path_2=args.image_path2,
        age_model_name=args.model,
        age_model_weights=args.weights,
        match_threshold=args.threshold
    )

    print(result)
