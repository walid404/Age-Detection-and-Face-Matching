import argparse
from controller.age_face_inference_controller import AgeFaceMatchingInference
from typing import Dict


def match_faces(
    image_path_1: str,
    image_path_2: str,
    age_model_name: str = "resnet50",
    age_model_weights: str = "saved_models/resnet50_random_bs16_lr0.0005_ep60.pt",
    match_threshold: float = 0.45
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
        Name of the age prediction model architecture (default "resnet50").
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
        age_model_name=age_model_name,
        age_model_weights=age_model_weights,
        match_threshold=match_threshold,
    )

    result = pipeline.infer(image_path_1, image_path_2)
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Age-aware Face Matching CLI")
    parser.add_argument("--image1", type=str, required=True, help="Path to the first image")
    parser.add_argument("--image2", type=str, required=True, help="Path to the second image")
    parser.add_argument("--model", type=str, default="resnet50", help="Age model architecture (default: resnet50)")
    parser.add_argument("--weights", type=str,
                        default="saved_models/resnet50_random_bs16_lr0.0005_ep60.pt",
                        help="Path to age model weights")
    parser.add_argument("--threshold", type=float, default=0.45, help="Matching threshold (default: 0.45)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    result = match_faces(
        image_path_1=args.image1,
        image_path_2=args.image2,
        age_model_name=args.model,
        age_model_weights=args.weights,
        match_threshold=args.threshold
    )

    print(result)
