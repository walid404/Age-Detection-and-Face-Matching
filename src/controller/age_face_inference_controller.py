from PIL import Image
import torch
import numpy as np
from model.networks.age_models import get_age_model
from controller.age_inference_controller import predict_age
from controller.face_match_inference_controller import match_faces

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class AgeFaceMatchingInference:
    """
    Full inference pipeline:
    - Age prediction
    - Face matching using ArcFace
    """

    def __init__(
        self,
        age_model_name: str,
        age_model_weights: str,
        img_size: int = 224,
        match_threshold: float = 0.45,
    ):
        # -------------------------------
        # Age model
        # -------------------------------
        self.age_model = get_age_model(age_model_name).to(DEVICE)
        self.age_model.load_state_dict(
            torch.load(age_model_weights, map_location=DEVICE)
        )
        self.img_size = img_size
        self.match_threshold = match_threshold

    # --------------------------------------------------
    # Internal utilities
    # --------------------------------------------------
    def _load_image(self, image_path: str):
        img = Image.open(image_path).convert("RGB")
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        return img

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------
    def infer(self, image_path_1: str, image_path_2: str):
        """
        Returns:
        {
            "image_1": {"age": float},
            "image_2": {"age": float},
            "match": 0 | 1,
            "similarity": float
        }
        """

        # -------------------------------
        # Load images
        # -------------------------------
        img1 = self._load_image(image_path_1)
        img2 = self._load_image(image_path_2)

        # -------------------------------
        # Age prediction
        # -------------------------------
        age1 = predict_age(self.age_model, img1, self.img_size)
        age2 = predict_age(self.age_model, img2, self.img_size)

        # -------------------------------
        # Identity matching
        # -------------------------------
        match, similarity = match_faces(img1, img2, self.match_threshold).values()

        return {
            "image_1": {
                "path": image_path_1,
                "predicted_age": age1,
            },
            "image_2": {
                "path": image_path_2,
                "predicted_age": age2,
            },
            "match": match,          # 1 = same person, 0 = different
            "similarity": round(similarity  , 3),
        }
