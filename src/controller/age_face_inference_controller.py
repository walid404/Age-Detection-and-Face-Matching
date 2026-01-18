import torch
from src.controller.age_inference_controller import predict_age
from src.controller.face_match_inference_controller import match_faces
from src.model.networks.arcface_model import ArcFaceExtractor
from src.model.networks.face_matcher import FaceMatcher
from src.model.utils.load import load_image, load_model

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
        self.age_model = load_model(age_model_name, age_model_weights)
        self.age_model.to(DEVICE)
        self.img_size = img_size
        self.match_threshold = match_threshold
        self.extractor = ArcFaceExtractor(device=DEVICE)
        self.matcher = FaceMatcher(match_threshold)

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
        img1 = load_image(image_path_1)
        img2 = load_image(image_path_2)

        # -------------------------------
        # Age prediction
        # -------------------------------
        age1 = predict_age(self.age_model, img1, self.img_size)
        age2 = predict_age(self.age_model, img2, self.img_size)

        # -------------------------------
        # Identity matching
        # -------------------------------
        match, similarity = match_faces(img1, img2, self.extractor, self.matcher).values()

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
