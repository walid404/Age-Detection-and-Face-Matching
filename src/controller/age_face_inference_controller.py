import torch
from src.controller.age_inference_controller import predict_age
from src.controller.face_match_inference_controller import match_faces

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class AgeFaceMatchingInference:
    """
    Full inference pipeline:
    - Age prediction
    - Face matching using ArcFace
    """

    def __init__(
        self,
        age_model,
        extractor,
        matcher,
        img_size: int = 224,
    ):
        # -------------------------------
        # Age model
        # -------------------------------
        self.age_model = age_model
        self.age_model.to(DEVICE)
        self.img_size = img_size
        self.extractor = extractor
        self.matcher = matcher

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------
    def infer(self, img1, img2):
        """
        Perform age prediction and face matching on two images.
        Parameters
        ----------
        img1 : Image.Image
            First input image.
        img2 : Image.Image
            Second input image.
        Returns
        -------
        dict
            Dictionary containing ages, match result, and similarity score.
        """

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
            "age_image_1": age1,
            "age_image_2": age2,
            "match": match,          # 1 = same person, 0 = different
            "similarity": round(similarity  , 3),
        }
