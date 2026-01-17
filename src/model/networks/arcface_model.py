import torch
from deepface import DeepFace
import numpy as np
from PIL import Image
from typing import List, Union

class ArcFaceExtractor:
    def __init__(self, model_name: str = "ArcFace", device: str = "cpu"):
        self.model_name = model_name
        self.device = "cuda" if device == "cuda" else "cpu"

    def _prepare_image(self, image):
        if isinstance(image, Image.Image):
            return np.array(image.convert("RGB"))
        return image

    def extract_embedding(
        self,
        image: Union[np.ndarray, Image.Image, List[Union[np.ndarray, Image.Image]]]
    ) -> torch.Tensor:
        """
        Supports single image or batch of images.

        Returns:
            torch.Tensor of shape:
              - (D,) for single image
              - (B, D) for batch
        """

        is_batch = isinstance(image, list)

        images = image if is_batch else [image]
        embeddings = []

        for img in images:
            img = self._prepare_image(img)

            try:
                rep = DeepFace.represent(
                    img_path=img,
                    model_name=self.model_name,
                    enforce_detection=True,
                    detector_backend="mtcnn",
                )
            except ValueError:
                raise ValueError("No face detected")

            if not rep:
                raise ValueError("No face detected")

            embeddings.append(rep[0]["embedding"])

        emb_tensor = torch.tensor(embeddings, dtype=torch.float32)
        if self.device == "cuda":
            emb_tensor = emb_tensor.cuda()

        return emb_tensor[0] if not is_batch else emb_tensor
