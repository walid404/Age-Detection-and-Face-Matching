import torch
from deepface import DeepFace
import numpy as np
from PIL import Image

class ArcFaceExtractor:
    def __init__(self, model_name: str = "ArcFace", device: str = "cpu"):
        """
        Initialize the ArcFace extractor using DeepFace.

        Parameters
        ----------
        model_name : str
            Name of the face recognition model to use (default "ArcFace").
        device : str
            Device to use: "cpu" or "cuda" (default "cpu").
        """
        self.model_name = model_name
        self.device = "cuda" if device == "cuda" else "cpu"

    def extract_embedding(self, image) -> torch.Tensor:
        """
        Extract face embedding from an image using DeepFace.

        Parameters
        ----------
        image : np.ndarray or PIL.Image.Image
            Input image in RGB format.

        Returns
        -------
        torch.Tensor
            Embedding vector of the first detected face.
        """
        # Convert PIL image to numpy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image.convert("RGB"))

        try:
            # remove 'model' argument
            embedding = DeepFace.represent(
                img_path=image,
                model_name=self.model_name,
                enforce_detection=True,  # keep detection enforced
                detector_backend='mtcnn' # optional, can also be 'opencv', 'ssd', etc.
            )
        except ValueError:
            raise ValueError("No face detected")

        # DeepFace returns a list of dicts; take the first embedding
        if isinstance(embedding, list) and len(embedding) > 0:
            emb_vector = embedding[0]["embedding"]
        else:
            raise ValueError("No face detected")

        # Convert to torch.Tensor
        emb_tensor = torch.tensor(emb_vector, dtype=torch.float32)
        if self.device == "cuda":
            emb_tensor = emb_tensor.cuda()

        return emb_tensor
