from src.model.utils.load import load_model
from src.model.networks.face_matcher import FaceMatcher
from src.model.networks.arcface_model import ArcFaceExtractor
from functools import lru_cache
import torch


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache
def get_age_model(model_name: str, checkpoint_path: str):
    """
    Load and cache the age model.

    Parameters
    ----------
    model_name : str
        Name of the age model architecture.
    checkpoint_path : str
        Path to the model weights checkpoint.

    Returns
    -------
    torch.nn.Module
        Loaded age model.
    """
    model = load_model(model_name, checkpoint_path)
    return model

@lru_cache
def get_face_matching_model(threshold: float = 0.25):
    """
    Load and cache the face matching model.

    Parameters
    ----------
    threshold : float
        Similarity threshold for face matching.

    Returns
    -------
    tuple
        (ArcFaceExtractor, FaceMatcher)
    """
    extractor = ArcFaceExtractor(device=DEVICE)
    matcher = FaceMatcher(threshold)
    return extractor, matcher

