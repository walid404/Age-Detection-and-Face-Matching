import torch
from typing import List, Union

from model.networks.arcface_model import ArcFaceExtractor
from model.networks.face_matcher import FaceMatcher

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def match_faces(
    img1: Union[object, List[object]],
    img2: Union[object, List[object]],
    threshold=0.5,
):
    """
    Supports:
      - single image pair
      - batch of image pairs
    """

    extractor = ArcFaceExtractor(device=DEVICE)
    matcher = FaceMatcher(threshold)

    emb1 = extractor.extract_embedding(img1)
    emb2 = extractor.extract_embedding(img2)

    match, similarity = matcher.match(emb1, emb2)

    return {
        "match": match,
        "similarity": similarity,
    }
