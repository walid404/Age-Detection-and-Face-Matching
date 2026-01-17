import torch

from src.model.networks.arcface_model import ArcFaceExtractor
from src.model.networks.face_matcher import FaceMatcher

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def match_faces(img1, img2, threshold=0.5):
    
    extractor = ArcFaceExtractor(device=DEVICE)
    matcher = FaceMatcher(threshold)

    emb1 = extractor.extract_embedding(img1)
    emb2 = extractor.extract_embedding(img2)

    match, similarity = matcher.match(emb1, emb2)

    return {
        "match": match,          # 1 = same person, 0 = different
        "similarity": similarity
    }
