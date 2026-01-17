import os
from PIL import Image
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from model.networks.arcface_model import ArcFaceExtractor
from model.networks.face_matcher import FaceMatcher

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_image(image_path):
    """Load image in RGB format."""
    img = Image.open(image_path).convert("RGB")
    return img


def evaluate_face_matching(
    pairs_csv: str,
    images_dir: str,
    threshold: float = 0.5,
    batch_size: int = 16,
):
    df = pd.read_csv(pairs_csv) if isinstance(pairs_csv, str) else pairs_csv

    extractor = ArcFaceExtractor(device=DEVICE)
    matcher = FaceMatcher(threshold)

    y_true, y_pred = [], []
    similarities_match, similarities_non_match = [], []

    for i in tqdm(range(0, len(df), batch_size), desc="Evaluating Face Matching"):
        batch = df.iloc[i:i + batch_size]

        imgs1 = [load_image(os.path.join(images_dir, r.image_name1)) for r in batch.itertuples()]
        imgs2 = [load_image(os.path.join(images_dir, r.image_name2)) for r in batch.itertuples()]

        emb1 = extractor.extract_embedding(imgs1)
        emb2 = extractor.extract_embedding(imgs2)

        preds, sims = matcher.match(emb1, emb2)

        for gt, p, s in zip(batch["match"], preds, sims):
            y_true.append(int(gt))
            y_pred.append(int(p))
            (similarities_match if gt else similarities_non_match).append(s)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "avg_similarity_match": np.mean(similarities_match),
        "avg_similarity_non_match": np.mean(similarities_non_match),
    }
