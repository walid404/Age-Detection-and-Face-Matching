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
):
    """
    Evaluate ArcFace-based face matching on a pair dataset.
    """

    df = pd.read_csv(pairs_csv) if isinstance(pairs_csv, str) else pairs_csv

    extractor = ArcFaceExtractor(device=DEVICE)
    matcher = FaceMatcher(threshold=threshold)

    y_true = []
    y_pred = []
    similarities_match = []
    similarities_non_match = []

    progress_bar = tqdm(
        df.iterrows(),
        total=len(df),
        desc="Evaluating Face Matching",
        dynamic_ncols=True,
    )

    for _, row in progress_bar:

        img1_path = os.path.join(images_dir, row["image_name1"])
        img2_path = os.path.join(images_dir, row["image_name2"])

        try:
            img1 = load_image(img1_path)
            img2 = load_image(img2_path)

            emb1 = extractor.extract_embedding(img1)
            emb2 = extractor.extract_embedding(img2)

            pred, similarity = matcher.match(emb1, emb2)

        except Exception as e:
            # Skip broken samples safely
            continue

        y_true.append(int(row["match"]))
        y_pred.append(int(pred))

        if row["match"]:
            similarities_match.append(similarity)
        else:
            similarities_non_match.append(similarity)

        progress_bar.set_postfix({
            "sim": f"{similarity:.2f}",
        })

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "avg_similarity_match": np.mean(similarities_match),
        "avg_similarity_non_match": np.mean(similarities_non_match),
    }

    return metrics
