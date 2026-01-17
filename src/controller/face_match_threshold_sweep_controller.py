import pandas as pd
from tqdm import tqdm

from src.controller.face_match_evaluation_controller import evaluate_face_matching


def sweep_face_matching_thresholds(
    pairs_csv: str,
    images_dir: str,
    thresholds: list =[0.3, 0.4, 0.5, 0.6, 0.7],
):
    """
    Evaluate ArcFace face matching over multiple thresholds.

    Returns:
        pd.DataFrame with metrics per threshold
    """

    results = []

    progress_bar = tqdm(
        thresholds,
        desc="Threshold Sweep",
        dynamic_ncols=True,
    )

    for threshold in progress_bar:
        metrics = evaluate_face_matching(
            pairs_csv=pairs_csv,
            images_dir=images_dir,
            threshold=threshold,
        )

        row = {
            "threshold": threshold,
            "accuracy": round(metrics["accuracy"], 4),
            "precision": round(metrics["precision"], 4),
            "recall": round(metrics["recall"], 4),
            "f1": round(metrics["f1"], 4),
            "avg_similarity_match": round(metrics["avg_similarity_match"], 4),
            "avg_similarity_non_match": round(metrics["avg_similarity_non_match"], 4),
        }

        results.append(row)

        progress_bar.set_postfix({
            "acc": f"{row['accuracy']:.2f}",
            "f1": f"{row['f1']:.2f}",
        })

    return pd.DataFrame(results)
