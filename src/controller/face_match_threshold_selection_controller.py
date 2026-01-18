import pandas as pd
from tqdm import tqdm

from src.controller.face_match_evaluation_controller import evaluate_face_matching


def select_threshold_on_train(
    train_pairs_csv: str,
    images_dir: str,
    thresholds,
    optimize_metric: str = "f1",
):
    """
    Select best threshold using training pairs only.
    """

    results = []

    progress_bar = tqdm(
        thresholds,
        desc="Selecting Threshold (Train)",
        dynamic_ncols=True,
    )

    for threshold in progress_bar:
        metrics = evaluate_face_matching(
            pairs_csv=train_pairs_csv,
            images_dir=images_dir,
            threshold=threshold,
        )

        results.append({
            "threshold": threshold,
            **{k: round(v, 4) for k, v in metrics.items()}
        })

        progress_bar.set_postfix({
            optimize_metric: f"{metrics[optimize_metric]:.2f}"
        })

    df = pd.DataFrame(results)

    best_row = df.loc[df[optimize_metric].idxmax()]
    best_threshold = best_row["threshold"]

    return best_threshold, df


def evaluate_chosen_threshold_on_test(
    test_pairs_csv: str,
    images_dir: str,
    threshold: float,
):
    """
    Evaluate chosen threshold on test dataset.
    """

    metrics = evaluate_face_matching(
        pairs_csv=test_pairs_csv,
        images_dir=images_dir,
        threshold=threshold,
    )

    return pd.DataFrame([{
        "threshold": threshold,
        **{k: round(v, 4) for k, v in metrics.items()}
    }])
