import argparse
import os

from controller.face_match_threshold_selection_controller import (
    select_threshold_on_train,
    evaluate_chosen_threshold_on_test,
)
from model.datasets.data_preparation import generate_face_matching_pairs
from scripts.prepare_data import prepare_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="ArcFace Threshold Selection & Evaluation"
    )

    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Root dataset directory (contains images/ and labels.csv)",
    )

    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6],
        help="List of thresholds to evaluate",
    )

    parser.add_argument(
        "--optimize_metric",
        type=str,
        default="f1",
        choices=["accuracy", "precision", "recall", "f1"],
        help="Metric used to select best threshold",
    )

    parser.add_argument(
        "--save_results",
        action="store_true",
        default=True,
        help="Save results to CSV files",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="reports",
        help="Directory to save results",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # --------------------------------------------------
    # Ensure dataset exists
    # --------------------------------------------------
    prepare_dataset()

    labels_csv = os.path.join(args.dataset_dir, "labels.csv")
    images_dir = os.path.join(args.dataset_dir, "images")

    # --------------------------------------------------
    # Generate face matching pairs (identity-aware)
    # --------------------------------------------------
    train_pairs, _ = generate_face_matching_pairs(
        input_csv=labels_csv,
        output_csv=os.path.join(args.dataset_dir, "train_pairs.csv"),
        split="train",
    )

    test_pairs, _ = generate_face_matching_pairs(
        input_csv=labels_csv,
        output_csv=os.path.join(args.dataset_dir, "test_pairs.csv"),
        split="test",
    )

    # --------------------------------------------------
    # Threshold selection on TRAIN
    # --------------------------------------------------
    best_threshold, train_table = select_threshold_on_train(
        train_pairs_csv=train_pairs,
        images_dir=images_dir,
        thresholds=args.thresholds,
        optimize_metric=args.optimize_metric,
    )

    print("\n========== TRAIN THRESHOLD SELECTION ==========")
    print(train_table.to_string(index=False))
    print(f"\nBest threshold ({args.optimize_metric}): {best_threshold}")

    # --------------------------------------------------
    # Final evaluation on TEST
    # --------------------------------------------------
    test_table = evaluate_chosen_threshold_on_test(
        test_pairs_csv=test_pairs,
        images_dir=images_dir,
        threshold=best_threshold,
    )

    print("\n========== TEST EVALUATION ==========")
    print(test_table.to_string(index=False))

    # --------------------------------------------------
    # Save results
    # --------------------------------------------------
    if args.save_results:
        os.makedirs(args.output_dir, exist_ok=True)

        train_path = os.path.join(
            args.output_dir, "threshold_selection_train.csv"
        )
        test_path = os.path.join(
            args.output_dir, "final_test_evaluation.csv"
        )

        train_table.to_csv(train_path, index=False)
        test_table.to_csv(test_path, index=False)

        print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
