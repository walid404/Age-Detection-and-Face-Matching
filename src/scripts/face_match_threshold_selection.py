import argparse
import os

from src.controller.face_match_threshold_selection_controller import (
    select_threshold_on_train,
    evaluate_chosen_threshold_on_test,
)
from src.model.datasets.data_preparation import generate_face_matching_pairs
from src.scripts.prepare_data import prepare_dataset



def parse_args():
    parser = argparse.ArgumentParser(
        description="ArcFace Threshold Selection & Evaluation"
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default='src/Dataset',
        help="Directory containing Datasets",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="FGNET",
        help="Directory of the FG-NET Dataset",
    )
    parser.add_argument(
        "--labels_csv_name",
        type=str,
        default="labels.csv",
        help="Name of the labels CSV file",
    )
    parser.add_argument(
        "--images_dir_name",
        type=str,
        default="images",
        help="Name of the images directory",
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
        type=bool,
        default=True,
        help="Save results to CSV files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="reports/tables",
        help="Directory to save results (if enabled)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # --------------------------------------------------
    # Threshold selection on TRAIN
    # --------------------------------------------------
    prepare_dataset(args.dataset_root, args.dataset_name,
                    args.images_dir_name, args.labels_csv_name)
    labels_csv = os.path.join(args.Dataset_dir, "labels.csv")

    train_pairs, _ = generate_face_matching_pairs(
        input_csv=labels_csv,
        output_csv=os.path.join(args.Dataset_dir, "train_pairs.csv"),
        split="train"
    )

    test_pairs, _ = generate_face_matching_pairs(
        input_csv=labels_csv,
        output_csv=os.path.join(args.Dataset_dir, "test_pairs.csv"),
        split="test"
    )

    best_threshold, train_table = select_threshold_on_train(
        train_pairs_csv=train_pairs,
        images_dir=os.path.join(args.Dataset_dir, "images"),
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
        images_dir=os.path.join(args.Dataset_dir, "images"),
        threshold=best_threshold,
    )

    print("\n========== TEST EVALUATION ==========")
    print(test_table.to_string(index=False))

    # --------------------------------------------------
    # Save results (optional)
    # --------------------------------------------------
    if args.save_results:
        os.makedirs(args.output_dir, exist_ok=True)

        train_path = os.path.join(args.output_dir, "threshold_selection_train.csv")
        test_path = os.path.join(args.output_dir, "final_test_evaluation.csv")

        train_table.to_csv(train_path, index=False)
        test_table.to_csv(test_path, index=False)

        print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
