from src.view.eda_plots import eda_plots
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run EDA on dataset")
    parser.add_argument(
        "--plot_dir", 
        type=str, 
        default="src/reports/plots",
        help="Directory to save the plots"
    )
    parser.add_argument(
        "--dataset_root", 
        type=str, 
        default="src/Dataset",
        help="Root directory of the dataset"
    )
    parser.add_argument(
        "--dataset_name", 
        type=str, 
        default="FGNET",
        help="Name of the dataset folder"
    )
    parser.add_argument(
        "--labels_csv_name", 
        type=str, 
        default="labels.csv",
        help="Name of the labels CSV file"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    eda_plots(
        plot_dir=args.plot_dir,
        dataset_root=args.dataset_root,
        dataset_name=args.dataset_name,
        labels_csv_name=args.labels_csv_name
    )