import os
import argparse
from src.model.datasets.data_preparation import (
    download_fgnet,
    extract_fgnet,
    generate_labels_csv
)


def prepare_dataset(dataset_root: str, dataset_name: str, images_dir_name: str, labels_csv_name: str):
    images_dir = os.path.join(dataset_root, dataset_name, images_dir_name)
    labels_csv = os.path.join(dataset_root, dataset_name, labels_csv_name)
    if os.path.exists(images_dir) and os.path.exists(labels_csv):
        print("Dataset already prepared.")
        return

    zip_path = download_fgnet(dataset_root)
    extract_fgnet(zip_path, dataset_root)
    generate_labels_csv(images_dir, labels_csv)


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare FG-NET Dataset")
    parser.add_argument(
        "--dataset_root", 
        type=str, 
        default="src/Dataset", 
        help="Root directory for the dataset"
    )
    parser.add_argument(
        "--dataset_name", 
        type=str, 
        default="FGNET", 
        help="Name of the dataset folder"
    )
    parser.add_argument(
        "--images_dir_name", 
        type=str, 
        default="images", 
        help="Name of the images directory"
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
    prepare_dataset(
        args.dataset_root, 
        args.dataset_name, 
        args.images_dir_name, 
        args.labels_csv_name
    )
