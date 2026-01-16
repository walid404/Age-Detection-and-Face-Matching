import os
from model.datasets.data_preparation import (
    download_fgnet,
    extract_fgnet,
    generate_labels_csv
)

DATASET_ROOT = "Dataset/FGNET"
IMAGES_DIR = os.path.join(DATASET_ROOT, "images")
LABELS_CSV = os.path.join(DATASET_ROOT, "labels.csv")


def prepare_dataset():
    if os.path.exists(IMAGES_DIR) and os.path.exists(LABELS_CSV):
        print("Dataset already prepared.")
        return

    zip_path = download_fgnet(DATASET_ROOT)
    extract_fgnet(zip_path, DATASET_ROOT)
    generate_labels_csv(IMAGES_DIR, LABELS_CSV)


if __name__ == "__main__":
    prepare_dataset()
