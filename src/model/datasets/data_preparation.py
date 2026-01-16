import os
import pandas as pd
import random
from typing import Tuple
import zipfile
import requests
import pandas as pd
from tqdm import tqdm
import random
from itertools import combinations
from model.datasets.identity_split import identity_aware_dataframe_split

FGNET_URL = "http://yanweifu.github.io/FG_NET_data/FGNET.zip"


def download_fgnet(dataset_root: str):
    os.makedirs(dataset_root, exist_ok=True)
    zip_path = os.path.join('Dataset', "FGNET.zip")

    if os.path.exists(zip_path):
        return zip_path

    print("Downloading FG-NET dataset...")
    response = requests.get(FGNET_URL, stream=True)
    total_size = int(response.headers.get("content-length", 0))

    with open(zip_path, "wb") as f, tqdm(
        desc="Downloading",
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=1024):
            f.write(chunk)
            bar.update(len(chunk))

    return zip_path


def extract_fgnet(zip_path: str, dataset_root: str):

    print("Extracting FG-NET dataset...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall('Dataset')

    print("Extraction completed.")


def parse_fgnet_filename(filename: str):
    """
    Parse FG-NET filename to extract person ID and age.

    Example:
        001A02.jpg -> person_id=1, age=2
        042A35.jpg -> person_id=42, age=35
    """
    name = filename.split(".")[0]
    person_part, age_part = name[:3], name[4:6]

    person_id = int(person_part)
    age = int(age_part)

    return person_id, age


def generate_labels_csv(images_dir: str, output_csv: str):
    records = []

    for img_name in sorted(os.listdir(images_dir)):
        if not img_name.lower().endswith(".jpg"):
            continue

        person_id, age = parse_fgnet_filename(img_name)

        records.append({
            "image_name": img_name,
            "person_id": person_id,
            "age": age
        })

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)

    print(f"labels.csv saved to {output_csv}")
    print(df.head())


def generate_face_matching_pairs(
    input_csv: str,
    output_csv: str,
    split: str = "train",
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    max_pairs: int = 10000,
    positive_ratio: float = 0.5,
    seed: int = 42,
) -> Tuple[pd.DataFrame, int]:
    """
    Generate identity-aware face matching pairs from dataset annotations.

    Parameters
    ----------
    input_csv : str
        CSV containing image_name, person_id, age
    output_csv : str
        Path to save generated pairs
    split : str
        One of: ['train', 'val', 'test']
    train_ratio : float
    val_ratio : float
    max_pairs : int
    positive_ratio : float
    seed : int

    Returns
    -------
    df_pairs : pd.DataFrame
    num_samples : int
    """

    random.seed(seed)

    df = pd.read_csv(input_csv)
    required_cols = {"image_name", "person_id", "age"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required_cols}")

    # --------------------------------------------------
    # Identity-aware split
    # --------------------------------------------------
    df_train, df_val, df_test = identity_aware_dataframe_split(
        df, train_ratio, val_ratio, seed
    )

    if split == "train":
        df_split = df_train
    elif split == "val":
        df_split = df_val
    elif split == "test":
        df_split = df_test
    else:
        raise ValueError("split must be one of ['train', 'val', 'test']")

    # --------------------------------------------------
    # Positive pairs (same identity)
    # --------------------------------------------------
    positive_pairs = []

    for person_id, group in df_split.groupby("person_id"):
        images = group[["image_name", "age"]].values.tolist()

        if len(images) < 2:
            continue

        for (img1, age1), (img2, age2) in combinations(images, 2):
            positive_pairs.append({
                "image_name1": img1,
                "image_name2": img2,
                "age1": age1,
                "age2": age2,
                "match": 1,
            })

    # --------------------------------------------------
    # Negative pairs (different identities)
    # --------------------------------------------------
    negative_pairs = []
    persons = df_split["person_id"].unique().tolist()

    while len(negative_pairs) < len(positive_pairs):
        p1, p2 = random.sample(persons, 2)

        img1 = df_split[df_split["person_id"] == p1].sample(1).iloc[0]
        img2 = df_split[df_split["person_id"] == p2].sample(1).iloc[0]

        negative_pairs.append({
            "image_name1": img1["image_name"],
            "image_name2": img2["image_name"],
            "age1": img1["age"],
            "age2": img2["age"],
            "match": 0,
        })

    # --------------------------------------------------
    # Balance & limit
    # --------------------------------------------------
    num_pos = int(max_pairs * positive_ratio)
    num_neg = max_pairs - num_pos

    random.shuffle(positive_pairs)
    random.shuffle(negative_pairs)

    pairs = positive_pairs[:num_pos] + negative_pairs[:num_neg]
    random.shuffle(pairs)

    # --------------------------------------------------
    # Save
    # --------------------------------------------------
    df_pairs = pd.DataFrame(pairs)
    df_pairs.to_csv(output_csv, index=False)

    return df_pairs, len(df_pairs)
