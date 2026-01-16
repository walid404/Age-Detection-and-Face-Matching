import os
import zipfile
import requests
import pandas as pd
from tqdm import tqdm

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
