import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from src.model.utils.load import load_image


class AgeDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        images_dir: str,
        transform=None,
        return_person_id: bool = False
    ):
        self.data = pd.read_csv(csv_path)
        self.images_dir = images_dir
        self.transform = transform
        self.return_person_id = return_person_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        img_path = os.path.join(self.images_dir, row["image_name"])
        image = load_image(img_path)

        if self.transform:
            image = self.transform(image)

        age = torch.tensor(row["age"], dtype=torch.float32)

        if self.return_person_id:
            person_id = torch.tensor(row["person_id"], dtype=torch.long)
            return image, age, person_id

        return image, age

