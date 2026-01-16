import numpy as np
from torch.utils.data import Subset, random_split
from collections import defaultdict
import torch
import pandas as pd


def random_dataset_split(dataset, train_ratio, val_ratio, seed=42):
    """
    Simple random image-level split (may cause identity leakage).
    """
    length = len(dataset)
    train_size = int(train_ratio * length)
    val_size = int(val_ratio * length)
    test_size = length - train_size - val_size

    generator = np.random.default_rng(seed)
    return random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )


def identity_aware_split(
    dataset,
    train_ratio: float,
    val_ratio: float,
    seed: int = 42
):
    """
    Identity-aware split based on image counts, not person counts.
    Each person_id appears in exactly one split.
    """
    df = dataset.data
    rng = np.random.default_rng(seed)

    # Group indices by person_id
    person_to_indices = defaultdict(list)
    for idx, row in df.iterrows():
        person_to_indices[row["person_id"]].append(idx)

    persons = list(person_to_indices.keys())
    rng.shuffle(persons)

    total_images = len(df)
    train_target = int(train_ratio * total_images)
    val_target = int(val_ratio * total_images)

    train_idx, val_idx, test_idx = [], [], []
    train_count = val_count = 0

    for pid in persons:
        indices = person_to_indices[pid]
        n_imgs = len(indices)

        if train_count + n_imgs <= train_target:
            train_idx.extend(indices)
            train_count += n_imgs

        elif val_count + n_imgs <= val_target:
            val_idx.extend(indices)
            val_count += n_imgs

        else:
            test_idx.extend(indices)

    return (
        Subset(dataset, train_idx),
        Subset(dataset, val_idx),
        Subset(dataset, test_idx),
    )


def check_identity_leakage(train_set, val_set, test_set):
    def get_ids(subset):
        return set(subset.dataset.data.iloc[subset.indices]["person_id"])

    assert get_ids(train_set).isdisjoint(get_ids(val_set))
    assert get_ids(train_set).isdisjoint(get_ids(test_set))
    assert get_ids(val_set).isdisjoint(get_ids(test_set))



def identity_aware_dataframe_split(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    seed: int = 42,
):
    """
    Identity-aware split on DataFrame level (image-count based).
    Each person_id appears in exactly one split.
    """

    rng = np.random.default_rng(seed)

    person_to_indices = defaultdict(list)
    for idx, row in df.iterrows():
        person_to_indices[row["person_id"]].append(idx)

    persons = list(person_to_indices.keys())
    rng.shuffle(persons)

    total_images = len(df)
    train_target = int(train_ratio * total_images)
    val_target = int(val_ratio * total_images)

    train_idx, val_idx, test_idx = [], [], []
    train_count = val_count = 0

    for pid in persons:
        indices = person_to_indices[pid]
        n_imgs = len(indices)

        if train_count < train_target:
            train_idx.extend(indices)
            train_count += n_imgs

        elif val_count < val_target:
            val_idx.extend(indices)
            val_count += n_imgs

        else:
            test_idx.extend(indices)

    return (
        df.loc[train_idx].reset_index(drop=True),
        df.loc[val_idx].reset_index(drop=True),
        df.loc[test_idx].reset_index(drop=True),
    )
