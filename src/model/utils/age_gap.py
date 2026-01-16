import numpy as np


def compute_cross_age_gaps(dataset, indices):
    """
    Computes age gaps (max_age - min_age) per person.
    """
    df = dataset.data.iloc[indices]

    gaps = []
    for pid, group in df.groupby("person_id"):
        if len(group) > 1:
            gap = group["age"].max() - group["age"].min()
            gaps.append(gap)

    return np.array(gaps)
