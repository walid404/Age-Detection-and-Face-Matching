from model.datasets.identity_split import (
    identity_aware_split,
    random_dataset_split,
    check_identity_leakage
)


def split_dataset(dataset, config):
    strategy = config["dataset"]["split_strategy"]

    train_ratio = config["dataset"]["train_split"]
    val_ratio = config["dataset"]["val_split"]

    if strategy == "identity":
        train_set, val_set, test_set = identity_aware_split(
            dataset, train_ratio, val_ratio
        )
        check_identity_leakage(train_set, val_set, test_set)

    elif strategy == "random":
        train_set, val_set, test_set = random_dataset_split(
            dataset, train_ratio, val_ratio
        )

    else:
        raise ValueError(
            f"Unknown split strategy: {strategy}. "
            "Use 'identity' or 'random'."
        )

    return train_set, val_set, test_set
