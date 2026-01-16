import os
import time
import mlflow
import torch
from itertools import product
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from model.datasets.age_dataset import AgeDataset
from model.datasets.split_factory import split_dataset
from model.networks.age_models import get_age_model
from model.training.trainer import train_epoch, evaluate_full
from model.training.losses import get_loss
from model.training.early_stopping import EarlyStopping
from scripts.prepare_data import prepare_dataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def run_training(config):
    """
    Full experimental pipeline with:
    - MSE loss
    - Random & identity-aware splits (computed once)
    - Early stopping with stop epoch tracking
    - Training time logging
    - Clean tqdm progress bars (no line spam)
    """

    # --------------------------------------------------
    # Ensure dataset exists
    # --------------------------------------------------
    prepare_dataset()

    # --------------------------------------------------
    # MLflow setup
    # --------------------------------------------------
    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    model_dir = config["mlflow"]["model_dir"]
    os.makedirs(model_dir, exist_ok=True)

    img_size = config["dataset"]["img_size"]

    # --------------------------------------------------
    # Transforms (ToTensor -> [0,1])
    # --------------------------------------------------
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    # --------------------------------------------------
    # Dataset instances
    # --------------------------------------------------
    train_dataset = AgeDataset(
        config["dataset"]["labels_csv"],
        config["dataset"]["images_dir"],
        transform=train_transform,
    )

    eval_dataset = AgeDataset(
        config["dataset"]["labels_csv"],
        config["dataset"]["images_dir"],
        transform=eval_transform,
    )

    # --------------------------------------------------
    # Split once (random & identity)
    # --------------------------------------------------
    config["dataset"]["split_strategy"] = "random"
    r_train, r_val, r_test = split_dataset(train_dataset, config)

    config["dataset"]["split_strategy"] = "identity"
    i_train, i_val, i_test = split_dataset(train_dataset, config)

    split_collections = {
        "random": {
            "train": r_train,
            "val": Subset(eval_dataset, r_val.indices),
            "test": Subset(eval_dataset, r_test.indices),
        },
        "identity": {
            "train": i_train,
            "val": Subset(eval_dataset, i_val.indices),
            "test": Subset(eval_dataset, i_test.indices),
        },
    }

    # --------------------------------------------------
    # Experiment grid
    # --------------------------------------------------
    experiment_grid = list(product(
        split_collections.items(),
        config["models"]["names"],
        config["training"]["batch_size"],
        config["training"]["learning_rate"],
        config["training"]["epochs"],
    ))

    # --------------------------------------------------
    # Global tqdm (ALL experiments)
    # --------------------------------------------------
    experiments_bar = tqdm(
        experiment_grid,
        desc="Running Experiments",
        total=len(experiment_grid),
        dynamic_ncols=True,
    )

    for (split_name, split_sets), model_name, batch_size, lr, max_epochs in experiments_bar:

        run_name = f"{model_name}_{split_name}_bs{batch_size}_lr{lr}_ep{max_epochs}"

        with mlflow.start_run(run_name=run_name):

            mlflow.log_params({
                "model": model_name,
                "split_strategy": split_name,
                "batch_size": batch_size,
                "learning_rate": lr,
                "max_epochs": max_epochs,
                "loss": "mse",
            })

            train_loader = DataLoader(
                split_sets["train"], batch_size=batch_size, shuffle=True
            )
            val_loader = DataLoader(
                split_sets["val"], batch_size=batch_size, shuffle=False
            )
            test_loader = DataLoader(
                split_sets["test"], batch_size=batch_size, shuffle=False
            )

            model = get_age_model(model_name).to(DEVICE)
            criterion = get_loss(config["training"]["loss"])
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            scheduler = ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=3
            )
            early_stopper = EarlyStopping(patience=config["training"]["patience"])

            # --------------------------------------------------
            # Start timer
            # --------------------------------------------------
            start_time = time.time()

            stop_epoch = max_epochs - 1
            early_stopped = False

            # --------------------------------------------------
            # Epoch tqdm (PER MODEL)
            # --------------------------------------------------
            epoch_bar = tqdm(
                range(max_epochs),
                desc=run_name,
                leave=False,
                dynamic_ncols=True,
            )

            for epoch in epoch_bar:

                train_loss = train_epoch(
                    model, train_loader, optimizer, criterion
                )

                val_metrics = evaluate_full(model, val_loader)
                val_loss = val_metrics[config["training"]["loss"]]

                epoch_bar.set_postfix({
                    "train_loss": f"{train_loss:.2f}",
                    "val_loss": f"{val_loss:.2f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.1e}",
                })

                mlflow.log_metric("train_loss", train_loss, step=epoch)
                mlflow.log_metric("val_loss", val_loss, step=epoch)

                scheduler.step(val_loss)
                early_stopper.step(val_loss, epoch)

                if early_stopper.stop:
                    stop_epoch = epoch
                    early_stopped = True
                    break

            # --------------------------------------------------
            # End timer
            # --------------------------------------------------
            training_time_sec = time.time() - start_time
            training_time_min = training_time_sec / 60.0
            epochs_run = stop_epoch + 1

            # --------------------------------------------------
            # Log stopping & timing info
            # --------------------------------------------------
            mlflow.log_metrics({
                "training_time_sec": training_time_sec,
                "training_time_min": training_time_min,
                "best_val_loss": early_stopper.best_loss,
            })

            mlflow.log_params({
                "epochs_run": epochs_run,
                "stop_epoch": stop_epoch,
                "early_stopped": early_stopped,
                "best_epoch": early_stopper.best_epoch,
            })

            # --------------------------------------------------
            # Test evaluation
            # --------------------------------------------------
            test_metrics = evaluate_full(model, test_loader)
            for k, v in test_metrics.items():
                mlflow.log_metric(f"test_{k}", v)

            # --------------------------------------------------
            # Save model
            # --------------------------------------------------
            model_path = os.path.join(model_dir, f"{run_name}.pt")
            torch.save(model.state_dict(), model_path)
            # mlflow.log_artifact(model_path)
            # mlflow.pytorch.log_model(model, artifact_path="model")

            # --------------------------------------------------
            # Update global tqdm (IN-PLACE)
            # --------------------------------------------------
            experiments_bar.set_postfix({
                "model": model_name,
                "split": split_name,
                "best_val_loss": f"{early_stopper.best_loss:.2f}",
                "epochs": epochs_run,
                "time_min": f"{training_time_min:.2f}",
            })
