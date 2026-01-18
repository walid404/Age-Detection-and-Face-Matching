import yaml
from src.controller.train_controller import run_training
from src.scripts.generate_mlflow_dashboard import generate_dashboard
from src.scripts.compare_splits import compare_split_performance
from src.view.eda_plots import eda_plots
from src.scripts.prepare_data import prepare_dataset


def main():

    with open("src/config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    prepare_dataset(
        config["dataset"]["dataset_root"],
        config["dataset"]["dataset_name"],
        config["dataset"]["images_dir_name"],
        config["dataset"]["labels_csv_name"]
    )
    eda_plots(
        config["reports"]["plots_dir"],
        config["dataset"]["dataset_root"],
        config["dataset"]["dataset_name"],
        config["dataset"]["labels_csv_name"]
    )
    run_training(config)
    compare_split_performance(
        plots_dir=config["reports"]["plots_dir"],
        experiment_name=config["mlflow"]["experiment_name"]
    )
    generate_dashboard(
        config["mlflow"]["experiment_name"],
        config["reports"]["plots_dir"],
        config["reports"]["tables_dir"],
        config["mlflow"]["sort_by_metric"])


if __name__ == "__main__":
    main()
