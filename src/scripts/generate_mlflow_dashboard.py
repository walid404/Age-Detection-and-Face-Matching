import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Generate MLflow Dashboard for Experiment")
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="Age_Prediction_Full_Comparison",
        help="MLflow experiment name containing the runs to analyze"
    )
    parser.add_argument(
        "--plots_dir",
        type=str,
        default="src/reports/plots",
        help="Directory to save the generated plots"
    )
    parser.add_argument(
        "--tables_dir",
        type=str,
        default="src/reports/tables",
        help="Directory to save the generated tables"
    )
    parser.add_argument(
        "--sort_by_metric",
        type=str,
        default="test_mse",
        help="Metric to sort the comparison table by"
    )
    return parser.parse_args()


def generate_dashboard(experiment_name: str = "Age_Prediction_Full_Comparison",
                       plots_dir: str = "src/reports/plots",
                       tables_dir: str = "src/reports/tables",
                       sort_by_metric: str = "test_mse"):
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    runs = client.search_runs(experiment.experiment_id)

    records = []

    for run in runs:
        p = run.data.params
        m = run.data.metrics

        records.append({
            "model": p.get("model"),
            "split": p.get("split_strategy"),
            "batch_size": int(p.get("batch_size")),
            "lr": float(p.get("learning_rate")),
            "stop_epoch": int(p.get("stop_epoch")),
            "training_time_min": m.get("training_time_min"),

            "test_mae": m.get("test_mae"),
            "test_mse": m.get("test_mse"),
            "test_rmse": m.get("test_rmse"),
            "test_r2": m.get("test_r2"),
        })

    df = pd.DataFrame(records)
    df = df.sort_values(by=sort_by_metric)
    # --------------------------------------------------
    # Round numeric values to 2 decimals
    # --------------------------------------------------
    numeric_cols = df.select_dtypes(include=["float", "int"]).columns
    df[numeric_cols] = df[numeric_cols].round(2)

    # --------------------------------------------------
    # Save comparison table
    # --------------------------------------------------
    table_path = os.path.join(tables_dir, "model_comparison_with_time.csv")
    df.to_csv(table_path, index=False)

    print("\n=== Model Comparison Table ===")
    print(df)

    # --------------------------------------------------
    # Visualization: Accuracy vs Training Time
    # --------------------------------------------------
    plt.figure()
    for split in df["split"].unique():
        subset = df[df["split"] == split]
        plt.scatter(
            subset["training_time_min"],
            subset["test_mae"],
            label=split,
        )

    plt.xlabel("Training Time (min)")
    plt.ylabel("Test MAE")
    plt.title("Accuracy vs Training Time")
    plt.legend()

    plot_path = os.path.join(plots_dir, "accuracy_vs_training_time.png")
    plt.savefig(plot_path)
    plt.close()

    # --------------------------------------------------
    # Visualization: Stop Epoch Distribution
    # --------------------------------------------------
    plt.figure()
    df.boxplot(column="stop_epoch", by="split")
    plt.title("Stop Epoch Distribution")
    plt.suptitle("")
    plt.ylabel("Epoch")
    stop_epoch_plot = os.path.join(plots_dir, "stop_epoch_distribution.png")
    plt.savefig(stop_epoch_plot)
    plt.close()


if __name__ == "__main__":
    args = parse_args()
    generate_dashboard(
        args.experiment_name, 
        args.plots_dir, 
        args.tables_dir, 
        args.sort_by_metric
    )
