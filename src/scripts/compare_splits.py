import os
import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import argparse


def compare_split_performance(plots_dir: str = "src/eda/eda_plots", experiment_name: str = "Age_Prediction_Split_Comparison"):
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    runs = client.search_runs(exp.experiment_id)

    records = []
    for run in runs:
        if "test_mae" in run.data.metrics:
            records.append({
                "split_strategy": run.data.params["split_strategy"],
                "test_mae": run.data.metrics["test_mae"]
            })

    df = pd.DataFrame(records)

    plt.figure()
    df.boxplot(by="split_strategy", column="test_mae")
    plt.title("Random vs Identity Split – Test MAE")
    plt.suptitle("")
    plt.ylabel("MAE")
    plt.savefig(os.path.join(plots_dir, "split_comparison.png"))
    plt.close()

def parse_args():
    parser = argparse.ArgumentParser(description="Compare Split Strategies Performance")
    parser.add_argument(
        "--plots_dir",
        type=str,
        default="src/report/plots",
        help="Directory to save the comparison plot"
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="Age_Prediction_Split_Comparison",
        help="MLflow experiment name containing the runs to compare"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    compare_split_performance(
        args.save_path,
        args.experiment_name
    )
