import mlflow
import pandas as pd
import matplotlib.pyplot as plt

EXPERIMENT_NAME = "Age_Prediction_Split_Comparison"


def compare_split_performance():
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
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
    plt.savefig("eda/eda_plots/split_comparison.png")
    plt.close()


if __name__ == "__main__":
    compare_split_performance()
