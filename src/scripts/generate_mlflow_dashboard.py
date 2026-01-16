import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import os

EXPERIMENT_NAME = "Age_Prediction_Full_Comparison"
OUT_DIR = "eda/eda_plots"
TABLE_DIR = "reports"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)


def generate_dashboard():
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
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
            "epochs_run": int(p.get("epochs_run")),
            "stop_epoch": int(p.get("stop_epoch")),
            "training_time_min": m.get("training_time_min"),

            "test_mae": m.get("test_mae"),
            "test_mse": m.get("test_mse"),
            "test_rmse": m.get("test_rmse"),
            "test_r2": m.get("test_r2"),
        })

    df = pd.DataFrame(records)

    # --------------------------------------------------
    # Round numeric values to 2 decimals
    # --------------------------------------------------
    numeric_cols = df.select_dtypes(include=["float", "int"]).columns
    df[numeric_cols] = df[numeric_cols].round(2)

    # --------------------------------------------------
    # Save comparison table
    # --------------------------------------------------
    table_path = os.path.join(TABLE_DIR, "model_comparison_with_time.csv")
    df.sort_values(by="test_mae").to_csv(table_path, index=False)

    print("\n=== Model Comparison Table ===")
    print(df.sort_values(by="test_mae"))

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

    plot_path = os.path.join(OUT_DIR, "accuracy_vs_training_time.png")
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
    stop_epoch_plot = os.path.join(OUT_DIR, "stop_epoch_distribution.png")
    plt.savefig(stop_epoch_plot)
    plt.close()

    # --------------------------------------------------
    # Log artifacts to MLflow
    # --------------------------------------------------
    with mlflow.start_run(run_name="dashboard_summary"):
        mlflow.log_artifact(table_path)
        mlflow.log_artifact(plot_path)
        mlflow.log_artifact(stop_epoch_plot)


if __name__ == "__main__":
    generate_dashboard()
