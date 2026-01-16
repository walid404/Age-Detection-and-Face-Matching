import mlflow
import pandas as pd
import os

EXPERIMENT_NAME = "Age_Prediction_Full_Comparison"
OUT_DIR = "reports"
os.makedirs(OUT_DIR, exist_ok=True)


def generate_tables():
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
            "batch_size": p.get("batch_size"),
            "lr": p.get("learning_rate"),
            "epochs": p.get("epochs"),
            "test_mae": m.get("test_mae"),
            "test_mse": m.get("test_mse"),
            "test_rmse": m.get("test_rmse"),
            "test_r2": m.get("test_r2"),
        })

    df = pd.DataFrame(records)
    df = df.sort_values(by="test_mae")

    csv_path = os.path.join(OUT_DIR, "model_comparison.csv")
    df.to_csv(csv_path, index=False)

    print(df)

    with mlflow.start_run(run_name="comparison_tables"):
        mlflow.log_artifact(csv_path)


if __name__ == "__main__":
    generate_tables()
