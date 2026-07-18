"""
Promote the latest version of the registered model to Production
if the weighted F1 score from the most recent evaluation run exceeds
the required threshold.

Run:
    python scripts/promote.py
"""

import os

import yaml
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

load_dotenv()


def load_params() -> dict:
    with open("params.yaml") as f:
        return yaml.safe_load(f)["promote"]


def get_latest_f1(client: MlflowClient, experiment_name: str) -> tuple[str, float]:
    """
    Return the run_id and f1_weighted of the most recent evaluation run.
    """
    experiment = client.get_experiment_by_name(experiment_name)
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )
    latest_run = runs[0]
    return latest_run.info.run_id, latest_run.data.metrics["f1_weighted"]


def promote(client: MlflowClient, model_name: str):
    # get the latest version of the model
    versions = client.search_model_versions(f"name='{model_name}'")
    latest_version = max(versions, key=lambda v: int(v.version))
    client.set_registered_model_alias(model_name, "production", latest_version.version)
    print(f"Promoted {model_name} version {latest_version.version} to 'production'.")


def main():
    threshold = load_params()["f1_threshold"]
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "finbert-sentiment")
    model_name = os.getenv("MODEL_NAME", "finbert")

    client = MlflowClient(tracking_uri=tracking_uri)

    run_id, f1_weighted = get_latest_f1(client, experiment_name)
    print(f"Latest evaluation run {run_id}: f1_weighted = {f1_weighted:.4f} (threshold = {threshold})")

    if f1_weighted >= threshold:
        promote(client, model_name)
    else:
        print(
            f"Model did not meet the F1 threshold "
            f"({f1_weighted:.4f} < {threshold}) — not promoting to production."
        )


if __name__ == "__main__":
    main()
