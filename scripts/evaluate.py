"""
Evaluate sentiment model on the test set and register it
in the MLflow Model Registry.

Run:
    python scripts/evaluate.py
"""

# `transformers` (torch) must be the first heavy import in this file — on this machine,
# importing numpy-based libs (pandas/sklearn/mlflow) before torch loads causes a native
# DLL init conflict when torch initializes afterward.
from transformers import pipeline

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import mlflow
import mlflow.transformers
import pandas as pd
from app.utils import _patch_model_config
from dotenv import load_dotenv
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

load_dotenv()


def load_test_data(test_path: str) -> tuple[list[str], list[str]]:
    df = pd.read_csv(test_path)
    return df["text"].tolist(), df["label"].tolist()


def build_classifier(model_id: str):
    print(f"Loading {model_id}...")
    _patch_model_config(model_id)
    return pipeline(
        "text-classification",
        model=model_id,
        tokenizer=model_id,
        truncation=True,
        max_length=512,
    )


def run_inference(classifier, texts: list[str]) -> list[str]:
    print(f"Running inference on {len(texts)} samples...")
    results = classifier(texts, batch_size=32)
    return [r["label"] for r in results]


def compute_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted"),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted"),
    }


def main():
    model_id = os.getenv("HF_MODEL_ID", "baptle/FinBERT_market_based")
    test_path = os.path.join("data", "test.csv")

    texts, y_true = load_test_data(test_path)
    classifier = build_classifier(model_id)
    y_pred = run_inference(classifier, texts)
    metrics = compute_metrics(y_true, y_pred)

    print("\nEvaluation Results:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "finbert-sentiment"))
    model_name = os.getenv("MODEL_NAME", "finbert")

    with mlflow.start_run():
        mlflow.log_param("model_id", model_id)
        mlflow.log_param("num_test_samples", len(texts))
        mlflow.log_metrics(metrics)
        # pip_requirements is passed explicitly to skip mlflow's automatic engine-type
        # inference, which errors on this transformers version (looks up a removed
        # `FlaxPreTrainedModel` symbol) and then crashes trying to version-check tensorflow,
        # which isn't installed since this project is torch-only.
        mlflow.transformers.log_model(
            transformers_model=classifier,
            artifact_path="model",
            registered_model_name=model_name,
            pip_requirements=["transformers", "torch"],
        )


if __name__ == "__main__":
    main()
