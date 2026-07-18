"""
Model quality check (drift) using deepchecks

Run:
    python scripts/run_deepchecks.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# `app.utils` (transformers/torch) must be imported before pandas/deepchecks — on this
# machine, importing numpy-based libs first causes a native DLL init conflict with torch.
from app.utils import load_classifier

import pandas as pd
import yaml
from deepchecks.nlp import TextData
from deepchecks.nlp.checks import PredictionDrift, PropertyDrift
from dotenv import load_dotenv

load_dotenv()


def load_params() -> dict:
    with open("params.yaml") as f:
        return yaml.safe_load(f)["deepchecks"]


def run_predictions(classifier, texts: list[str]) -> list[str]:
    results = classifier(texts, batch_size=32)
    return [r["label"] for r in results]


def main():
    params = load_params()
    property_drift_threshold = params["property_drift_threshold"]
    prediction_drift_threshold = params["prediction_drift_threshold"]

    print("Loading production model...")

    classifier = load_classifier()

    stream_df = pd.read_csv("data/stream.csv")
    test_df = pd.read_csv("data/test.csv")

    stream_texts = stream_df["text"].tolist()
    test_texts = test_df["text"].tolist()

    test_dataset = TextData(raw_text=test_texts, task_type="text_classification")
    stream_dataset = TextData(raw_text=stream_texts, task_type="text_classification")

    print("Calculating NLP properties...")
    test_dataset.calculate_builtin_properties(include_properties=["Sentiment"])
    stream_dataset.calculate_builtin_properties(include_properties=["Sentiment"])

    failed = False

    print("Checking property drift...")
    property_result = PropertyDrift(properties=["Sentiment"]).run(
        train_dataset=test_dataset, test_dataset=stream_dataset
    )
    property_drift_score = property_result.value["Sentiment"]["Drift score"]
    print(f"  Sentiment property drift score: {property_drift_score:.4f} (threshold: {property_drift_threshold})")
    if property_drift_score > property_drift_threshold:
        print(f"[FAIL] Property drift ({property_drift_score:.4f}) exceeds threshold ({property_drift_threshold}).")
        failed = True

    print("Running predictions for prediction drift...")
    test_predictions = run_predictions(classifier, test_texts)
    stream_predictions = run_predictions(classifier, stream_texts)

    prediction_result = PredictionDrift().run(
        train_dataset=test_dataset,
        test_dataset=stream_dataset,
        train_predictions=test_predictions,
        test_predictions=stream_predictions,
    )
    prediction_drift_score = prediction_result.value["Drift score"]
    print(f"  Prediction drift score: {prediction_drift_score:.4f} (threshold: {prediction_drift_threshold})")
    if prediction_drift_score > prediction_drift_threshold:
        print(
            f"[FAIL] Prediction drift ({prediction_drift_score:.4f}) exceeds "
            f"threshold ({prediction_drift_threshold})."
        )
        failed = True

    if failed:
        sys.exit(1)

    print("All drift checks passed.")


if __name__ == "__main__":
    main()
