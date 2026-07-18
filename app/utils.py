"""
Model loading utilities.
"""

import json
import os

from huggingface_hub import hf_hub_download
from transformers import pipeline


def _patch_model_config(model_id: str) -> None:
    """
    baptle/FinBERT_market_based's config.json stores id2label/label2id as numeric
    floats instead of string labels, which newer transformers versions reject
    outright on load. Patches the cached config in place before the pipeline is
    built — mirrors the fix in scripts/smoke_test.py, but applied unconditionally
    here since a fresh container/runner has no pre-existing patched cache to rely on.
    """
    config_path = hf_hub_download(repo_id=model_id, filename="config.json")
    with open(config_path) as f:
        config = json.load(f)

    if not isinstance(config.get("id2label", {}).get("0"), str):
        config["id2label"] = {"0": "negative", "1": "neutral", "2": "positive"}
        config["label2id"] = {"negative": "0", "neutral": "1", "positive": "2"}
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)


def load_classifier():
    """
    Load the sentiment classifier.

    Controlled by the MODEL_SOURCE environment variable:
      - "mlflow" (default) — loads from the MLflow Model Registry
      - "huggingface"       — loads directly from HuggingFace Hub (used in CI/prod)
    """
    model_source = os.getenv("MODEL_SOURCE", "mlflow")

    if model_source == "huggingface":
        hf_model_id = os.getenv("HF_MODEL_ID", "baptle/FinBERT_market_based")
        print(f"Loading model from HuggingFace: {hf_model_id}")
        _patch_model_config(hf_model_id)
        return pipeline(
            "text-classification",
            model=hf_model_id,
            tokenizer=hf_model_id,
            device="cpu",
        )

    import mlflow.transformers

    model_name = os.getenv("MODEL_NAME", "finbert")
    model_alias = os.getenv("MODEL_STAGE", "production")
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    print(f"Loading model from MLflow: models:/{model_name}@{model_alias}")
    mlflow.set_tracking_uri(tracking_uri)
    return mlflow.transformers.load_model(f"models:/{model_name}@{model_alias}")
