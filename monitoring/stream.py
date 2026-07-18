"""
Production monitoring stream for FinBERT sentiment API.

Reads headlines from data/stream.csv, sends them to the /predict endpoint,
and logs aggregated metrics to MLflow every WINDOW_SIZE predictions.

Each observation window logs:
    - Sentiment distribution (% positive, % negative, % neutral)
    - Average confidence score
    - Average latency (ms)

Run from the project root:
    python monitoring/stream.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import mlflow
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

# Not os.getenv("API_HOST", ...) — .env sets API_HOST=0.0.0.0 for the server to bind
# to all interfaces, but that's not a valid address for a client to connect to.
API_URL = f"http://localhost:{os.getenv('API_PORT', 8000)}"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "finbert-evaluation")
WINDOW_SIZE = 50  # number of predictions per observation window
SLEEP_MS = 100    # delay between requests to simulate real traffic (ms)


def predict(text: str) -> dict:
    response = requests.post(
        f"{API_URL}/predict",
        json={"text": text},
        timeout=10,
    )
    response.raise_for_status()
    return response.json()

def log_window(window: list[dict], window_idx: int) -> None:
    """Log aggregated metrics for one observation window to MLflow."""
    n = len(window)
    sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
    for result in window:
        sentiment_counts[result["sentiment"]] += 1

    metrics = {
        "pct_positive": 100 * sentiment_counts["positive"] / n,
        "pct_negative": 100 * sentiment_counts["negative"] / n,
        "pct_neutral": 100 * sentiment_counts["neutral"] / n,
        "avg_confidence": sum(r["confidence"] for r in window) / n,
        "avg_latency_ms": sum(r["latency_ms"] for r in window) / n,
    }
    mlflow.log_metrics(metrics, step=window_idx)
    print(f"[window {window_idx}] {metrics}")


def main():
    stream_df = pd.read_csv("data/stream.csv")
    headlines = stream_df["text"].tolist()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    window = []
    window_idx = 0

    with mlflow.start_run(run_name="monitoring-stream"):
        for headline in headlines:
            result = predict(headline)
            window.append(result)

            if len(window) == WINDOW_SIZE:
                log_window(window, window_idx)
                window_idx += 1
                window = []

            time.sleep(SLEEP_MS / 1000)

        if window:
            log_window(window, window_idx)
            window_idx += 1

    print(f"Streamed {len(headlines)} headlines across {window_idx} windows.")


if __name__ == "__main__":
    main()
