"""
Sentiment Prediction API.

Endpoints:
    GET  /health          — service health status
    POST /predict         — single headline sentiment
    POST /predict/batch   — batch headline sentiment
    GET  /metrics         — Prometheus metrics
"""

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field
from utils import load_classifier

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("finbert-api")


def log(level: str, message: str, **kwargs) -> None:
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "level": level,
        "message": message,
        **kwargs,
    }
    logger.log(getattr(logging, level.upper(), logging.INFO), json.dumps(entry))


PREDICTION_REQUESTS = Counter(
    "prediction_requests_total",
    "Total number of prediction requests",
    ["sentiment"],
)
PREDICTION_LATENCY = Histogram(
    "prediction_latency_ms",
    "Prediction latency in milliseconds",
    # Default buckets are calibrated for second-scale latencies; ours are ~tens to
    # low-thousands of milliseconds, so they need their own bucket boundaries.
    buckets=(10, 25, 50, 75, 100, 250, 500, 750, 1000, 2500, 5000, 10000),
)
PREDICTION_ERRORS = Counter(
    "prediction_errors_total",
    "Total number of prediction errors",
)

load_dotenv()

classifiers = {}


class PredictRequest(BaseModel):
    text: str = Field(min_length=1)


class PredictBatchRequest(BaseModel):
    texts: list[str] = Field(min_length=1)


class PredictionResult(BaseModel):
    text: str
    sentiment: str
    confidence: float
    latency_ms: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    log("INFO", "Loading model...")
    classifiers["sentiment"] = load_classifier()
    log("INFO", "Model loaded successfully")
    yield
    classifiers.clear()
    log("INFO", "Model unloaded")


app = FastAPI(title="Sentiment Analysis API", lifespan=lifespan)


def run_predictions(texts: list[str]) -> list[PredictionResult]:
    try:
        for text in texts:
            if len(text) > 2000:
                log("WARNING", "Input text exceeds 2000 characters and will be truncated by the model", length=len(text))

        start = time.perf_counter()
        raw_results = classifiers["sentiment"](texts)
        latency_ms = (time.perf_counter() - start) * 1000

        results = [
            PredictionResult(
                text=text,
                sentiment=raw["label"],
                confidence=raw["score"],
                latency_ms=latency_ms,
            )
            for text, raw in zip(texts, raw_results)
        ]

        for result in results:
            PREDICTION_LATENCY.observe(result.latency_ms)
            PREDICTION_REQUESTS.labels(sentiment=result.sentiment).inc()
            log(
                "INFO",
                "Prediction served",
                sentiment=result.sentiment,
                confidence=result.confidence,
                latency_ms=result.latency_ms,
            )

        return results
    except Exception as e:
        PREDICTION_ERRORS.inc()
        log("ERROR", "Prediction failed", error=str(e))
        raise e


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
def health():
    if "sentiment" not in classifiers:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResult)
def predict(request: PredictRequest):
    if "sentiment" not in classifiers:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return run_predictions([request.text])[0]


@app.post("/predict/batch", response_model=list[PredictionResult])
def predict_batch(request: PredictBatchRequest):
    if "sentiment" not in classifiers:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return run_predictions(request.texts)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", 8000)),
    )
