"""
Integration tests for the Sentiment Analysis API.
"""

import pytest
from fastapi.testclient import TestClient

VALID_SENTIMENTS = {"positive", "negative", "neutral"}


@pytest.mark.parametrize(
    "text",
    [
        "The company reported record profits this quarter.",
        "Shares plummeted after the disappointing earnings call.",
        "The stock closed flat with no major news today.",
    ],
)
def test_predict_returns_valid_response(client: TestClient, text):
    response = client.post("/predict", json={"text": text})
    assert response.status_code == 200

    body = response.json()
    assert body["text"] == text
    assert body["sentiment"] in VALID_SENTIMENTS
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["latency_ms"] >= 0.0


def test_predict_batch(client: TestClient):
    texts = [
        "The company reported record profits this quarter.",
        "Shares plummeted after the disappointing earnings call.",
    ]
    response = client.post("/predict/batch", json={"texts": texts})
    assert response.status_code == 200

    results = response.json()
    assert len(results) == len(texts)
    for text, result in zip(texts, results):
        assert result["text"] == text
        assert result["sentiment"] in VALID_SENTIMENTS
        assert 0.0 <= result["confidence"] <= 1.0
        assert result["latency_ms"] >= 0.0


def test_predict_batch_empty_list_returns_422(client: TestClient):
    response = client.post("/predict/batch", json={"texts": []})
    assert response.status_code == 422
