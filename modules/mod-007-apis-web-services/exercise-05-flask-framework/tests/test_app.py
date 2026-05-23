"""Tests for the Flask app — exercise the validation, routing, health probes.

Uses joblib + a tiny in-memory sklearn model as the fixture, so the tests
run without needing real model artifacts.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'solutions'))
from app import create_app  # noqa: E402


@pytest.fixture(scope="session")
def model_path(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("model")
    model = LinearRegression().fit(np.eye(4), np.arange(4, dtype=float))
    path = tmp / "model.joblib"
    joblib.dump(model, path)
    return str(path)


@pytest.fixture
def client(model_path):
    app = create_app({"TESTING": True, "MODEL_PATH": model_path, "FEATURE_COUNT": 4})
    with app.test_client() as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.get_json() == {"status": "ok"}


def test_ready_before_model_loaded_returns_503(model_path):
    # Use a fresh module global reset by reloading the loader module.
    from app.ml import loader
    loader._model = None
    app = create_app({"TESTING": True, "MODEL_PATH": model_path, "FEATURE_COUNT": 4})
    with app.test_client() as c:
        r = c.get("/ready")
    assert r.status_code == 503


def test_predict_happy_path(client):
    r = client.post("/v1/predict", json={"features": [1.0, 0.0, 0.0, 0.0]})
    assert r.status_code == 200
    body = r.get_json()
    assert "prediction" in body
    assert body["model_version"] == "latest"
    assert body["latency_ms"] >= 0


def test_predict_wrong_feature_count(client):
    r = client.post("/v1/predict", json={"features": [1.0, 2.0]})
    assert r.status_code == 400


def test_predict_validation_missing_features(client):
    r = client.post("/v1/predict", json={})
    assert r.status_code == 422  # Flask-Smorest validation error


def test_predict_batch(client):
    payload = {"items": [{"features": [1, 0, 0, 0]}, {"features": [0, 1, 0, 0]}]}
    r = client.post("/v1/predict/batch", json=payload)
    assert r.status_code == 200
    body = r.get_json()
    assert len(body["predictions"]) == 2


def test_predict_batch_too_large(client):
    payload = {"items": [{"features": [1, 0, 0, 0]}] * 1000}
    r = client.post("/v1/predict/batch", json=payload)
    assert r.status_code == 413


def test_metrics_endpoint_served(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    assert b"flask_http_request_total" in r.data
