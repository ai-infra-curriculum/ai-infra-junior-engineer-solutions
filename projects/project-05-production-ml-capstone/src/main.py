"""Production ML serving capstone.

Integrates the building blocks from projects 01-04 into a single
production-grade Flask application:

- ``ModelManager`` loads a model from MLflow Model Registry.
- ``@require_api_key`` enforces shared-secret authentication.
- Prometheus metrics expose latency, throughput, and model version.
- Endpoints: ``/health``, ``/predict``, ``/info``, ``/metrics``, ``/reload``.

When MLflow / torch / PIL are unavailable (e.g., minimal CI containers)
the module degrades gracefully via a deterministic stub model so the
HTTP surface stays testable.
"""

from __future__ import annotations

import io
import logging
import os
import sys
import time
from functools import wraps
from typing import Any, Callable, Dict, Optional

from flask import Flask, Response, jsonify, request
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

try:
    import mlflow
    import mlflow.pytorch

    MLFLOW_AVAILABLE = True
except ImportError:
    mlflow = None
    MLFLOW_AVAILABLE = False

try:
    import torch
    from torchvision import transforms

    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    transforms = None
    TORCH_AVAILABLE = False

try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    Image = None
    PIL_AVAILABLE = False


MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "image-classifier")
MODEL_VERSION = os.getenv("MODEL_VERSION", "latest")
API_KEYS = [k.strip() for k in os.getenv("API_KEYS", "").split(",") if k.strip()]
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
PORT = int(os.getenv("PORT", "5000"))
MODEL_AUTOLOAD = os.getenv("MODEL_AUTOLOAD", "1") == "1"

_ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/jpg"}

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(stream=sys.stdout)],
)
logger = logging.getLogger("ml-capstone")


request_count = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)
request_latency = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency",
    ["method", "endpoint"],
)
prediction_count = Counter(
    "model_predictions_total",
    "Total model predictions",
    ["model_name", "model_version", "status"],
)
prediction_latency = Histogram(
    "model_prediction_duration_seconds",
    "Model prediction latency",
    ["model_name", "model_version"],
)
model_version_info = Gauge(
    "model_version_info",
    "Currently loaded model version",
    ["model_name", "version"],
)


class _StubModel:
    classes = ("cat", "dog", "bird", "fish")

    def __call__(self, _tensor: Any) -> Any:
        return [[0.05, 0.85, 0.05, 0.05]]


class ModelManager:
    """Load + cache an ML model from MLflow Model Registry."""

    def __init__(
        self,
        mlflow_uri: str,
        model_name: str,
        model_version: str = "latest",
    ) -> None:
        self.mlflow_uri = mlflow_uri
        self.model_name = model_name
        self.model_version = model_version
        self.model: Any = None
        self.model_metadata: Dict[str, Any] = {}
        if MLFLOW_AVAILABLE and mlflow is not None:
            mlflow.set_tracking_uri(self.mlflow_uri)

    def load_model(self) -> None:
        logger.info(
            "Loading model %s version=%s from %s",
            self.model_name,
            self.model_version,
            self.mlflow_uri,
        )
        if not MLFLOW_AVAILABLE or mlflow is None:
            logger.warning(
                "MLflow unavailable; falling back to deterministic stub model."
            )
            self.model = _StubModel()
            self.model_metadata = {
                "name": self.model_name,
                "version": "stub",
                "uri": "local://stub",
            }
            model_version_info.labels(
                model_name=self.model_name, version="stub"
            ).set(1)
            return

        try:
            client = mlflow.tracking.MlflowClient()
            if self.model_version == "latest":
                versions = client.get_latest_versions(
                    self.model_name, stages=["Production"]
                )
                if not versions:
                    raise ValueError(
                        f"No Production model found for {self.model_name!r}"
                    )
                resolved_version = versions[0].version
            else:
                resolved_version = self.model_version

            model_uri = f"models:/{self.model_name}/{resolved_version}"
            if TORCH_AVAILABLE:
                self.model = mlflow.pytorch.load_model(model_uri)
            else:
                self.model = mlflow.pyfunc.load_model(model_uri)
            self.model_metadata = {
                "name": self.model_name,
                "version": resolved_version,
                "uri": model_uri,
            }
            model_version_info.labels(
                model_name=self.model_name, version=resolved_version
            ).set(1)
            logger.info("Model loaded successfully: %s", model_uri)
        except Exception:
            logger.exception("Failed to load model")
            raise

    def predict(self, input_data: Any) -> Dict[str, Any]:
        if self.model is None:
            raise RuntimeError("Model is not loaded")

        version = self.model_metadata.get("version", "unknown")
        start_time = time.time()
        try:
            if TORCH_AVAILABLE and torch is not None and hasattr(self.model, "to"):
                self.model.eval()
                with torch.no_grad():
                    outputs = self.model(input_data)
                logits = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                top_values, top_indices = torch.topk(
                    probabilities, k=min(5, probabilities.shape[-1])
                )
                predictions = [
                    {"class_index": int(idx), "confidence": float(value)}
                    for idx, value in zip(top_indices.tolist(), top_values.tolist())
                ]
            else:
                stub_outputs = self.model(input_data)
                logits = stub_outputs[0]
                total = sum(logits) or 1.0
                predictions = sorted(
                    (
                        {"class_index": idx, "confidence": value / total}
                        for idx, value in enumerate(logits)
                    ),
                    key=lambda item: item["confidence"],
                    reverse=True,
                )

            latency = time.time() - start_time
            prediction_latency.labels(
                model_name=self.model_name, model_version=version
            ).observe(latency)
            prediction_count.labels(
                model_name=self.model_name, model_version=version, status="success"
            ).inc()
            return {
                "predictions": predictions,
                "model_version": version,
                "latency_ms": latency * 1000,
            }
        except Exception:
            prediction_count.labels(
                model_name=self.model_name, model_version=version, status="error"
            ).inc()
            logger.exception("Prediction failed")
            raise

    def get_info(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_version": self.model_metadata.get("version", "unknown"),
            "model_uri": self.model_metadata.get("uri", "unknown"),
            "status": "loaded" if self.model is not None else "not_loaded",
        }


def require_api_key(func: Callable[..., Any]) -> Callable[..., Any]:
    """Reject requests missing or supplying an unknown ``X-API-Key``."""

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        api_key = request.headers.get("X-API-Key")
        if not API_KEYS:
            logger.warning(
                "API key check skipped because API_KEYS is empty (path=%s)",
                request.path,
            )
            return func(*args, **kwargs)
        if not api_key:
            logger.warning("Missing API key (path=%s)", request.path)
            return jsonify({"error": "API key required"}), 401
        if api_key not in API_KEYS:
            logger.warning("Invalid API key attempt (path=%s)", request.path)
            return jsonify({"error": "Invalid API key"}), 403
        return func(*args, **kwargs)

    return wrapper


def validate_image_upload(file: Any) -> bool:
    """Validate an uploaded file is an image within size limits."""
    if file is None or not getattr(file, "filename", None):
        raise ValueError("No file provided")

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    if file_size > MAX_UPLOAD_BYTES:
        raise ValueError(
            f"File too large: {file_size} bytes (max {MAX_UPLOAD_BYTES})"
        )

    mimetype = (file.mimetype or "").lower()
    if mimetype and mimetype not in _ALLOWED_MIME_TYPES:
        raise ValueError(f"Unsupported MIME type: {mimetype}")

    if PIL_AVAILABLE and Image is not None:
        try:
            data = file.read()
            img = Image.open(io.BytesIO(data))
            img.verify()
            file.seek(0)
        except Exception as exc:
            raise ValueError(f"Invalid image file: {exc}") from exc
    return True


def preprocess_image(file: Any) -> Any:
    """Convert an uploaded image into a model-ready tensor."""
    if not (PIL_AVAILABLE and TORCH_AVAILABLE):
        return file.read()

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    img = Image.open(file).convert("RGB")
    tensor = transform(img)
    return tensor.unsqueeze(0)


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
model_manager = ModelManager(
    mlflow_uri=MLFLOW_TRACKING_URI,
    model_name=MODEL_NAME,
    model_version=MODEL_VERSION,
)


def _record_request(method: str, endpoint: str, status: int, started: float) -> None:
    request_latency.labels(method=method, endpoint=endpoint).observe(
        time.time() - started
    )
    request_count.labels(method=method, endpoint=endpoint, status=str(status)).inc()


@app.route("/health", methods=["GET"])
def health():
    started = time.time()
    try:
        info_payload = model_manager.get_info()
        if info_payload["status"] != "loaded":
            _record_request("GET", "/health", 503, started)
            return jsonify({"status": "unhealthy", "reason": "model not loaded"}), 503
        _record_request("GET", "/health", 200, started)
        return jsonify({"status": "healthy", "model": info_payload}), 200
    except Exception as exc:
        logger.exception("Health check failed")
        _record_request("GET", "/health", 503, started)
        return jsonify({"status": "unhealthy", "reason": str(exc)}), 503


@app.route("/predict", methods=["POST"])
@require_api_key
def predict():
    started = time.time()
    try:
        if "file" not in request.files:
            _record_request("POST", "/predict", 400, started)
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]
        validate_image_upload(file)
        input_tensor = preprocess_image(file)
        result = model_manager.predict(input_tensor)

        response = {
            "predictions": result["predictions"],
            "model_version": result["model_version"],
            "inference_time_ms": result["latency_ms"],
        }
        _record_request("POST", "/predict", 200, started)
        return jsonify(response), 200

    except ValueError as exc:
        logger.warning("Validation error: %s", exc)
        _record_request("POST", "/predict", 400, started)
        return jsonify({"error": str(exc)}), 400
    except Exception:
        logger.exception("Prediction failed")
        _record_request("POST", "/predict", 500, started)
        return jsonify({"error": "Internal server error"}), 500


@app.route("/info", methods=["GET"])
@require_api_key
def info():
    started = time.time()
    try:
        info_payload = model_manager.get_info()
        _record_request("GET", "/info", 200, started)
        return (
            jsonify({"service": "ml-api", "version": "1.0.0", "model": info_payload}),
            200,
        )
    except Exception as exc:
        logger.exception("Info endpoint failed")
        _record_request("GET", "/info", 500, started)
        return jsonify({"error": str(exc)}), 500


@app.route("/metrics", methods=["GET"])
def metrics():
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)


@app.route("/reload", methods=["POST"])
@require_api_key
def reload_model():
    started = time.time()
    try:
        model_manager.load_model()
        _record_request("POST", "/reload", 200, started)
        return (
            jsonify(
                {
                    "status": "success",
                    "message": "Model reloaded",
                    "model": model_manager.get_info(),
                }
            ),
            200,
        )
    except Exception as exc:
        logger.exception("Model reload failed")
        _record_request("POST", "/reload", 500, started)
        return jsonify({"status": "error", "message": str(exc)}), 500


def startup() -> None:
    logger.info("Application starting up...")
    logger.info("MLflow URI: %s", MLFLOW_TRACKING_URI)
    logger.info("Model: %s v%s", MODEL_NAME, MODEL_VERSION)
    logger.info("API keys configured: %d", len(API_KEYS))
    model_manager.load_model()


if MODEL_AUTOLOAD:
    try:
        startup()
    except Exception:
        logger.exception("Startup failed; service will report unhealthy")


if __name__ == "__main__":
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=PORT, debug=debug)
