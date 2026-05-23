"""Environment-driven config. Hard-fails on missing required values."""
from __future__ import annotations

import os


def _required(name: str) -> str:
    value = os.environ.get(name)
    if value is None or value == "":
        raise RuntimeError(f"required env var {name!r} not set")
    return value


class Config:
    LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
    MODEL_PATH = os.environ.get("MODEL_PATH", "/models/current/model.joblib")
    FEATURE_COUNT = int(os.environ.get("FEATURE_COUNT", "10"))
    MAX_BATCH = int(os.environ.get("MAX_BATCH", "128"))

    # Flask-Smorest config
    API_TITLE = "Model API"
    API_VERSION = "v1"
    OPENAPI_VERSION = "3.0.3"
    OPENAPI_URL_PREFIX = "/"
    OPENAPI_SWAGGER_UI_PATH = "/docs"
    OPENAPI_SWAGGER_UI_URL = "https://cdn.jsdelivr.net/npm/swagger-ui-dist/"
