"""Thread-safe lazy model loader.

One model per worker process. Loaded once on first request (or at startup if a
warmup endpoint is hit). Re-loading is intentionally not supported — to push a
new model, deploy a new container image.
"""
from __future__ import annotations

import logging
from pathlib import Path
from threading import Lock
from typing import Any

import joblib


_log = logging.getLogger(__name__)
_lock = Lock()
_model: Any = None
_loaded_from: str | None = None


def get_model(model_path: str) -> Any:
    global _model, _loaded_from
    with _lock:
        if _model is None:
            path = Path(model_path)
            if not path.is_file():
                raise FileNotFoundError(f"model not found at {model_path}")
            _log.info(f"loading model from {model_path}")
            _model = joblib.load(path)
            _loaded_from = model_path
            _log.info(f"loaded model class {type(_model).__name__}")
        return _model


def is_loaded() -> bool:
    return _model is not None


def loaded_from() -> str | None:
    return _loaded_from
