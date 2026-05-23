"""Flask app factory."""
from __future__ import annotations

import logging
import sys
from typing import Any

from flask import Flask
from flask_smorest import Api
from prometheus_flask_exporter import PrometheusMetrics

from .config import Config
from .routes.health import bp as health_bp
from .routes.predict import blp as predict_blp


def _configure_logging(level: str) -> None:
    """Plain structured logger to stdout. Replace with python-json-logger in prod."""
    root = logging.getLogger()
    root.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        '{"ts":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","msg":"%(message)s"}'
    ))
    root.addHandler(handler)
    root.setLevel(level)


def create_app(overrides: dict[str, Any] | None = None) -> Flask:
    app = Flask(__name__)
    app.config.from_object(Config())
    if overrides:
        app.config.update(overrides)

    _configure_logging(app.config["LOG_LEVEL"])

    PrometheusMetrics(app, group_by="endpoint")

    app.register_blueprint(health_bp)

    api = Api(app)
    api.register_blueprint(predict_blp)

    @app.errorhandler(Exception)
    def _unhandled(err: Exception):
        app.logger.exception("unhandled error")
        return {"error": "internal_server_error"}, 500

    return app
