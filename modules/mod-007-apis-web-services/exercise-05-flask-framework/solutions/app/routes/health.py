"""Liveness and readiness probes.

Liveness  (/health): process up. K8s restarts the pod if this fails.
Readiness (/ready):  ready to take traffic — implies the model is loaded.
"""
from __future__ import annotations

from flask import Blueprint, current_app

from ..ml.loader import is_loaded, loaded_from

bp = Blueprint("health", __name__)


@bp.get("/health")
def health():
    return {"status": "ok"}


@bp.get("/ready")
def ready():
    if is_loaded():
        return {"status": "ready", "model": loaded_from()}
    # 503 so the LB takes us out of rotation
    return {"status": "not_ready"}, 503
