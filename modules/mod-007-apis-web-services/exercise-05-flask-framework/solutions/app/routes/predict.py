"""/predict endpoint with Flask-Smorest for auto-docs."""
from __future__ import annotations

import time

from flask import abort, current_app
from flask_smorest import Blueprint
from marshmallow import Schema, fields, validate

from ..ml.loader import get_model


class PredictRequest(Schema):
    features = fields.List(
        fields.Float(),
        required=True,
        validate=validate.Length(min=1, max=1024),
        metadata={"description": "Numeric feature vector."},
    )
    model_version = fields.String(load_default="latest")


class PredictResponse(Schema):
    prediction = fields.Float(required=True)
    latency_ms = fields.Float(required=True)
    model_version = fields.String(required=True)


class BatchPredictRequest(Schema):
    items = fields.List(
        fields.Nested(PredictRequest),
        required=True,
        validate=validate.Length(min=1),
    )


class BatchPredictResponse(Schema):
    predictions = fields.List(fields.Nested(PredictResponse), required=True)


blp = Blueprint(
    "predict", "predict", url_prefix="/v1",
    description="Single and batch prediction endpoints.",
)


@blp.route("/predict", methods=["POST"])
@blp.arguments(PredictRequest)
@blp.response(200, PredictResponse)
def predict(payload):
    expected = current_app.config["FEATURE_COUNT"]
    if len(payload["features"]) != expected:
        abort(400, description=f"features must have length {expected}")

    model = get_model(current_app.config["MODEL_PATH"])
    t0 = time.perf_counter()
    pred = float(model.predict([payload["features"]])[0])
    elapsed_ms = (time.perf_counter() - t0) * 1000

    return {
        "prediction": pred,
        "latency_ms": round(elapsed_ms, 3),
        "model_version": payload["model_version"],
    }


@blp.route("/predict/batch", methods=["POST"])
@blp.arguments(BatchPredictRequest)
@blp.response(200, BatchPredictResponse)
def predict_batch(payload):
    max_batch = current_app.config["MAX_BATCH"]
    if len(payload["items"]) > max_batch:
        abort(413, description=f"batch larger than max {max_batch}")

    expected = current_app.config["FEATURE_COUNT"]
    rows = []
    for item in payload["items"]:
        if len(item["features"]) != expected:
            abort(400, description=f"every item must have features of length {expected}")
        rows.append(item["features"])

    model = get_model(current_app.config["MODEL_PATH"])
    t0 = time.perf_counter()
    preds = model.predict(rows)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    per_item_ms = elapsed_ms / len(rows)

    return {
        "predictions": [
            {
                "prediction": float(p),
                "latency_ms": round(per_item_ms, 3),
                "model_version": item.get("model_version", "latest"),
            }
            for p, item in zip(preds, payload["items"])
        ],
    }
