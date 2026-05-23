"""
ML API Main Application

Simple Flask API for serving ML model predictions.
"""

import logging
import os
from datetime import datetime

from flask import Flask, jsonify, request

from model import MLModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Load model
MODEL_PATH = os.getenv("MODEL_PATH", "/models/model.pkl")
try:
    model = MLModel(MODEL_PATH)
    logger.info(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    logger.warning(f"Could not load model: {e}. Using dummy model.")
    model = MLModel()  # Use dummy model


@app.route("/")
def root():
    """Root endpoint with API information."""
    return jsonify({
        "name": "ML Inference API",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": model.is_loaded(),
        "timestamp": datetime.utcnow().isoformat()
    })


@app.route("/health")
def health():
    """Health check endpoint."""
    status = {
        "status": "healthy",
        "model_loaded": model.is_loaded(),
        "timestamp": datetime.utcnow().isoformat()
    }

    if not model.is_loaded():
        logger.warning("Health check: Model not loaded")
        return jsonify(status), 503

    return jsonify(status), 200


@app.route("/ready")
def ready():
    """Readiness check endpoint."""
    if model.is_loaded():
        return jsonify({"ready": True}), 200
    else:
        return jsonify({"ready": False, "reason": "Model not loaded"}), 503


@app.route("/predict", methods=["POST"])
def predict():
    """
    Prediction endpoint.

    Request body:
    {
        "features": [1.0, 2.0, 3.0, ...]
    }

    Response:
    {
        "prediction": 0,
        "probability": [0.7, 0.3],
        "timestamp": "2024-..."
    }
    """
    try:
        # Validate request
        if not request.json:
            return jsonify({"error": "Request body must be JSON"}), 400

        if "features" not in request.json:
            return jsonify({"error": "Missing 'features' in request"}), 400

        features = request.json["features"]

        # Make prediction
        prediction, probability = model.predict(features)

        response = {
            "prediction": int(prediction),
            "probability": probability.tolist() if hasattr(probability, "tolist") else probability,
            "timestamp": datetime.utcnow().isoformat()
        }

        logger.info(f"Prediction made: {prediction}")
        return jsonify(response), 200

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": "Internal server error"}), 500


@app.route("/batch-predict", methods=["POST"])
def batch_predict():
    """
    Batch prediction endpoint.

    Request body:
    {
        "features": [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ]
    }
    """
    try:
        if not request.json or "features" not in request.json:
            return jsonify({"error": "Missing 'features' in request"}), 400

        features_list = request.json["features"]

        if not isinstance(features_list, list):
            return jsonify({"error": "'features' must be a list"}), 400

        # Make batch predictions
        predictions, probabilities = model.predict_batch(features_list)

        response = {
            "predictions": [int(p) for p in predictions],
            "probabilities": [p.tolist() if hasattr(p, "tolist") else p for p in probabilities],
            "count": len(predictions),
            "timestamp": datetime.utcnow().isoformat()
        }

        logger.info(f"Batch prediction: {len(predictions)} samples")
        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/model/info")
def model_info():
    """Get model information."""
    try:
        info = model.get_info()
        return jsonify(info), 200
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        return jsonify({"error": str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal error: {error}")
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    debug = os.getenv("DEBUG", "false").lower() == "true"

    logger.info(f"Starting ML API on port {port}")
    app.run(host="0.0.0.0", port=port, debug=debug)
