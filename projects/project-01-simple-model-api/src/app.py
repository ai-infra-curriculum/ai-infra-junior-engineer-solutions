#!/usr/bin/env python3
"""
ML Model Serving API

Flask application that serves predictions from a pre-trained image classification model.

Usage:
    python app.py

    # Test endpoints:
    curl http://localhost:5000/health
    curl http://localhost:5000/info
    curl -X POST -F "file=@image.jpg" http://localhost:5000/predict
"""

from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
from PIL import Image
import io
import logging
import logging.config
import time
from typing import Dict, Any, Tuple
import sys
from pathlib import Path

from config import get_settings
from model_loader import ModelLoader


# Initialize Flask app
app = Flask(__name__)

# Load settings
settings = get_settings()

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Initialize model loader (global to load once)
model_loader: ModelLoader = None


def setup_model():
    """Initialize model loader on startup"""
    global model_loader

    logger.info("Starting model initialization...")
    try:
        model_loader = ModelLoader(
            model_name=settings.model_name,
            device=settings.model_device
        )

        # Warmup model
        model_loader.warmup()

        logger.info("Model initialized successfully")
        return True

    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        return False


def allowed_file(filename: str) -> bool:
    """
    Check if file extension is allowed

    Args:
        filename: Name of the file

    Returns:
        True if file extension is allowed
    """
    return Path(filename).suffix.lower() in settings.allowed_extensions


def validate_image(file) -> Tuple[bool, str]:
    """
    Validate uploaded image file

    Args:
        file: Uploaded file object

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check if file exists
    if not file:
        return False, "No file provided"

    # Check filename
    if file.filename == '':
        return False, "Empty filename"

    # Check file extension
    if not allowed_file(file.filename):
        return False, f"Invalid file type. Allowed: {settings.allowed_extensions}"

    # Check file size
    file.seek(0, 2)  # Seek to end
    file_size = file.tell()
    file.seek(0)  # Reset to beginning

    if file_size > settings.max_file_size:
        return False, f"File too large. Max size: {settings.max_file_size} bytes"

    if file_size == 0:
        return False, "Empty file"

    # Try to open as image
    try:
        img = Image.open(file)
        img.verify()
        file.seek(0)  # Reset for actual processing
        return True, ""
    except Exception as e:
        return False, f"Invalid image file: {str(e)}"


@app.before_request
def before_request():
    """Log request information"""
    request.start_time = time.time()
    logger.info(f"Request: {request.method} {request.path}")


@app.after_request
def after_request(response):
    """Log response information"""
    if hasattr(request, 'start_time'):
        duration = time.time() - request.start_time
        logger.info(f"Response: {response.status_code} ({duration:.3f}s)")
    return response


@app.errorhandler(413)
@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    """Handle file size limit exceeded"""
    return jsonify({
        "error": "File too large",
        "message": f"Maximum file size is {settings.max_file_size} bytes",
        "status": "error"
    }), 413


@app.errorhandler(400)
def handle_bad_request(e):
    """Handle bad request"""
    return jsonify({
        "error": "Bad request",
        "message": str(e),
        "status": "error"
    }), 400


@app.errorhandler(500)
def handle_internal_error(e):
    """Handle internal server error"""
    logger.error(f"Internal error: {e}")
    return jsonify({
        "error": "Internal server error",
        "message": "An unexpected error occurred",
        "status": "error"
    }), 500


@app.route('/health', methods=['GET'])
def health() -> Dict[str, Any]:
    """
    Health check endpoint

    Returns:
        JSON with health status
    """
    try:
        # Check if model is loaded
        if model_loader is None:
            return jsonify({
                "status": "unhealthy",
                "message": "Model not loaded",
                "timestamp": time.time()
            }), 503

        # Run a quick inference test
        return jsonify({
            "status": "healthy",
            "model": settings.model_name,
            "device": settings.model_device,
            "timestamp": time.time()
        }), 200

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            "status": "unhealthy",
            "message": str(e),
            "timestamp": time.time()
        }), 503


@app.route('/info', methods=['GET'])
def info() -> Dict[str, Any]:
    """
    Model information endpoint

    Returns:
        JSON with model metadata
    """
    try:
        if model_loader is None:
            return jsonify({
                "error": "Model not loaded",
                "status": "error"
            }), 503

        model_info = model_loader.get_model_info()

        return jsonify({
            "status": "success",
            "model": model_info,
            "config": {
                "max_file_size": settings.max_file_size,
                "allowed_extensions": settings.allowed_extensions,
                "request_timeout": settings.request_timeout
            }
        }), 200

    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        return jsonify({
            "error": "Failed to get model info",
            "message": str(e),
            "status": "error"
        }), 500


@app.route('/predict', methods=['POST'])
def predict() -> Dict[str, Any]:
    """
    Prediction endpoint

    Accepts image file and returns top-K predictions

    Request:
        file: Image file (multipart/form-data)
        top_k: Number of predictions to return (optional, default=5)

    Returns:
        JSON with predictions
    """
    try:
        # Check if model is loaded
        if model_loader is None:
            return jsonify({
                "error": "Model not loaded",
                "status": "error"
            }), 503

        # Get top_k parameter
        top_k = request.form.get('top_k', 5, type=int)
        if not (1 <= top_k <= 10):
            return jsonify({
                "error": "Invalid top_k value",
                "message": "top_k must be between 1 and 10",
                "status": "error"
            }), 400

        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({
                "error": "No file provided",
                "message": "Request must include 'file' field",
                "status": "error"
            }), 400

        file = request.files['file']

        # Validate file
        is_valid, error_msg = validate_image(file)
        if not is_valid:
            return jsonify({
                "error": "Invalid image",
                "message": error_msg,
                "status": "error"
            }), 400

        # Load image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        # Run prediction
        start_time = time.time()
        predictions = model_loader.predict(image, top_k=top_k)
        inference_time = time.time() - start_time

        # Format response
        response = {
            "status": "success",
            "predictions": predictions,
            "metadata": {
                "filename": secure_filename(file.filename),
                "inference_time": round(inference_time, 3),
                "model": settings.model_name,
                "device": settings.model_device,
                "image_size": image.size
            }
        }

        logger.info(f"Prediction successful: {predictions[0]['class']} ({predictions[0]['confidence']:.3f})")

        return jsonify(response), 200

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return jsonify({
            "error": "Validation error",
            "message": str(e),
            "status": "error"
        }), 400

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return jsonify({
            "error": "Prediction failed",
            "message": str(e),
            "status": "error"
        }), 500


@app.route('/', methods=['GET'])
def root() -> Dict[str, Any]:
    """
    Root endpoint with API documentation

    Returns:
        JSON with API information
    """
    return jsonify({
        "name": "ML Model Serving API",
        "version": "1.0.0",
        "endpoints": {
            "/": "API documentation (this page)",
            "/health": "Health check endpoint",
            "/info": "Model information",
            "/predict": "Image classification endpoint (POST with file)"
        },
        "model": settings.model_name,
        "status": "running"
    }), 200


# Flask configuration
app.config['MAX_CONTENT_LENGTH'] = settings.max_file_size


def main():
    """Main entry point"""
    logger.info("Starting ML Model API Server...")
    logger.info(f"Configuration: {settings.to_dict()}")

    # Initialize model
    if not setup_model():
        logger.error("Failed to initialize model. Exiting.")
        sys.exit(1)

    # Run Flask app
    logger.info(f"Starting Flask server on {settings.host}:{settings.port}")
    app.run(
        host=settings.host,
        port=settings.port,
        debug=settings.debug,
        threaded=True
    )


if __name__ == '__main__':
    main()
