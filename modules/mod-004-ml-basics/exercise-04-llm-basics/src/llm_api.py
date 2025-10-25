"""LLM Flask API

A production-ready Flask API for serving LLM text generation.
Includes health checks, validation, error handling, and monitoring.
"""

import logging
import os
import time
from typing import Dict, Any, Optional
from flask import Flask, request, jsonify, Response
from transformers import pipeline
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration from environment variables
MODEL_NAME = os.getenv('MODEL_NAME', 'gpt2')
DEVICE = int(os.getenv('DEVICE', '-1'))  # -1 for CPU, 0+ for GPU
MAX_LENGTH_LIMIT = int(os.getenv('MAX_LENGTH_LIMIT', '200'))
DEFAULT_MAX_LENGTH = int(os.getenv('DEFAULT_MAX_LENGTH', '50'))
DEFAULT_TEMPERATURE = float(os.getenv('DEFAULT_TEMPERATURE', '0.7'))

# Initialize Flask app
app = Flask(__name__)

# Global variable for the model (loaded once at startup)
generator = None


def initialize_model():
    """Initialize the LLM model at startup."""
    global generator

    logger.info("=" * 80)
    logger.info("Initializing LLM API")
    logger.info("=" * 80)
    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"Device: {'CPU' if DEVICE == -1 else f'GPU:{DEVICE}'}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"Max length limit: {MAX_LENGTH_LIMIT}")

    try:
        logger.info("Loading model... This may take a moment.")
        start_time = time.time()

        generator = pipeline(
            'text-generation',
            model=MODEL_NAME,
            device=DEVICE
        )

        load_time = time.time() - start_time
        logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
        logger.info("API is ready to serve requests")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise


@app.route('/health', methods=['GET'])
def health() -> Response:
    """Health check endpoint.

    Returns:
        JSON response with health status
    """
    if generator is None:
        return jsonify({
            "status": "unhealthy",
            "error": "Model not loaded"
        }), 503

    return jsonify({
        "status": "healthy",
        "model": MODEL_NAME,
        "device": "CPU" if DEVICE == -1 else f"GPU:{DEVICE}",
        "cuda_available": torch.cuda.is_available()
    }), 200


@app.route('/info', methods=['GET'])
def info() -> Response:
    """Get API and model information.

    Returns:
        JSON response with API information
    """
    return jsonify({
        "api_version": "1.0.0",
        "model": MODEL_NAME,
        "device": "CPU" if DEVICE == -1 else f"GPU:{DEVICE}",
        "cuda_available": torch.cuda.is_available(),
        "max_length_limit": MAX_LENGTH_LIMIT,
        "default_max_length": DEFAULT_MAX_LENGTH,
        "default_temperature": DEFAULT_TEMPERATURE,
        "supported_parameters": {
            "prompt": "Text prompt (required)",
            "max_length": f"Maximum length (default: {DEFAULT_MAX_LENGTH}, max: {MAX_LENGTH_LIMIT})",
            "temperature": "Sampling temperature (default: 0.7, range: 0.0-2.0)",
            "top_k": "Top-K sampling (default: 50)",
            "top_p": "Top-P/nucleus sampling (default: 1.0)"
        }
    }), 200


def validate_generation_request(data: Dict[str, Any]) -> tuple[Optional[str], Optional[int]]:
    """Validate generation request parameters.

    Args:
        data: Request data

    Returns:
        Tuple of (error_message, status_code) or (None, None) if valid
    """
    # Check if prompt exists
    if not data or 'prompt' not in data:
        return "Missing required parameter: 'prompt'", 400

    prompt = data.get('prompt', '')
    if not prompt or not prompt.strip():
        return "Prompt cannot be empty", 400

    # Validate max_length
    max_length = data.get('max_length', DEFAULT_MAX_LENGTH)
    if not isinstance(max_length, (int, float)):
        return "max_length must be a number", 400

    max_length = int(max_length)
    if max_length < 1:
        return "max_length must be at least 1", 400

    if max_length > MAX_LENGTH_LIMIT:
        return f"max_length cannot exceed {MAX_LENGTH_LIMIT}", 400

    # Validate temperature
    temperature = data.get('temperature', DEFAULT_TEMPERATURE)
    if not isinstance(temperature, (int, float)):
        return "temperature must be a number", 400

    if temperature < 0.0 or temperature > 2.0:
        return "temperature must be between 0.0 and 2.0", 400

    # Validate top_k if provided
    top_k = data.get('top_k')
    if top_k is not None:
        if not isinstance(top_k, (int, float)):
            return "top_k must be a number", 400
        if int(top_k) < 0:
            return "top_k must be non-negative", 400

    # Validate top_p if provided
    top_p = data.get('top_p')
    if top_p is not None:
        if not isinstance(top_p, (int, float)):
            return "top_p must be a number", 400
        if not (0.0 <= float(top_p) <= 1.0):
            return "top_p must be between 0.0 and 1.0", 400

    return None, None


@app.route('/generate', methods=['POST'])
def generate() -> Response:
    """Text generation endpoint.

    Expected JSON payload:
        {
            "prompt": "Your text prompt",
            "max_length": 50,  # optional, default: 50
            "temperature": 0.7,  # optional, default: 0.7
            "top_k": 50,  # optional
            "top_p": 1.0  # optional
        }

    Returns:
        JSON response with generated text and metadata
    """
    if generator is None:
        return jsonify({
            "error": "Model not loaded",
            "status": "Service unavailable"
        }), 503

    try:
        # Get request data
        data = request.get_json()

        # Validate request
        error, status_code = validate_generation_request(data)
        if error:
            return jsonify({"error": error}), status_code

        # Extract parameters
        prompt = data.get('prompt', '').strip()
        max_length = int(data.get('max_length', DEFAULT_MAX_LENGTH))
        temperature = float(data.get('temperature', DEFAULT_TEMPERATURE))
        top_k = int(data.get('top_k', 50)) if data.get('top_k') is not None else 50
        top_p = float(data.get('top_p', 1.0)) if data.get('top_p') is not None else 1.0

        logger.info(f"Generation request - Prompt: '{prompt[:50]}...', max_length: {max_length}, temp: {temperature}")

        # Generate text
        start_time = time.time()

        generation_kwargs = {
            'max_length': max_length,
            'temperature': temperature,
            'num_return_sequences': 1,
            'do_sample': True if temperature > 0 else False
        }

        if top_k > 0:
            generation_kwargs['top_k'] = top_k
        if top_p < 1.0:
            generation_kwargs['top_p'] = top_p

        result = generator(prompt, **generation_kwargs)

        inference_time = time.time() - start_time
        generated_text = result[0]['generated_text']

        logger.info(f"Generation completed in {inference_time:.2f}s")

        # Prepare response
        response_data = {
            "success": True,
            "prompt": prompt,
            "generated_text": generated_text,
            "inference_time_seconds": round(inference_time, 3),
            "parameters": {
                "max_length": max_length,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p
            },
            "metadata": {
                "model": MODEL_NAME,
                "device": "CPU" if DEVICE == -1 else f"GPU:{DEVICE}"
            }
        }

        return jsonify(response_data), 200

    except Exception as e:
        logger.error(f"Error during generation: {str(e)}", exc_info=True)
        return jsonify({
            "error": "Internal server error during text generation",
            "details": str(e)
        }), 500


@app.errorhandler(404)
def not_found(error) -> Response:
    """Handle 404 errors."""
    return jsonify({
        "error": "Endpoint not found",
        "available_endpoints": ["/health", "/info", "/generate"]
    }), 404


@app.errorhandler(405)
def method_not_allowed(error) -> Response:
    """Handle 405 errors."""
    return jsonify({
        "error": "Method not allowed",
        "hint": "Check the HTTP method (GET/POST) for this endpoint"
    }), 405


@app.errorhandler(500)
def internal_error(error) -> Response:
    """Handle 500 errors."""
    logger.error(f"Internal server error: {str(error)}", exc_info=True)
    return jsonify({
        "error": "Internal server error",
        "details": str(error)
    }), 500


def main():
    """Main entry point for the API server."""
    # Initialize model at startup
    initialize_model()

    # Get server configuration
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', '5000'))
    debug = os.getenv('DEBUG', 'false').lower() == 'true'

    logger.info(f"Starting server on {host}:{port}")

    # Run the Flask app
    app.run(
        host=host,
        port=port,
        debug=debug,
        threaded=True  # Enable threading for concurrent requests
    )


if __name__ == '__main__':
    main()
