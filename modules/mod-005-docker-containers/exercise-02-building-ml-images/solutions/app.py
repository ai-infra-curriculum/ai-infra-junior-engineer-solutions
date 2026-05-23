#!/usr/bin/env python3
"""
Sample ML application for Docker container testing.
"""

from flask import Flask, jsonify, request
import os
import sys

app = Flask(__name__)


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'framework': get_framework_info()
    })


@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint."""
    data = request.get_json()
    return jsonify({
        'prediction': 'sample_output',
        'framework': get_framework_info()
    })


@app.route('/info')
def info():
    """System information endpoint."""
    return jsonify({
        'python_version': sys.version,
        'framework': get_framework_info(),
        'environment': dict(os.environ)
    })


def get_framework_info():
    """Get ML framework information."""
    info = {}

    # Try TensorFlow
    try:
        import tensorflow as tf
        info['tensorflow'] = tf.__version__
    except ImportError:
        pass

    # Try PyTorch
    try:
        import torch
        info['pytorch'] = torch.__version__
        info['cuda_available'] = torch.cuda.is_available()
    except ImportError:
        pass

    # Try scikit-learn
    try:
        import sklearn
        info['sklearn'] = sklearn.__version__
    except ImportError:
        pass

    return info


if __name__ == '__main__':
    port = int(os.getenv('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=False)
