"""Flask-based ML model serving API.

This is the legacy Flask implementation that we're migrating from.
Demonstrates traditional Flask patterns with manual validation.

Pain points addressed in FastAPI version:
- Manual request validation
- No automatic documentation
- Synchronous request handling
- Manual error handling
- Type hints not enforced at runtime
"""

from flask import Flask, request, jsonify
from functools import wraps
import time
import jwt
import pickle
import numpy as np
from datetime import datetime, timedelta
import os
import sys

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-change-in-production')

# Global variables
model = None
prediction_cache = {}
cache_stats = {
    'hits': 0,
    'misses': 0,
    'total': 0
}
start_time = time.time()


def load_model():
    """Load the ML model at startup."""
    global model
    model_path = os.getenv('MODEL_PATH', 'model.pkl')

    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"✅ Model loaded successfully from {model_path}")
        return True
    except FileNotFoundError:
        print(f"⚠️  Warning: Model file not found at {model_path}")
        print("   Creating dummy model for demonstration...")
        # Create dummy model for testing
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        # Train on dummy data
        X_dummy = np.random.rand(100, 10)
        y_dummy = np.random.rand(100)
        model.fit(X_dummy, y_dummy)
        print("✅ Dummy model created")
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False


# Load model at startup
load_model()


# Authentication decorator
def token_required(f):
    """Decorator to require JWT authentication."""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')

        if not token:
            return jsonify({'error': 'Token is missing'}), 401

        try:
            # Remove 'Bearer ' prefix if present
            if token.startswith('Bearer '):
                token = token[7:]

            # Decode token
            data = jwt.decode(
                token,
                app.config['SECRET_KEY'],
                algorithms=["HS256"]
            )

            # Token is valid, continue
            return f(*args, **kwargs)

        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token has expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Token is invalid'}), 401
        except Exception as e:
            return jsonify({'error': f'Authentication error: {str(e)}'}), 401

    return decorated


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint.

    Returns:
        JSON response with status and timestamp
    """
    uptime = time.time() - start_time

    return jsonify({
        'status': 'healthy' if model is not None else 'unhealthy',
        'timestamp': datetime.utcnow().isoformat(),
        'uptime_seconds': uptime,
        'model_loaded': model is not None
    })


@app.route('/predict', methods=['POST'])
@token_required
def predict():
    """Make predictions on input features.

    Expects:
        JSON with 'features' key containing list of 10 floats

    Returns:
        JSON with prediction, cache status, and model version
    """
    try:
        # Get request data
        data = request.get_json()

        # Validate input - MANUAL VALIDATION (pain point #1)
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        if 'features' not in data:
            return jsonify({'error': 'Missing features field'}), 400

        features = data['features']

        # More manual validation (pain point #2)
        if not isinstance(features, list):
            return jsonify({'error': 'Features must be a list'}), 400

        if len(features) != 10:
            return jsonify({
                'error': f'Expected 10 features, got {len(features)}'
            }), 400

        # Validate feature values (pain point #3)
        for i, feature in enumerate(features):
            if not isinstance(feature, (int, float)):
                return jsonify({
                    'error': f'Feature {i} must be a number'
                }), 400

            if not (-1000 <= feature <= 1000):
                return jsonify({
                    'error': f'Feature {i} out of valid range'
                }), 400

        # Check cache
        feature_key = str(features)
        if feature_key in prediction_cache:
            cache_stats['hits'] += 1
            cache_stats['total'] += 1

            response = {
                'prediction': prediction_cache[feature_key],
                'cached': True,
                'model_version': '1.0'
            }

            # Add request_id if provided
            if 'request_id' in data:
                response['request_id'] = data['request_id']

            return jsonify(response)

        # Make prediction (synchronous - pain point #4)
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 503

        features_array = np.array([features])
        prediction = model.predict(features_array)[0]

        # Cache result
        prediction_cache[feature_key] = float(prediction)
        cache_stats['misses'] += 1
        cache_stats['total'] += 1

        response = {
            'prediction': float(prediction),
            'cached': False,
            'model_version': '1.0'
        }

        # Add request_id if provided
        if 'request_id' in data:
            response['request_id'] = data['request_id']

        return jsonify(response)

    except Exception as e:
        # Generic error handling (pain point #5)
        return jsonify({'error': str(e)}), 500


@app.route('/batch-predict', methods=['POST'])
@token_required
def batch_predict():
    """Make predictions on multiple samples.

    Expects:
        JSON with 'samples' key containing list of feature vectors

    Returns:
        JSON with predictions, count, and model version
    """
    try:
        start = time.time()
        data = request.get_json()

        # Manual validation (pain point #1)
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        if 'samples' not in data:
            return jsonify({'error': 'Missing samples field'}), 400

        samples = data['samples']

        # More manual validation (pain point #2)
        if not isinstance(samples, list):
            return jsonify({'error': 'Samples must be a list'}), 400

        if len(samples) == 0:
            return jsonify({'error': 'Samples list cannot be empty'}), 400

        if len(samples) > 100:
            return jsonify({'error': 'Maximum 100 samples allowed'}), 400

        # Validate each sample (pain point #3)
        for i, sample in enumerate(samples):
            if not isinstance(sample, list):
                return jsonify({
                    'error': f'Sample {i} must be a list'
                }), 400

            if len(sample) != 10:
                return jsonify({
                    'error': f'Sample {i} must have 10 features, got {len(sample)}'
                }), 400

            # Validate each feature
            for j, feature in enumerate(sample):
                if not isinstance(feature, (int, float)):
                    return jsonify({
                        'error': f'Sample {i}, feature {j} must be a number'
                    }), 400

                if not (-1000 <= feature <= 1000):
                    return jsonify({
                        'error': f'Sample {i}, feature {j} out of valid range'
                    }), 400

        # Make predictions (synchronous - pain point #4)
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 503

        features_array = np.array(samples)
        predictions = model.predict(features_array)

        processing_time = (time.time() - start) * 1000  # milliseconds

        response = {
            'predictions': [float(p) for p in predictions],
            'count': len(predictions),
            'model_version': '1.0',
            'processing_time_ms': processing_time
        }

        # Add request_id if provided
        if 'request_id' in data:
            response['request_id'] = data['request_id']

        return jsonify(response)

    except Exception as e:
        # Generic error handling (pain point #5)
        return jsonify({'error': str(e)}), 500


@app.route('/model-info', methods=['GET'])
def model_info():
    """Get information about the model.

    Returns:
        JSON with model metadata
    """
    return jsonify({
        'version': '1.0',
        'input_features': 10,
        'model_type': 'random_forest',
        'training_date': '2024-01-15',
        'accuracy': 0.94,
        'framework': 'scikit-learn'
    })


@app.route('/login', methods=['POST'])
def login():
    """Generate authentication token.

    Expects:
        JSON with 'username' and 'password'

    Returns:
        JSON with JWT token
    """
    data = request.get_json()

    # Manual validation (pain point #1)
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    if 'username' not in data or 'password' not in data:
        return jsonify({'error': 'Missing credentials'}), 400

    username = data['username']
    password = data['password']

    # Validate lengths
    if len(username) < 3 or len(username) > 50:
        return jsonify({'error': 'Username must be 3-50 characters'}), 400

    if len(password) < 8:
        return jsonify({'error': 'Password must be at least 8 characters'}), 400

    # Simplified auth (in production, validate against database with hashed passwords)
    if username == 'admin' and password == 'password':
        # Create token
        token = jwt.encode(
            {
                'user': username,
                'exp': datetime.utcnow() + timedelta(hours=24)
            },
            app.config['SECRET_KEY'],
            algorithm="HS256"
        )

        return jsonify({
            'token': token,
            'token_type': 'bearer',
            'expires_in': 86400  # 24 hours in seconds
        })

    return jsonify({'error': 'Invalid credentials'}), 401


@app.route('/cache/stats', methods=['GET'])
@token_required
def cache_stats_endpoint():
    """Get cache statistics.

    Returns:
        JSON with cache metrics
    """
    total = cache_stats['total']
    hit_rate = cache_stats['hits'] / total if total > 0 else 0.0

    return jsonify({
        'size': len(prediction_cache),
        'hit_rate': hit_rate,
        'total_requests': total,
        'cache_hits': cache_stats['hits'],
        'cache_misses': cache_stats['misses']
    })


@app.route('/cache/clear', methods=['POST'])
@token_required
def clear_cache():
    """Clear the prediction cache.

    Returns:
        JSON with success message
    """
    global prediction_cache, cache_stats

    entries_cleared = len(prediction_cache)
    prediction_cache.clear()

    return jsonify({
        'message': 'Cache cleared successfully',
        'entries_cleared': entries_cleared
    })


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        'error': 'Endpoint not found',
        'detail': str(error)
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({
        'error': 'Internal server error',
        'detail': str(error)
    }), 500


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('DEBUG', 'True').lower() == 'true'

    print(f"🚀 Starting Flask ML API on port {port}")
    print(f"   Debug mode: {debug}")
    print(f"   Model loaded: {model is not None}")
    print(f"   Endpoints available:")
    print(f"     - POST /login")
    print(f"     - GET  /health")
    print(f"     - POST /predict (requires auth)")
    print(f"     - POST /batch-predict (requires auth)")
    print(f"     - GET  /model-info")
    print(f"     - GET  /cache/stats (requires auth)")
    print(f"     - POST /cache/clear (requires auth)")
    print()

    app.run(
        debug=debug,
        host='0.0.0.0',
        port=port
    )
