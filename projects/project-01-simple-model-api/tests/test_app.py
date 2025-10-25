#!/usr/bin/env python3
"""
Tests for Flask API endpoints

Run with: pytest tests/test_app.py -v
"""

import pytest
import io
import json
from PIL import Image
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from app import app, setup_model


@pytest.fixture
def client():
    """Create Flask test client"""
    app.config['TESTING'] = True
    with app.test_client() as client:
        # Initialize model
        setup_model()
        yield client


@pytest.fixture
def sample_image():
    """Create a sample test image"""
    # Create a simple RGB image
    img = Image.new('RGB', (224, 224), color='red')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    return img_bytes


class TestHealthEndpoint:
    """Tests for /health endpoint"""

    def test_health_check(self, client):
        """Test health endpoint returns 200"""
        response = client.get('/health')
        assert response.status_code == 200

        data = json.loads(response.data)
        assert data['status'] == 'healthy'
        assert 'model' in data
        assert 'device' in data
        assert 'timestamp' in data

    def test_health_check_format(self, client):
        """Test health response format"""
        response = client.get('/health')
        data = json.loads(response.data)

        assert isinstance(data['timestamp'], (int, float))
        assert data['status'] in ['healthy', 'unhealthy']


class TestInfoEndpoint:
    """Tests for /info endpoint"""

    def test_info_endpoint(self, client):
        """Test info endpoint returns model information"""
        response = client.get('/info')
        assert response.status_code == 200

        data = json.loads(response.data)
        assert data['status'] == 'success'
        assert 'model' in data
        assert 'config' in data

    def test_info_model_details(self, client):
        """Test model information is complete"""
        response = client.get('/info')
        data = json.loads(response.data)

        model_info = data['model']
        assert 'name' in model_info
        assert 'device' in model_info
        assert 'num_classes' in model_info
        assert 'total_parameters' in model_info
        assert 'framework' in model_info

    def test_info_config_details(self, client):
        """Test configuration information is present"""
        response = client.get('/info')
        data = json.loads(response.data)

        config = data['config']
        assert 'max_file_size' in config
        assert 'allowed_extensions' in config
        assert 'request_timeout' in config


class TestPredictEndpoint:
    """Tests for /predict endpoint"""

    def test_predict_success(self, client, sample_image):
        """Test successful prediction"""
        response = client.post(
            '/predict',
            data={'file': (sample_image, 'test.jpg')},
            content_type='multipart/form-data'
        )

        assert response.status_code == 200
        data = json.loads(response.data)

        assert data['status'] == 'success'
        assert 'predictions' in data
        assert 'metadata' in data
        assert len(data['predictions']) == 5  # Default top_k

    def test_predict_response_format(self, client, sample_image):
        """Test prediction response format"""
        response = client.post(
            '/predict',
            data={'file': (sample_image, 'test.jpg')},
            content_type='multipart/form-data'
        )

        data = json.loads(response.data)
        predictions = data['predictions']

        # Check first prediction format
        assert 'class' in predictions[0]
        assert 'confidence' in predictions[0]
        assert 'class_id' in predictions[0]

        # Check confidence is between 0 and 1
        assert 0 <= predictions[0]['confidence'] <= 1

    def test_predict_custom_top_k(self, client, sample_image):
        """Test prediction with custom top_k"""
        response = client.post(
            '/predict',
            data={'file': (sample_image, 'test.jpg'), 'top_k': '3'},
            content_type='multipart/form-data'
        )

        assert response.status_code == 200
        data = json.loads(response.data)
        assert len(data['predictions']) == 3

    def test_predict_no_file(self, client):
        """Test prediction without file"""
        response = client.post('/predict')
        assert response.status_code == 400

        data = json.loads(response.data)
        assert data['status'] == 'error'
        assert 'No file provided' in data['message']

    def test_predict_invalid_top_k(self, client, sample_image):
        """Test prediction with invalid top_k"""
        response = client.post(
            '/predict',
            data={'file': (sample_image, 'test.jpg'), 'top_k': '20'},
            content_type='multipart/form-data'
        )

        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'top_k' in data['message'].lower()

    def test_predict_invalid_file_type(self, client):
        """Test prediction with invalid file type"""
        # Create a text file
        text_file = io.BytesIO(b"This is not an image")

        response = client.post(
            '/predict',
            data={'file': (text_file, 'test.txt')},
            content_type='multipart/form-data'
        )

        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['status'] == 'error'

    def test_predict_empty_file(self, client):
        """Test prediction with empty file"""
        empty_file = io.BytesIO(b"")

        response = client.post(
            '/predict',
            data={'file': (empty_file, 'test.jpg')},
            content_type='multipart/form-data'
        )

        assert response.status_code == 400

    def test_predict_metadata(self, client, sample_image):
        """Test prediction metadata"""
        response = client.post(
            '/predict',
            data={'file': (sample_image, 'test.jpg')},
            content_type='multipart/form-data'
        )

        data = json.loads(response.data)
        metadata = data['metadata']

        assert 'filename' in metadata
        assert 'inference_time' in metadata
        assert 'model' in metadata
        assert 'device' in metadata
        assert 'image_size' in metadata

        # Check inference time is reasonable
        assert metadata['inference_time'] > 0
        assert metadata['inference_time'] < 10  # Should be less than 10 seconds


class TestRootEndpoint:
    """Tests for / endpoint"""

    def test_root_endpoint(self, client):
        """Test root endpoint"""
        response = client.get('/')
        assert response.status_code == 200

        data = json.loads(response.data)
        assert 'name' in data
        assert 'version' in data
        assert 'endpoints' in data
        assert 'status' in data

    def test_root_endpoints_listed(self, client):
        """Test root endpoint lists all endpoints"""
        response = client.get('/')
        data = json.loads(response.data)

        endpoints = data['endpoints']
        assert '/' in endpoints
        assert '/health' in endpoints
        assert '/info' in endpoints
        assert '/predict' in endpoints


class TestErrorHandling:
    """Tests for error handling"""

    def test_404_not_found(self, client):
        """Test 404 for non-existent endpoint"""
        response = client.get('/nonexistent')
        assert response.status_code == 404

    def test_method_not_allowed(self, client):
        """Test method not allowed"""
        response = client.post('/health')
        assert response.status_code == 405


class TestPerformance:
    """Performance tests"""

    def test_prediction_latency(self, client, sample_image):
        """Test prediction latency is acceptable"""
        import time

        start = time.time()
        response = client.post(
            '/predict',
            data={'file': (sample_image, 'test.jpg')},
            content_type='multipart/form-data'
        )
        duration = time.time() - start

        assert response.status_code == 200
        # Should complete within 5 seconds on CPU
        assert duration < 5.0

    def test_concurrent_requests(self, client, sample_image):
        """Test handling multiple concurrent requests"""
        import concurrent.futures

        def make_request():
            img = Image.new('RGB', (224, 224), color='blue')
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='JPEG')
            img_bytes.seek(0)

            return client.post(
                '/predict',
                data={'file': (img_bytes, 'test.jpg')},
                content_type='multipart/form-data'
            )

        # Send 5 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(5)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        # All should succeed
        for response in results:
            assert response.status_code == 200


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
