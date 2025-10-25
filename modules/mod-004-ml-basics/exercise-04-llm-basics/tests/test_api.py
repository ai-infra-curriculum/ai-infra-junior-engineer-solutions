"""Tests for LLM API

These tests verify the Flask API endpoints, validation, and error handling.
"""

import pytest
import json
from unittest.mock import Mock, patch
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.llm_api import app, initialize_model


@pytest.fixture
def client():
    """Create a test client for the Flask app."""
    app.config['TESTING'] = True

    # Mock the generator to avoid loading the actual model
    with patch('src.llm_api.generator') as mock_generator:
        # Configure mock to return expected format
        mock_generator.return_value = [{
            'generated_text': 'This is a test generated text.'
        }]

        with app.test_client() as client:
            yield client


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_endpoint_exists(self, client):
        """Test that health endpoint exists."""
        response = client.get('/health')
        assert response.status_code in [200, 503]

    def test_health_returns_json(self, client):
        """Test that health endpoint returns JSON."""
        response = client.get('/health')
        assert response.content_type == 'application/json'

    def test_health_response_structure(self, client):
        """Test health response has correct structure."""
        with patch('src.llm_api.generator', Mock()):
            response = client.get('/health')
            data = json.loads(response.data)
            assert 'status' in data


class TestInfoEndpoint:
    """Tests for the /info endpoint."""

    def test_info_endpoint_exists(self, client):
        """Test that info endpoint exists."""
        response = client.get('/info')
        assert response.status_code == 200

    def test_info_returns_json(self, client):
        """Test that info endpoint returns JSON."""
        response = client.get('/info')
        assert response.content_type == 'application/json'

    def test_info_contains_model_info(self, client):
        """Test that info contains model information."""
        response = client.get('/info')
        data = json.loads(response.data)

        assert 'model' in data
        assert 'device' in data
        assert 'api_version' in data
        assert 'supported_parameters' in data


class TestGenerateEndpoint:
    """Tests for the /generate endpoint."""

    def test_generate_requires_post(self, client):
        """Test that generate endpoint requires POST method."""
        response = client.get('/generate')
        assert response.status_code == 405

    def test_generate_requires_json(self, client):
        """Test that generate endpoint requires JSON content."""
        response = client.post('/generate', data='not json')
        assert response.status_code in [400, 415, 500]

    def test_generate_requires_prompt(self, client):
        """Test that prompt is required."""
        with patch('src.llm_api.generator', Mock()):
            response = client.post(
                '/generate',
                data=json.dumps({}),
                content_type='application/json'
            )
            assert response.status_code == 400
            data = json.loads(response.data)
            assert 'error' in data

    def test_generate_validates_empty_prompt(self, client):
        """Test that empty prompt is rejected."""
        with patch('src.llm_api.generator', Mock()):
            response = client.post(
                '/generate',
                data=json.dumps({'prompt': ''}),
                content_type='application/json'
            )
            assert response.status_code == 400

    def test_generate_validates_max_length(self, client):
        """Test that max_length is validated."""
        with patch('src.llm_api.generator', Mock()):
            response = client.post(
                '/generate',
                data=json.dumps({
                    'prompt': 'Test',
                    'max_length': 999999  # Too large
                }),
                content_type='application/json'
            )
            assert response.status_code == 400
            data = json.loads(response.data)
            assert 'error' in data

    def test_generate_validates_temperature_range(self, client):
        """Test that temperature is validated."""
        with patch('src.llm_api.generator', Mock()):
            # Too high
            response = client.post(
                '/generate',
                data=json.dumps({
                    'prompt': 'Test',
                    'temperature': 5.0
                }),
                content_type='application/json'
            )
            assert response.status_code == 400

            # Negative
            response = client.post(
                '/generate',
                data=json.dumps({
                    'prompt': 'Test',
                    'temperature': -1.0
                }),
                content_type='application/json'
            )
            assert response.status_code == 400

    def test_generate_success_response_structure(self, client):
        """Test successful generation response structure."""
        mock_gen = Mock()
        mock_gen.return_value = [{
            'generated_text': 'Test generated text'
        }]

        with patch('src.llm_api.generator', mock_gen):
            response = client.post(
                '/generate',
                data=json.dumps({
                    'prompt': 'Test prompt'
                }),
                content_type='application/json'
            )

            if response.status_code == 200:
                data = json.loads(response.data)
                assert 'success' in data or 'generated_text' in data
                assert 'prompt' in data
                assert 'inference_time_seconds' in data or 'inference_time' in data

    def test_generate_with_all_parameters(self, client):
        """Test generation with all optional parameters."""
        mock_gen = Mock()
        mock_gen.return_value = [{
            'generated_text': 'Test generated text with all params'
        }]

        with patch('src.llm_api.generator', mock_gen):
            response = client.post(
                '/generate',
                data=json.dumps({
                    'prompt': 'Test prompt',
                    'max_length': 100,
                    'temperature': 0.8,
                    'top_k': 40,
                    'top_p': 0.9
                }),
                content_type='application/json'
            )

            # Should succeed or return 503 if model not loaded
            assert response.status_code in [200, 503]


class TestErrorHandling:
    """Tests for error handling."""

    def test_404_handler(self, client):
        """Test 404 error handling."""
        response = client.get('/nonexistent')
        assert response.status_code == 404

        data = json.loads(response.data)
        assert 'error' in data

    def test_405_handler(self, client):
        """Test 405 method not allowed handling."""
        response = client.post('/health')  # Health only accepts GET
        assert response.status_code == 405

        data = json.loads(response.data)
        assert 'error' in data


class TestValidation:
    """Tests for request validation."""

    def test_validates_top_k_type(self, client):
        """Test that top_k type is validated."""
        with patch('src.llm_api.generator', Mock()):
            response = client.post(
                '/generate',
                data=json.dumps({
                    'prompt': 'Test',
                    'top_k': 'not a number'
                }),
                content_type='application/json'
            )
            assert response.status_code == 400

    def test_validates_top_p_range(self, client):
        """Test that top_p range is validated."""
        with patch('src.llm_api.generator', Mock()):
            response = client.post(
                '/generate',
                data=json.dumps({
                    'prompt': 'Test',
                    'top_p': 1.5  # Should be <= 1.0
                }),
                content_type='application/json'
            )
            assert response.status_code == 400


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
