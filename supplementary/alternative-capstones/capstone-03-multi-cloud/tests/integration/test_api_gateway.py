"""
Integration tests for API Gateway
Tests API endpoints and multi-cloud integration
"""

import pytest
from fastapi.testclient import TestClient
import sys
sys.path.append('../../src/api-gateway')

from main import app

client = TestClient(app)


class TestAPIEndpoints:
    """Test API Gateway endpoints"""

    def test_root_endpoint(self):
        """Test 32: Root endpoint returns correct response"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert data["version"] == "1.0.0"

    def test_health_check(self):
        """Test 33: Health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "clouds" in data
        assert "aws" in data["clouds"]
        assert "gcp" in data["clouds"]
        assert "azure" in data["clouds"]

    def test_predict_auto_cloud(self):
        """Test 34: Prediction with auto cloud selection"""
        response = client.post(
            "/predict",
            json={
                "model_name": "fraud-detection",
                "input_data": {
                    "amount": 100.0,
                    "merchant": "test"
                },
                "cloud_provider": "auto"
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "cloud_provider" in data
        assert "latency_ms" in data
        assert data["latency_ms"] < 200

    def test_predict_aws(self):
        """Test 35: Prediction with AWS"""
        response = client.post(
            "/predict",
            json={
                "model_name": "fraud-detection",
                "input_data": {"amount": 100.0},
                "cloud_provider": "aws"
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["cloud_provider"] == "aws"

    def test_predict_gcp(self):
        """Test 36: Prediction with GCP"""
        response = client.post(
            "/predict",
            json={
                "model_name": "fraud-detection",
                "input_data": {"amount": 100.0},
                "cloud_provider": "gcp"
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["cloud_provider"] == "gcp"

    def test_predict_azure(self):
        """Test 37: Prediction with Azure"""
        response = client.post(
            "/predict",
            json={
                "model_name": "fraud-detection",
                "input_data": {"amount": 100.0},
                "cloud_provider": "azure"
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["cloud_provider"] == "azure"

    def test_list_models(self):
        """Test 38: List all models across clouds"""
        response = client.get("/models")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0
        assert "cloud_provider" in data[0]

    def test_get_cloud_metrics(self):
        """Test 39: Get cloud performance metrics"""
        response = client.get("/metrics/clouds")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 3
        for metrics in data:
            assert "cloud_provider" in metrics
            assert "latency_p95" in metrics
            assert "error_rate" in metrics

    def test_deploy_model_single_cloud(self):
        """Test 40: Deploy model to single cloud"""
        response = client.post(
            "/models/test-model/deploy",
            json={
                "clouds": ["aws"],
                "model_version": "v1.0.0"
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "aws" in data
        assert data["aws"]["status"] == "success"

    def test_deploy_model_multi_cloud(self):
        """Test 41: Deploy model to multiple clouds"""
        response = client.post(
            "/models/test-model/deploy",
            json={
                "clouds": ["aws", "gcp", "azure"],
                "model_version": "v1.0.0"
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "aws" in data
        assert "gcp" in data
        assert "azure" in data

    def test_predict_missing_model_name(self):
        """Test 42: Prediction with missing model name"""
        response = client.post(
            "/predict",
            json={
                "input_data": {"amount": 100.0}
            }
        )
        assert response.status_code == 422  # Validation error

    def test_predict_missing_input_data(self):
        """Test 43: Prediction with missing input data"""
        response = client.post(
            "/predict",
            json={
                "model_name": "fraud-detection"
            }
        )
        assert response.status_code == 422  # Validation error

    def test_custom_request_id(self):
        """Test 44: Prediction with custom request ID"""
        response = client.post(
            "/predict",
            json={
                "model_name": "fraud-detection",
                "input_data": {"amount": 100.0}
            },
            headers={"X-Request-ID": "custom-id-123"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["request_id"] == "custom-id-123"

    def test_latency_threshold(self):
        """Test 45: Prediction with latency threshold"""
        response = client.post(
            "/predict",
            json={
                "model_name": "fraud-detection",
                "input_data": {"amount": 100.0},
                "latency_threshold_ms": 50
            }
        )
        assert response.status_code == 200
        data = response.json()
        # Should select fastest cloud
        assert data["latency_ms"] <= 100

    def test_model_version_specified(self):
        """Test 46: Prediction with specific model version"""
        response = client.post(
            "/predict",
            json={
                "model_name": "fraud-detection",
                "model_version": "v1.2.0",
                "input_data": {"amount": 100.0}
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "model_version" in data

    def test_prometheus_metrics_endpoint(self):
        """Test 47: Prometheus metrics endpoint"""
        response = client.get("/metrics")
        assert response.status_code == 200
        # Metrics should be in Prometheus format
        assert b"api_requests_total" in response.content


class TestConcurrentRequests:
    """Test concurrent request handling"""

    def test_concurrent_predictions(self):
        """Test 48: Handle concurrent predictions"""
        import concurrent.futures

        def make_prediction():
            return client.post(
                "/predict",
                json={
                    "model_name": "fraud-detection",
                    "input_data": {"amount": 100.0}
                }
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_prediction) for _ in range(10)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert all(r.status_code == 200 for r in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
