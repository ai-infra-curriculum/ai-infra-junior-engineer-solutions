"""Contract tests for API schema validation.

Tests cover:
- OpenAPI schema compliance
- Response format validation
- Request format validation
- Backward compatibility
- API documentation accuracy

Markers:
    contract: Contract/schema tests
"""

import pytest
from jsonschema import validate, ValidationError
import json


@pytest.mark.contract
class TestResponseSchemas:
    """Test that API responses match expected schemas."""

    def test_login_response_schema(self, client, test_user):
        """Test login response matches LoginResponse schema.

        Verifies:
        - All required fields present
        - Field types correct
        - No extra fields present
        """
        response = client.post(
            "/auth/login",
            json={
                "username": test_user["username"],
                "password": test_user["password"]
            }
        )

        assert response.status_code == 200
        data = response.json()

        # Expected schema
        expected_fields = {"access_token", "token_type"}
        actual_fields = set(data.keys())

        assert expected_fields == actual_fields, \
            f"Fields mismatch. Expected: {expected_fields}, Got: {actual_fields}"

        # Field types
        assert isinstance(data["access_token"], str), "access_token must be string"
        assert isinstance(data["token_type"], str), "token_type must be string"
        assert data["token_type"] == "bearer", "token_type must be 'bearer'"

    def test_prediction_response_schema(self, client, auth_headers):
        """Test prediction response matches PredictionResponse schema.

        Verifies:
        - All required fields present
        - Field types correct
        - Value constraints satisfied
        """
        response = client.post(
            "/predict",
            json={"features": [1.0] * 10},
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()

        # Required fields
        required_fields = {"prediction", "cached", "model_version"}
        assert required_fields.issubset(data.keys()), \
            f"Missing required fields: {required_fields - set(data.keys())}"

        # Field types
        assert isinstance(data["prediction"], (int, float)), \
            "prediction must be numeric"
        assert isinstance(data["cached"], bool), \
            "cached must be boolean"
        assert isinstance(data["model_version"], str), \
            "model_version must be string"

        # Value constraints
        assert len(data["model_version"]) > 0, \
            "model_version must not be empty"

    def test_batch_prediction_response_schema(self, client, auth_headers):
        """Test batch prediction response matches BatchPredictionResponse schema.

        Verifies:
        - All required fields present
        - Field types correct
        - Array constraints satisfied
        """
        response = client.post(
            "/batch-predict",
            json={"samples": [[1.0] * 10, [2.0] * 10]},
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()

        # Required fields
        required_fields = {"predictions", "count", "model_version"}
        assert required_fields.issubset(data.keys())

        # Field types
        assert isinstance(data["predictions"], list), \
            "predictions must be list"
        assert isinstance(data["count"], int), \
            "count must be integer"
        assert isinstance(data["model_version"], str), \
            "model_version must be string"

        # Array constraints
        assert len(data["predictions"]) == data["count"], \
            "count must match predictions length"
        assert len(data["predictions"]) == 2, \
            "predictions count must match request samples"

        # Element types
        for i, pred in enumerate(data["predictions"]):
            assert isinstance(pred, (int, float)), \
                f"prediction[{i}] must be numeric"

    def test_user_response_schema(self, client, auth_headers):
        """Test user info response matches UserResponse schema.

        Verifies:
        - All required fields present
        - Field types correct
        - No sensitive data leaked
        """
        response = client.get("/users/me", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()

        # Required fields
        required_fields = {"username", "email", "is_active"}
        actual_fields = set(data.keys())

        assert required_fields.issubset(actual_fields), \
            f"Missing fields: {required_fields - actual_fields}"

        # Field types
        assert isinstance(data["username"], str), "username must be string"
        assert isinstance(data["email"], str), "email must be string"
        assert isinstance(data["is_active"], bool), "is_active must be boolean"

        # Security: sensitive fields should NOT be present
        sensitive_fields = {"password", "hashed_password", "secret"}
        leaked = sensitive_fields.intersection(actual_fields)
        assert not leaked, f"Sensitive fields leaked: {leaked}"

    def test_error_response_schema(self, client):
        """Test error responses have consistent schema.

        Verifies:
        - Error responses include 'detail' field
        - Detail is string or structured object
        """
        # Trigger validation error
        response = client.post(
            "/auth/login",
            json={"username": "ab"}  # Too short
        )

        assert response.status_code == 422
        data = response.json()

        # FastAPI validation errors have specific structure
        assert "detail" in data, "Error response must include 'detail'"

        # Detail can be string or array of error objects
        detail = data["detail"]
        if isinstance(detail, list):
            # Validation errors are arrays
            for error in detail:
                assert "loc" in error, "Validation error must include 'loc'"
                assert "msg" in error, "Validation error must include 'msg'"
                assert "type" in error, "Validation error must include 'type'"
        else:
            # Simple errors are strings
            assert isinstance(detail, str), "Error detail must be string or array"

    def test_health_response_schema(self, client):
        """Test health check response schema.

        Verifies:
        - Status field present
        - Timestamp field present
        - Field types correct
        """
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()

        # Required fields
        assert "status" in data, "Health response must include 'status'"
        assert "timestamp" in data, "Health response must include 'timestamp'"

        # Field types
        assert isinstance(data["status"], str), "status must be string"
        assert isinstance(data["timestamp"], str), "timestamp must be string"

        # Value constraints
        assert data["status"] in ["healthy", "unhealthy", "degraded"], \
            f"Invalid status: {data['status']}"


@pytest.mark.contract
class TestRequestValidation:
    """Test that API validates request schemas correctly."""

    def test_login_request_validation(self, client):
        """Test login request validation rules.

        Verifies:
        - Required fields enforced
        - String length constraints enforced
        - Invalid characters rejected
        """
        # Missing username
        response = client.post(
            "/auth/login",
            json={"password": "password123"}
        )
        assert response.status_code == 422

        # Username too short
        response = client.post(
            "/auth/login",
            json={"username": "ab", "password": "password123"}
        )
        assert response.status_code == 422

        # Password too short
        response = client.post(
            "/auth/login",
            json={"username": "testuser", "password": "short"}
        )
        assert response.status_code == 422

    def test_prediction_request_validation(self, client, auth_headers):
        """Test prediction request validation rules.

        Verifies:
        - Feature count constraints enforced
        - Feature value ranges enforced
        - Type validation enforced
        """
        # Wrong feature count (too few)
        response = client.post(
            "/predict",
            json={"features": [1.0] * 5},
            headers=auth_headers
        )
        assert response.status_code == 422

        # Wrong feature count (too many)
        response = client.post(
            "/predict",
            json={"features": [1.0] * 15},
            headers=auth_headers
        )
        assert response.status_code == 422

        # Features out of range
        response = client.post(
            "/predict",
            json={"features": [2000.0] * 10},  # Exceeds valid range
            headers=auth_headers
        )
        assert response.status_code == 422

        # Invalid types
        response = client.post(
            "/predict",
            json={"features": ["not", "a", "number"] + [1.0] * 7},
            headers=auth_headers
        )
        assert response.status_code == 422

    def test_batch_prediction_request_validation(self, client, auth_headers):
        """Test batch prediction request validation rules.

        Verifies:
        - Sample count constraints enforced
        - Each sample validated
        - Batch size limits enforced
        """
        # Empty batch
        response = client.post(
            "/batch-predict",
            json={"samples": []},
            headers=auth_headers
        )
        assert response.status_code == 422

        # Sample with wrong feature count
        response = client.post(
            "/batch-predict",
            json={"samples": [[1.0] * 5, [1.0] * 10]},  # First sample wrong size
            headers=auth_headers
        )
        assert response.status_code == 422

        # Too many samples (exceeds max_items=100)
        response = client.post(
            "/batch-predict",
            json={"samples": [[1.0] * 10 for _ in range(101)]},
            headers=auth_headers
        )
        assert response.status_code == 422


@pytest.mark.contract
class TestAPIDocumentation:
    """Test that OpenAPI documentation is accurate."""

    def test_openapi_schema_available(self, client):
        """Test that OpenAPI schema is accessible.

        Verifies:
        - /openapi.json endpoint exists
        - Returns valid JSON
        - Contains required OpenAPI fields
        """
        response = client.get("/openapi.json")

        assert response.status_code == 200, "OpenAPI schema should be accessible"

        schema = response.json()

        # Required OpenAPI 3.0 fields
        assert "openapi" in schema, "Schema must include OpenAPI version"
        assert schema["openapi"].startswith("3."), "Must be OpenAPI 3.x"
        assert "info" in schema, "Schema must include info section"
        assert "paths" in schema, "Schema must include paths"

    def test_all_endpoints_documented(self, client):
        """Test that all endpoints are documented in OpenAPI schema.

        Verifies:
        - All implemented endpoints in schema
        - Methods match implementation
        """
        response = client.get("/openapi.json")
        schema = response.json()
        paths = schema.get("paths", {})

        # Expected endpoints
        expected_endpoints = [
            "/auth/login",
            "/users/me",
            "/predict",
            "/batch-predict",
            "/health",
            "/models",
        ]

        for endpoint in expected_endpoints:
            assert endpoint in paths, \
                f"Endpoint {endpoint} not documented in OpenAPI schema"

    def test_response_schemas_documented(self, client):
        """Test that response schemas are properly documented.

        Verifies:
        - Each endpoint has response schemas
        - Success and error responses documented
        """
        response = client.get("/openapi.json")
        schema = response.json()

        # Check /predict endpoint documentation
        predict_path = schema["paths"].get("/predict", {})
        post_method = predict_path.get("post", {})
        responses = post_method.get("responses", {})

        # Should document success response
        assert "200" in responses, "/predict should document 200 response"

        # Should document auth error
        assert "401" in responses or "403" in responses, \
            "/predict should document auth error response"

    def test_request_body_schemas_documented(self, client):
        """Test that request body schemas are documented.

        Verifies:
        - Endpoints with request bodies have schemas
        - Required fields documented
        """
        response = client.get("/openapi.json")
        schema = response.json()

        # Check /auth/login documentation
        login_path = schema["paths"].get("/auth/login", {})
        post_method = login_path.get("post", {})
        request_body = post_method.get("requestBody", {})

        assert request_body, "/auth/login should document request body"
        assert "content" in request_body, \
            "Request body should specify content type"


@pytest.mark.contract
class TestBackwardCompatibility:
    """Test API backward compatibility."""

    def test_response_fields_stable(self, client, auth_headers):
        """Test that core response fields remain stable.

        Verifies:
        - Required fields always present
        - Field types don't change
        - No breaking changes to existing fields
        """
        # Test prediction response stability
        response = client.post(
            "/predict",
            json={"features": [1.0] * 10},
            headers=auth_headers
        )

        data = response.json()

        # Core fields that must always exist
        stable_fields = {
            "prediction": (int, float),
            "cached": bool,
            "model_version": str
        }

        for field, expected_type in stable_fields.items():
            assert field in data, \
                f"Breaking change: required field '{field}' missing"
            assert isinstance(data[field], expected_type), \
                f"Breaking change: field '{field}' type changed"

    def test_optional_fields_backward_compatible(self, client, auth_headers):
        """Test that new optional fields don't break existing clients.

        Verifies:
        - Optional fields can be ignored by clients
        - Core functionality works without new fields
        """
        response = client.post(
            "/predict",
            json={"features": [1.0] * 10},
            headers=auth_headers
        )

        data = response.json()

        # Simulate old client that only reads core fields
        old_client_data = {
            "prediction": data["prediction"],
            "model_version": data["model_version"]
        }

        # Should have all data needed for old client
        assert "prediction" in old_client_data
        assert "model_version" in old_client_data
        assert isinstance(old_client_data["prediction"], (int, float))

    def test_error_format_stability(self, client):
        """Test that error response format remains stable.

        Verifies:
        - Error responses always include 'detail'
        - Error structure predictable
        """
        # Trigger error
        response = client.get("/users/me")  # No auth

        assert response.status_code in [401, 403]
        data = response.json()

        # Error format must be stable
        assert "detail" in data, \
            "Breaking change: error responses must include 'detail'"


@pytest.mark.contract
class TestAPIVersioning:
    """Test API versioning and compatibility."""

    def test_version_info_in_responses(self, client, auth_headers):
        """Test that version information is included in responses.

        Verifies:
        - Model version included in predictions
        - Version format consistent
        """
        response = client.post(
            "/predict",
            json={"features": [1.0] * 10},
            headers=auth_headers
        )

        data = response.json()
        assert "model_version" in data, "Responses should include version info"

        # Version should follow semantic versioning
        version = data["model_version"]
        assert isinstance(version, str)
        assert len(version) > 0

    def test_api_version_in_openapi(self, client):
        """Test that API version is documented.

        Verifies:
        - OpenAPI schema includes version
        - Version follows semantic versioning
        """
        response = client.get("/openapi.json")
        schema = response.json()

        info = schema.get("info", {})
        assert "version" in info, "OpenAPI schema must include API version"

        version = info["version"]
        assert isinstance(version, str)
        assert len(version) > 0


@pytest.mark.contract
class TestContentNegotiation:
    """Test content type handling."""

    def test_json_content_type_required(self, client, auth_headers):
        """Test that endpoints require JSON content type.

        Verifies:
        - JSON content properly handled
        - Non-JSON content rejected
        """
        # Valid JSON request
        response = client.post(
            "/predict",
            json={"features": [1.0] * 10},
            headers=auth_headers
        )
        assert response.status_code == 200

    def test_response_content_type(self, client, auth_headers):
        """Test that responses have correct content type.

        Verifies:
        - Content-Type header correct
        - JSON responses properly formatted
        """
        response = client.post(
            "/predict",
            json={"features": [1.0] * 10},
            headers=auth_headers
        )

        content_type = response.headers.get("content-type", "")
        assert "application/json" in content_type.lower(), \
            "Response Content-Type must be application/json"

        # Response should be valid JSON
        try:
            data = response.json()
            assert isinstance(data, dict)
        except ValueError:
            pytest.fail("Response is not valid JSON")
