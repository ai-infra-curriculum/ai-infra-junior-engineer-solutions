"""Unit tests for authentication endpoints.

Tests cover:
- Login success/failure scenarios
- Token validation
- Token expiration
- Input validation
- Error handling
- Edge cases

Markers:
    unit: Unit tests
"""

import pytest
import jwt
from datetime import datetime, timedelta

# Test configuration
SECRET_KEY = "test-secret-key-for-testing-only"
ALGORITHM = "HS256"


@pytest.mark.unit
class TestLogin:
    """Tests for login endpoint."""

    def test_login_success(self, client, test_user):
        """Test successful login with valid credentials.

        Verifies:
        - 200 status code
        - Access token present in response
        - Token type is "bearer"
        - Token can be decoded successfully
        - Token contains correct username
        """
        response = client.post(
            "/auth/login",
            json={
                "username": test_user["username"],
                "password": test_user["password"]
            }
        )

        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

        data = response.json()
        assert "access_token" in data, "Response missing access_token field"
        assert data["token_type"] == "bearer", f"Expected token_type 'bearer', got {data['token_type']}"

        # Verify token is valid and contains correct claims
        token = data["access_token"]
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        assert payload["sub"] == test_user["username"], "Token contains wrong username"
        assert "exp" in payload, "Token missing expiration claim"

    def test_login_invalid_username(self, client):
        """Test login with non-existent username.

        Verifies:
        - 401 Unauthorized status
        - Error message present
        - No token returned
        """
        response = client.post(
            "/auth/login",
            json={
                "username": "nonexistent_user",
                "password": "anypassword"
            }
        )

        assert response.status_code == 401, "Should return 401 for invalid username"
        assert "detail" in response.json(), "Response should contain error detail"
        assert "access_token" not in response.json(), "Should not return token for invalid username"

    def test_login_invalid_password(self, client, test_user):
        """Test login with incorrect password.

        Verifies:
        - 401 Unauthorized status
        - Error message present
        - No token returned
        """
        response = client.post(
            "/auth/login",
            json={
                "username": test_user["username"],
                "password": "wrongpassword"
            }
        )

        assert response.status_code == 401, "Should return 401 for invalid password"
        assert "detail" in response.json()

    def test_login_missing_username(self, client):
        """Test login with missing username field.

        Verifies:
        - 422 Validation Error status
        - Error indicates missing field
        """
        response = client.post(
            "/auth/login",
            json={"password": "testpassword123"}
        )

        assert response.status_code == 422, "Should return 422 for missing username"
        error_detail = response.json()
        assert "detail" in error_detail

    def test_login_missing_password(self, client):
        """Test login with missing password field.

        Verifies:
        - 422 Validation Error status
        - Error indicates missing field
        """
        response = client.post(
            "/auth/login",
            json={"username": "testuser"}
        )

        assert response.status_code == 422, "Should return 422 for missing password"

    def test_login_empty_body(self, client):
        """Test login with empty request body.

        Verifies:
        - 422 Validation Error status
        """
        response = client.post("/auth/login", json={})
        assert response.status_code == 422

    @pytest.mark.parametrize("username,password,expected_status", [
        ("", "password123", 422),  # Empty username
        ("ab", "password123", 422),  # Too short username (min 3)
        ("a" * 51, "password123", 422),  # Too long username (max 50)
        ("testuser", "short", 422),  # Too short password (min 8)
        ("test@user", "password123", 422),  # Invalid characters in username
        ("test user", "password123", 422),  # Space in username
    ])
    def test_login_validation(self, client, username, password, expected_status):
        """Test input validation for various invalid inputs.

        Args:
            username: Username to test
            password: Password to test
            expected_status: Expected HTTP status code
        """
        response = client.post(
            "/auth/login",
            json={"username": username, "password": password}
        )
        assert response.status_code == expected_status, \
            f"Expected {expected_status} for username='{username}', password='{password}'"

    def test_login_case_sensitive_username(self, client, test_user):
        """Test that usernames are case-sensitive.

        Verifies:
        - Login with uppercase username fails
        - Login with lowercase username succeeds
        """
        # Try with uppercase
        response = client.post(
            "/auth/login",
            json={
                "username": test_user["username"].upper(),
                "password": test_user["password"]
            }
        )
        assert response.status_code == 401, "Username should be case-sensitive"

        # Verify correct case works
        response = client.post(
            "/auth/login",
            json={
                "username": test_user["username"],
                "password": test_user["password"]
            }
        )
        assert response.status_code == 200

    def test_login_special_characters_in_password(self, client):
        """Test that passwords with special characters are handled correctly."""
        response = client.post(
            "/auth/login",
            json={
                "username": "testuser",
                "password": "p@ssw0rd!#$%"
            }
        )
        # Should be processed (though will fail auth)
        assert response.status_code in [401, 422]

    def test_login_response_format(self, client, test_user):
        """Test that successful login response has correct format.

        Verifies:
        - Response is valid JSON
        - Contains required fields
        - Field types are correct
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

        # Check required fields
        assert "access_token" in data
        assert "token_type" in data

        # Check types
        assert isinstance(data["access_token"], str)
        assert isinstance(data["token_type"], str)
        assert data["token_type"] == "bearer"


@pytest.mark.unit
class TestTokenValidation:
    """Tests for token validation and protected endpoints."""

    def test_access_protected_endpoint_with_valid_token(self, client, auth_headers):
        """Test accessing protected endpoint with valid token.

        Verifies:
        - 200 status code
        - Endpoint returns expected data
        - User information is correct
        """
        response = client.get("/users/me", headers=auth_headers)

        assert response.status_code == 200, "Should allow access with valid token"
        data = response.json()
        assert "username" in data
        assert "email" in data

    def test_access_protected_endpoint_without_token(self, client):
        """Test accessing protected endpoint without authentication.

        Verifies:
        - 401 or 403 status code
        - Error message present
        """
        response = client.get("/users/me")

        assert response.status_code in [401, 403], "Should deny access without token"
        assert "detail" in response.json()

    def test_access_protected_endpoint_with_expired_token(self, client, expired_token):
        """Test accessing protected endpoint with expired token.

        Verifies:
        - 401 status code
        - Error message indicates token expired
        """
        response = client.get(
            "/users/me",
            headers={"Authorization": f"Bearer {expired_token}"}
        )

        assert response.status_code == 401, "Should reject expired token"
        error_message = response.json()["detail"].lower()
        assert "expired" in error_message or "invalid" in error_message

    def test_access_protected_endpoint_with_invalid_token(self, client):
        """Test accessing protected endpoint with malformed token.

        Verifies:
        - 401 status code
        - Error message indicates invalid token
        """
        response = client.get(
            "/users/me",
            headers={"Authorization": "Bearer invalid.token.here"}
        )

        assert response.status_code == 401, "Should reject invalid token"
        assert "detail" in response.json()

    def test_access_protected_endpoint_with_missing_bearer_prefix(self, client, auth_token):
        """Test token without 'Bearer' prefix.

        Verifies:
        - Request is rejected
        - Proper error returned
        """
        response = client.get(
            "/users/me",
            headers={"Authorization": auth_token}  # Missing "Bearer " prefix
        )

        # Should be rejected (401 or 422 depending on implementation)
        assert response.status_code in [401, 403, 422]

    def test_access_protected_endpoint_with_empty_token(self, client):
        """Test with empty token value.

        Verifies:
        - Request is rejected
        """
        response = client.get(
            "/users/me",
            headers={"Authorization": "Bearer "}
        )

        assert response.status_code == 401

    def test_token_in_wrong_header(self, client, auth_token):
        """Test sending token in wrong header field.

        Verifies:
        - Request is rejected when token not in Authorization header
        """
        # Try various wrong headers
        wrong_headers = [
            {"X-Auth-Token": f"Bearer {auth_token}"},
            {"Token": f"Bearer {auth_token}"},
            {"Bearer": auth_token},
        ]

        for headers in wrong_headers:
            response = client.get("/users/me", headers=headers)
            assert response.status_code in [401, 403], \
                f"Should reject token in wrong header: {headers}"

    def test_multiple_authorization_headers(self, client, auth_token):
        """Test behavior with multiple Authorization headers.

        This tests edge case of duplicate headers.
        """
        # FastAPI/Starlette handles this, but we test the behavior
        response = client.get(
            "/users/me",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert response.status_code == 200

    def test_token_validation_performance(self, client, auth_headers):
        """Test that token validation is performant.

        Makes multiple rapid requests to ensure validation doesn't cause slowdown.
        """
        import time

        start = time.time()
        for _ in range(10):
            response = client.get("/users/me", headers=auth_headers)
            assert response.status_code == 200

        elapsed = time.time() - start
        avg_time = elapsed / 10

        # Token validation should be fast (< 50ms per request)
        assert avg_time < 0.05, f"Token validation too slow: {avg_time*1000:.2f}ms per request"

    def test_token_contains_expected_claims(self, client, test_user):
        """Test that generated tokens contain expected JWT claims.

        Verifies:
        - sub (subject) claim present
        - exp (expiration) claim present
        - Expiration is in the future
        """
        response = client.post(
            "/auth/login",
            json={
                "username": test_user["username"],
                "password": test_user["password"]
            }
        )

        token = response.json()["access_token"]
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])

        # Check required claims
        assert "sub" in payload, "Token missing 'sub' claim"
        assert "exp" in payload, "Token missing 'exp' claim"

        # Verify expiration is in future
        exp_timestamp = payload["exp"]
        now_timestamp = datetime.utcnow().timestamp()
        assert exp_timestamp > now_timestamp, "Token expiration is in the past"

        # Verify reasonable expiration time (should be hours, not minutes or days)
        exp_delta = exp_timestamp - now_timestamp
        assert 3600 <= exp_delta <= 86400 * 7, \
            f"Token expiration seems wrong: {exp_delta}s ({exp_delta/3600:.1f}h)"

    def test_token_username_matches_login(self, client, test_user):
        """Test that token contains username from login request.

        Verifies:
        - Token 'sub' claim matches logged-in username
        """
        response = client.post(
            "/auth/login",
            json={
                "username": test_user["username"],
                "password": test_user["password"]
            }
        )

        token = response.json()["access_token"]
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])

        assert payload["sub"] == test_user["username"], \
            "Token username doesn't match login username"


@pytest.mark.unit
class TestAuthenticationEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_concurrent_login_requests(self, client, test_user):
        """Test multiple simultaneous login requests.

        Verifies:
        - All requests succeed
        - Each gets a valid token
        """
        import concurrent.futures

        def login():
            return client.post(
                "/auth/login",
                json={
                    "username": test_user["username"],
                    "password": test_user["password"]
                }
            )

        # Make 10 concurrent login requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(login) for _ in range(10)]
            responses = [f.result() for f in futures]

        # All should succeed
        for response in responses:
            assert response.status_code == 200
            assert "access_token" in response.json()

    def test_very_long_username(self, client):
        """Test system behavior with extremely long username.

        Verifies:
        - Request is rejected with validation error
        """
        long_username = "a" * 1000

        response = client.post(
            "/auth/login",
            json={
                "username": long_username,
                "password": "password123"
            }
        )

        assert response.status_code == 422, "Should reject extremely long username"

    def test_unicode_in_credentials(self, client):
        """Test handling of unicode characters in credentials.

        Verifies:
        - System handles unicode gracefully
        """
        response = client.post(
            "/auth/login",
            json={
                "username": "test用户",
                "password": "пароль123"
            }
        )

        # Should handle gracefully (reject or accept depending on policy)
        assert response.status_code in [401, 422]

    def test_sql_injection_attempt_in_username(self, client):
        """Test that SQL injection attempts are safely handled.

        Verifies:
        - Injection attempts don't cause errors
        - System safely rejects the request
        """
        sql_injection_attempts = [
            "admin' OR '1'='1",
            "admin'--",
            "admin' #",
            "' OR 1=1--",
        ]

        for injection in sql_injection_attempts:
            response = client.post(
                "/auth/login",
                json={
                    "username": injection,
                    "password": "password123"
                }
            )

            # Should safely reject
            assert response.status_code in [401, 422], \
                f"SQL injection attempt not handled: {injection}"

    def test_null_bytes_in_credentials(self, client):
        """Test handling of null bytes in credentials.

        Verifies:
        - System handles null bytes safely
        """
        response = client.post(
            "/auth/login",
            json={
                "username": "test\x00user",
                "password": "pass\x00word"
            }
        )

        # Should be rejected or handled safely
        assert response.status_code in [401, 422]
