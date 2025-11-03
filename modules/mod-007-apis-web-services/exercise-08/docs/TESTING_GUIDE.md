# Comprehensive API Testing Implementation Guide

**Module:** APIs & Web Services - Exercise 08
**Target Audience:** Junior AI Infrastructure Engineers
**Estimated Reading Time:** 45-60 minutes

## Table of Contents

1. [Introduction](#introduction)
2. [Testing Philosophy](#testing-philosophy)
3. [Phase 1: Project Setup](#phase-1-project-setup)
4. [Phase 2: Test Infrastructure](#phase-2-test-infrastructure)
5. [Phase 3: Unit Tests](#phase-3-unit-tests)
6. [Phase 4: Integration Tests](#phase-4-integration-tests)
7. [Phase 5: Contract Tests](#phase-5-contract-tests)
8. [Phase 6: Performance Tests](#phase-6-performance-tests)
9. [Phase 7: CI/CD Integration](#phase-7-cicd-integration)
10. [Advanced Topics](#advanced-topics)
11. [Troubleshooting Guide](#troubleshooting-guide)
12. [Conclusion](#conclusion)

---

## Introduction

This guide provides step-by-step instructions for implementing comprehensive API testing for a production ML system. You'll learn to create a robust test suite covering unit tests, integration tests, contract tests, and performance tests.

### What You'll Build

By the end of this guide, you'll have:

- **150+ test cases** covering all API functionality
- **Reusable test fixtures** for clean, maintainable tests
- **CI/CD pipeline** with automated testing
- **Load testing framework** for performance validation
- **Stress testing tools** for identifying system limits
- **Coverage tracking** and quality metrics

### Time Breakdown

- Phase 1: Project Setup (15 minutes)
- Phase 2: Test Infrastructure (20 minutes)
- Phase 3: Unit Tests (30 minutes)
- Phase 4: Integration Tests (20 minutes)
- Phase 5: Contract Tests (20 minutes)
- Phase 6: Performance Tests (25 minutes)
- Phase 7: CI/CD Integration (20 minutes)

**Total**: ~2.5 hours

---

## Testing Philosophy

### The Testing Pyramid

```
        /\
       /E2E\        Few, slow, expensive
      /------\
     /Integr.\     Some, moderate speed
    /----------\
   /   Unit     \  Many, fast, cheap
  /--------------\
```

**Key Principles:**

1. **Write many unit tests**: Fast, focused, easy to maintain
2. **Some integration tests**: Validate workflows and interactions
3. **Few end-to-end tests**: Cover critical user paths
4. **Test behavior, not implementation**: Tests should survive refactoring
5. **Independent tests**: Each test should run in isolation
6. **Clear test names**: Test name = specification

### Test Quality Metrics

**Coverage Goals:**
- Overall code coverage: 80%+
- Critical paths: 100%
- New code: 100%

**Performance Goals:**
- Unit test suite: <10 seconds
- Integration test suite: <30 seconds
- Full test suite: <2 minutes

---

## Phase 1: Project Setup

### Step 1.1: Create Project Structure

```bash
# Create project directory
mkdir exercise-08
cd exercise-08

# Create directory structure
mkdir -p tests docs .github/workflows

# Verify structure
tree -L 2
```

Expected output:
```
exercise-08/
├── tests/
├── docs/
└── .github/
    └── workflows/
```

### Step 1.2: Set Up Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Verify activation
which python  # Should show path in venv/
python --version  # Should be 3.9+
```

### Step 1.3: Create Requirements Files

**requirements.txt** (Production Dependencies):

```txt
# Web framework
fastapi==0.109.0
uvicorn[standard]==0.27.0
pydantic==2.5.3

# Authentication
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4

# ML libraries
numpy==1.26.3
scikit-learn==1.4.0

# Utilities
python-dotenv==1.0.0
```

**requirements-test.txt** (Testing Dependencies):

```txt
-r requirements.txt

# Testing framework
pytest==7.4.4
pytest-asyncio==0.23.3
pytest-cov==4.1.0
pytest-xdist==3.5.0

# HTTP testing
requests==2.31.0
httpx==0.26.0

# Load testing
locust==2.20.0
aiohttp==3.9.1
psutil==5.9.7

# Schema validation
jsonschema==4.20.0
```

### Step 1.4: Install Dependencies

```bash
# Install testing dependencies
pip install -r requirements-test.txt

# Verify installation
pip list | grep pytest
pytest --version
```

### Step 1.5: Initialize Git Repository (Optional)

```bash
# Initialize repository
git init

# Create .gitignore
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/

# Testing
.pytest_cache/
.coverage
htmlcov/
test-results*.xml
*.log

# IDE
.vscode/
.idea/
*.swp
EOF

# Initial commit
git add .
git commit -m "Initial project setup"
```

---

## Phase 2: Test Infrastructure

### Step 2.1: Create pytest Configuration

Create `pytest.ini`:

```ini
[pytest]
# Test discovery
python_files = test_*.py
python_classes = Test*
python_functions = test_*

# Test paths
testpaths = tests

# Command line options
addopts =
    -v
    --strict-markers
    --tb=short

# Custom markers
markers =
    unit: Unit tests
    integration: Integration tests
    contract: Contract tests
    slow: Slow tests
```

### Step 2.2: Create conftest.py

Create `tests/conftest.py` with base fixtures:

```python
"""Pytest configuration and shared fixtures."""

import pytest
from fastapi.testclient import TestClient
import jwt
from datetime import datetime, timedelta
import tempfile
import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Configuration
SECRET_KEY = "test-secret-key-for-testing-only"
ALGORITHM = "HS256"


@pytest.fixture(scope="session")
def test_model():
    """Create a dummy ML model for testing."""
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    X = np.random.rand(100, 10)
    y = np.random.rand(100)
    model.fit(X, y)
    return model


@pytest.fixture(scope="session")
def model_file(test_model, tmp_path_factory):
    """Save model to temporary file."""
    tmp_dir = tmp_path_factory.mktemp("models")
    model_path = tmp_dir / "test_model.pkl"

    with open(model_path, 'wb') as f:
        pickle.dump(test_model, f)

    return str(model_path)
```

**Key Concepts:**

- **Fixture Scopes**: `session` (once), `module` (per module), `function` (per test)
- **Fixture Dependencies**: Fixtures can depend on other fixtures
- **tmp_path_factory**: Pytest provides temporary directories
- **Model Creation**: We create a real scikit-learn model for realistic testing

### Step 2.3: Create Test Client Fixture

Add to `tests/conftest.py`:

```python
@pytest.fixture(scope="function")
def client(model_file):
    """Create test client with embedded FastAPI app."""
    import os
    os.environ['SECRET_KEY'] = SECRET_KEY
    os.environ['MODEL_PATH'] = model_file

    # Import and create FastAPI app
    from fastapi import FastAPI, Depends, HTTPException, status
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from pydantic import BaseModel, Field
    from typing import List

    app = FastAPI(title="Test ML API")
    security = HTTPBearer()

    # In-memory user storage
    test_users = {
        "testuser": {
            "username": "testuser",
            "email": "test@example.com",
            "hashed_password": "hashed_testpassword123",
            "is_active": True
        }
    }

    # Models
    class LoginRequest(BaseModel):
        username: str = Field(..., min_length=3, max_length=50)
        password: str = Field(..., min_length=8)

    class LoginResponse(BaseModel):
        access_token: str
        token_type: str = "bearer"

    # Auth dependency
    async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
        try:
            token = credentials.credentials
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            username = payload.get("sub")
            if username not in test_users:
                raise HTTPException(status_code=401, detail="User not found")
            return test_users[username]
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token has expired")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token")

    # Endpoints
    @app.post("/auth/login", response_model=LoginResponse)
    async def login(credentials: LoginRequest):
        user = test_users.get(credentials.username)
        if not user or user["hashed_password"] != f"hashed_{credentials.password}":
            raise HTTPException(status_code=401, detail="Invalid credentials")

        token = jwt.encode(
            {"sub": credentials.username, "exp": datetime.utcnow() + timedelta(hours=24)},
            SECRET_KEY,
            algorithm=ALGORITHM
        )
        return LoginResponse(access_token=token)

    # Create test client
    with TestClient(app) as test_client:
        yield test_client
```

**Key Concepts:**

- **TestClient**: FastAPI provides a test client that simulates HTTP requests
- **Embedded App**: We embed a minimal FastAPI app for testing
- **Dependency Injection**: FastAPI's dependency system simplifies testing
- **Context Manager**: `with` statement ensures proper cleanup

### Step 2.4: Create Authentication Fixtures

Add to `tests/conftest.py`:

```python
@pytest.fixture
def test_user():
    """Test user credentials."""
    return {
        "username": "testuser",
        "email": "test@example.com",
        "password": "testpassword123"
    }


@pytest.fixture
def auth_token(test_user):
    """Generate valid JWT token."""
    payload = {
        "sub": test_user["username"],
        "exp": datetime.utcnow() + timedelta(hours=1)
    }
    token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
    return token


@pytest.fixture
def expired_token(test_user):
    """Generate expired JWT token."""
    payload = {
        "sub": test_user["username"],
        "exp": datetime.utcnow() - timedelta(hours=1)
    }
    token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
    return token


@pytest.fixture
def auth_headers(auth_token):
    """Generate authorization headers."""
    return {"Authorization": f"Bearer {auth_token}"}
```

**Key Concepts:**

- **Fixture Chaining**: `auth_headers` depends on `auth_token` depends on `test_user`
- **JWT Tokens**: We generate real JWT tokens for authentication testing
- **Expired Tokens**: Test token expiration by creating past-dated tokens

### Step 2.5: Verify Test Infrastructure

Create `tests/test_setup.py`:

```python
"""Verify test infrastructure is working."""

import pytest


def test_pytest_working():
    """Verify pytest is configured correctly."""
    assert True


def test_fixtures_available(client, test_user, auth_headers):
    """Verify all fixtures are available."""
    assert client is not None
    assert test_user is not None
    assert auth_headers is not None
    assert "Authorization" in auth_headers


def test_client_can_make_requests(client):
    """Verify test client can make HTTP requests."""
    # This should fail with 404 but proves client works
    response = client.get("/nonexistent")
    assert response.status_code == 404
```

Run tests:

```bash
pytest tests/test_setup.py -v
```

Expected output:
```
tests/test_setup.py::test_pytest_working PASSED
tests/test_setup.py::test_fixtures_available PASSED
tests/test_setup.py::test_client_can_make_requests PASSED

======== 3 passed in 0.15s ========
```

---

## Phase 3: Unit Tests

### Step 3.1: Authentication Unit Tests

Create `tests/test_auth.py`:

```python
"""Unit tests for authentication endpoints."""

import pytest
import jwt
from datetime import datetime, timedelta

SECRET_KEY = "test-secret-key-for-testing-only"
ALGORITHM = "HS256"


@pytest.mark.unit
class TestLogin:
    """Tests for login endpoint."""

    def test_login_success(self, client, test_user):
        """Test successful login with valid credentials."""
        response = client.post(
            "/auth/login",
            json={
                "username": test_user["username"],
                "password": test_user["password"]
            }
        )

        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"

        # Verify token is valid
        token = data["access_token"]
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        assert payload["sub"] == test_user["username"]
        assert "exp" in payload

    def test_login_invalid_username(self, client):
        """Test login with non-existent username."""
        response = client.post(
            "/auth/login",
            json={
                "username": "nonexistent_user",
                "password": "anypassword"
            }
        )

        assert response.status_code == 401
        assert "detail" in response.json()

    def test_login_invalid_password(self, client, test_user):
        """Test login with incorrect password."""
        response = client.post(
            "/auth/login",
            json={
                "username": test_user["username"],
                "password": "wrongpassword"
            }
        )

        assert response.status_code == 401
```

**Testing Pattern:**

1. **Arrange**: Set up test data
2. **Act**: Make the API call
3. **Assert**: Verify the response

### Step 3.2: Input Validation Tests

Add to `tests/test_auth.py`:

```python
@pytest.mark.unit
class TestLoginValidation:
    """Tests for login input validation."""

    @pytest.mark.parametrize("username,password,expected", [
        ("", "password123", 422),           # Empty username
        ("ab", "password123", 422),         # Too short username
        ("a" * 51, "password123", 422),     # Too long username
        ("testuser", "short", 422),         # Too short password
    ])
    def test_validation(self, client, username, password, expected):
        """Test input validation rules."""
        response = client.post(
            "/auth/login",
            json={"username": username, "password": password}
        )
        assert response.status_code == expected
```

**Key Concepts:**

- **Parametrize**: Run same test with different inputs
- **Validation Testing**: Verify input constraints are enforced
- **Expected Failures**: Test that invalid input is rejected

### Step 3.3: Run Unit Tests

```bash
# Run all unit tests
pytest tests/test_auth.py -m unit -v

# Run specific test
pytest tests/test_auth.py::TestLogin::test_login_success -v

# Run with coverage
pytest tests/test_auth.py -m unit --cov=src --cov-report=term-missing
```

---

## Phase 4: Integration Tests

### Step 4.1: Complete Workflow Tests

Create `tests/test_integration.py`:

```python
"""Integration tests for complete workflows."""

import pytest


@pytest.mark.integration
class TestCompleteAuthFlow:
    """Test complete authentication workflow."""

    def test_login_and_access_protected_resources(self, client, test_user):
        """Test complete flow: login → access protected endpoint."""
        # Step 1: Login
        login_response = client.post(
            "/auth/login",
            json={
                "username": test_user["username"],
                "password": test_user["password"]
            }
        )
        assert login_response.status_code == 200
        token = login_response.json()["access_token"]

        # Step 2: Access protected resource
        user_response = client.get(
            "/users/me",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert user_response.status_code == 200

        # Step 3: Verify data
        user_data = user_response.json()
        assert user_data["username"] == test_user["username"]
        assert user_data["email"] == test_user["email"]
```

**Key Concepts:**

- **Multi-Step Tests**: Integration tests cover multiple operations
- **State Propagation**: Token from login used in subsequent requests
- **End-to-End Validation**: Verify complete user journeys

### Step 4.2: System Behavior Tests

Add to `tests/test_integration.py`:

```python
@pytest.mark.integration
class TestSystemBehavior:
    """Test system-level behavior."""

    def test_sequential_requests_maintain_consistency(self, client, auth_headers):
        """Test that sequential requests maintain consistent behavior."""
        features = [1.0] * 10

        # Make 20 sequential predictions
        predictions = []
        for i in range(20):
            response = client.post(
                "/predict",
                json={"features": features},
                headers=auth_headers
            )
            assert response.status_code == 200
            predictions.append(response.json()["prediction"])

        # All predictions for same input should be identical
        assert len(set(predictions)) == 1, "Predictions should be consistent"

    def test_error_recovery(self, client, auth_headers):
        """Test that system recovers from errors gracefully."""
        # Valid request
        response1 = client.post(
            "/predict",
            json={"features": [1.0] * 10},
            headers=auth_headers
        )
        assert response1.status_code == 200

        # Invalid request
        response2 = client.post(
            "/predict",
            json={"features": [1.0] * 5},  # Wrong count
            headers=auth_headers
        )
        assert response2.status_code == 422

        # System should still work
        response3 = client.post(
            "/predict",
            json={"features": [1.0] * 10},
            headers=auth_headers
        )
        assert response3.status_code == 200
```

**Key Concepts:**

- **Consistency Testing**: Verify deterministic behavior
- **Error Recovery**: Ensure errors don't corrupt system state
- **Resilience**: System should handle failures gracefully

---

## Phase 5: Contract Tests

### Step 5.1: Response Schema Validation

Create `tests/test_contracts.py`:

```python
"""Contract tests for API schema validation."""

import pytest


@pytest.mark.contract
class TestResponseSchemas:
    """Test that API responses match expected schemas."""

    def test_login_response_schema(self, client, test_user):
        """Test login response matches LoginResponse schema."""
        response = client.post(
            "/auth/login",
            json={
                "username": test_user["username"],
                "password": test_user["password"]
            }
        )

        assert response.status_code == 200
        data = response.json()

        # Expected fields
        expected_fields = {"access_token", "token_type"}
        actual_fields = set(data.keys())
        assert expected_fields == actual_fields

        # Field types
        assert isinstance(data["access_token"], str)
        assert isinstance(data["token_type"], str)
        assert data["token_type"] == "bearer"
```

**Key Concepts:**

- **Schema Validation**: Verify response structure matches specification
- **Field Types**: Ensure correct data types
- **Required Fields**: Verify all required fields present

### Step 5.2: Backward Compatibility Tests

Add to `tests/test_contracts.py`:

```python
@pytest.mark.contract
class TestBackwardCompatibility:
    """Test API backward compatibility."""

    def test_response_fields_stable(self, client, auth_headers):
        """Test that core response fields remain stable."""
        response = client.post(
            "/predict",
            json={"features": [1.0] * 10},
            headers=auth_headers
        )

        data = response.json()

        # Core fields that must always exist
        stable_fields = {
            "prediction": (int, float),
            "model_version": str
        }

        for field, expected_type in stable_fields.items():
            assert field in data, \
                f"Breaking change: required field '{field}' missing"
            assert isinstance(data[field], expected_type), \
                f"Breaking change: field '{field}' type changed"
```

**Key Concepts:**

- **API Stability**: Ensure existing clients don't break
- **Version Compatibility**: Test across API versions
- **Breaking Changes**: Detect incompatible changes early

---

## Phase 6: Performance Tests

### Step 6.1: Load Testing with Locust

Create `tests/locustfile.py`:

```python
"""Locust load testing scenarios."""

from locust import HttpUser, task, between
import random


class MLAPIUser(HttpUser):
    """Simulates typical ML API user behavior."""

    wait_time = between(1, 3)
    weight = 70

    def on_start(self):
        """Login at start."""
        response = self.client.post(
            "/auth/login",
            json={
                "username": "testuser",
                "password": "testpassword123"
            }
        )

        if response.status_code == 200:
            self.token = response.json()["access_token"]
            self.headers = {"Authorization": f"Bearer {self.token}"}
        else:
            self.headers = {}

    @task(50)
    def predict_single(self):
        """Make single prediction request."""
        features = [random.uniform(-10, 10) for _ in range(10)]

        self.client.post(
            "/predict",
            json={"features": features},
            headers=self.headers,
            name="/predict [single]"
        )

    @task(20)
    def predict_batch(self):
        """Make batch prediction request."""
        batch_size = random.randint(2, 10)
        samples = [
            [random.uniform(-10, 10) for _ in range(10)]
            for _ in range(batch_size)
        ]

        self.client.post(
            "/batch-predict",
            json={"samples": samples},
            headers=self.headers,
            name=f"/batch-predict [size={batch_size}]"
        )
```

**Running Load Tests:**

```bash
# Start Locust web UI
locust -f tests/locustfile.py --host=http://localhost:8000

# Headless mode
locust -f tests/locustfile.py \
       --host=http://localhost:8000 \
       --users 50 \
       --spawn-rate 5 \
       --run-time 5m \
       --headless
```

### Step 6.2: Stress Testing

Create basic stress test script (see full version in solution):

```python
"""Stress testing script."""

import requests
import time
from concurrent.futures import ThreadPoolExecutor

def make_request(url, headers):
    """Make single request."""
    try:
        response = requests.post(
            url,
            json={"features": [1.0] * 10},
            headers=headers,
            timeout=10
        )
        return response.status_code == 200
    except:
        return False

# Run stress test
url = "http://localhost:8000/predict"
headers = {"Authorization": "Bearer YOUR_TOKEN"}

with ThreadPoolExecutor(max_workers=50) as executor:
    futures = [executor.submit(make_request, url, headers) for _ in range(1000)]
    results = [f.result() for f in futures]

success_rate = sum(results) / len(results) * 100
print(f"Success rate: {success_rate:.1f}%")
```

---

## Phase 7: CI/CD Integration

### Step 7.1: Create GitHub Actions Workflow

Create `.github/workflows/test.yml`:

```yaml
name: ML API Testing CI/CD

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  test-unit:
    name: Unit Tests
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements-test.txt

      - name: Run unit tests
        run: |
          pytest tests/ -m unit \
            --cov=src \
            --cov-report=xml \
            --junitxml=test-results.xml

      - name: Upload coverage
        uses: codecov/codecov-action@v4
        with:
          files: ./coverage.xml
```

### Step 7.2: Pre-commit Hooks

Create `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 24.1.1
    hooks:
      - id: black
        language_version: python3.11

  - repo: https://github.com/PyCQA/flake8
    rev: 7.0.0
    hooks:
      - id: flake8
        args: ['--max-line-length=100']
```

Install and use:

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

---

## Advanced Topics

### Mocking External Services

```python
from unittest.mock import patch, Mock

def test_external_api_call():
    """Test with mocked external service."""
    with patch('requests.get') as mock_get:
        # Configure mock
        mock_response = Mock()
        mock_response.json.return_value = {"data": "mocked"}
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        # Test code that calls requests.get
        # ...
```

### Database Testing

```python
@pytest.fixture
def db_session():
    """Create test database session."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    # Use in-memory SQLite for tests
    engine = create_engine("sqlite:///:memory:")
    Session = sessionmaker(bind=engine)
    session = Session()

    yield session

    session.close()
```

### Async Testing

```python
import pytest

@pytest.mark.asyncio
async def test_async_endpoint():
    """Test async endpoint."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/async-endpoint")
        assert response.status_code == 200
```

---

## Troubleshooting Guide

### Common Issues and Solutions

**Issue: Import Errors**
```bash
# Solution: Install package in editable mode
pip install -e .
```

**Issue: Fixtures Not Found**
```bash
# Solution: Check conftest.py location
ls tests/conftest.py

# List all available fixtures
pytest --fixtures
```

**Issue: Slow Tests**
```bash
# Solution: Run in parallel
pip install pytest-xdist
pytest -n auto
```

---

## Conclusion

You've now learned how to build a comprehensive API testing solution:

✅ Unit tests for individual components
✅ Integration tests for complete workflows
✅ Contract tests for API stability
✅ Performance tests for load and stress
✅ CI/CD integration for automation

### Next Steps

1. Apply these patterns to your own APIs
2. Expand test coverage to 90%+
3. Add monitoring and alerting
4. Implement chaos engineering

---

*Implementation guide for AI Infrastructure Junior Engineer Learning Curriculum*
*Last updated: 2025-10-30*
