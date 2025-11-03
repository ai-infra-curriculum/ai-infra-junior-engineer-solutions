"""Pytest configuration and shared fixtures.

This module provides reusable fixtures for testing the ML API:
- Test database setup/teardown
- Test client with dependency overrides
- Authentication fixtures (users, tokens, headers)
- Mock model fixtures
- Sample data generators

Usage:
    Fixtures are automatically discovered by pytest and can be used
    in any test by including them as function parameters.
"""

import pytest
from fastapi.testclient import TestClient
import jwt
from datetime import datetime, timedelta
import tempfile
import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Configuration
SECRET_KEY = "test-secret-key-for-testing-only"
ALGORITHM = "HS256"


@pytest.fixture(scope="session")
def test_model():
    """Create a dummy ML model for testing.

    Returns:
        Trained scikit-learn model
    """
    # Create simple model
    model = RandomForestRegressor(n_estimators=10, random_state=42)

    # Train on dummy data
    X = np.random.rand(100, 10)
    y = np.random.rand(100)
    model.fit(X, y)

    return model


@pytest.fixture(scope="session")
def model_file(test_model, tmp_path_factory):
    """Save model to temporary file for testing.

    Args:
        test_model: Trained model fixture
        tmp_path_factory: Pytest temporary directory factory

    Returns:
        Path to saved model file
    """
    tmp_dir = tmp_path_factory.mktemp("models")
    model_path = tmp_dir / "test_model.pkl"

    with open(model_path, 'wb') as f:
        pickle.dump(test_model, f)

    return str(model_path)


@pytest.fixture(scope="function")
def client(model_file):
    """Create test client with configured FastAPI app.

    This fixture:
    - Sets up test environment variables
    - Creates FastAPI test client
    - Overrides model path to use test model
    - Cleans up after each test

    Args:
        model_file: Path to test model fixture

    Yields:
        TestClient: Configured test client
    """
    # Set environment variables for testing
    os.environ['SECRET_KEY'] = SECRET_KEY
    os.environ['MODEL_PATH'] = model_file
    os.environ['DEBUG'] = 'False'

    # Import app after setting env vars
    # Note: In real implementation, import from your app module
    # For this solution, we'll create a minimal FastAPI app for testing
    from fastapi import FastAPI, Depends, HTTPException, status, Header
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from pydantic import BaseModel, Field, validator
    from typing import List, Optional

    app = FastAPI(title="Test ML API", version="1.0.0")
    security = HTTPBearer()

    # Simple in-memory storage for testing
    test_users = {
        "testuser": {
            "username": "testuser",
            "email": "test@example.com",
            "hashed_password": "hashed_testpassword123",  # In real app, use bcrypt
            "is_active": True
        }
    }

    prediction_cache = {}

    # Pydantic models
    class LoginRequest(BaseModel):
        username: str = Field(..., min_length=3, max_length=50)
        password: str = Field(..., min_length=8)

    class LoginResponse(BaseModel):
        access_token: str
        token_type: str = "bearer"

    class PredictionRequest(BaseModel):
        features: List[float] = Field(..., min_items=10, max_items=10)

        @validator('features')
        def validate_features(cls, v):
            for i, f in enumerate(v):
                if not (-1000 <= f <= 1000):
                    raise ValueError(f'Feature {i} out of range')
            return v

    class PredictionResponse(BaseModel):
        prediction: float
        cached: bool = False
        model_version: str = "1.0.0"

    class BatchPredictionRequest(BaseModel):
        samples: List[List[float]] = Field(..., min_items=1, max_items=100)

        @validator('samples')
        def validate_samples(cls, v):
            for i, sample in enumerate(v):
                if len(sample) != 10:
                    raise ValueError(f'Sample {i} must have 10 features')
                for j, f in enumerate(sample):
                    if not (-1000 <= f <= 1000):
                        raise ValueError(f'Sample {i}, feature {j} out of range')
            return v

    class BatchPredictionResponse(BaseModel):
        predictions: List[float]
        count: int
        model_version: str = "1.0.0"

    class UserResponse(BaseModel):
        username: str
        email: str
        is_active: bool

    # Load model
    with open(model_file, 'rb') as f:
        model = pickle.load(f)

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

    @app.get("/users/me", response_model=UserResponse)
    async def get_current_user(user: dict = Depends(verify_token)):
        return UserResponse(**user)

    @app.post("/predict", response_model=PredictionResponse)
    async def predict(request: PredictionRequest, user: dict = Depends(verify_token)):
        feature_key = str(request.features)

        if feature_key in prediction_cache:
            return PredictionResponse(
                prediction=prediction_cache[feature_key],
                cached=True
            )

        features_array = np.array([request.features])
        prediction = float(model.predict(features_array)[0])
        prediction_cache[feature_key] = prediction

        return PredictionResponse(prediction=prediction, cached=False)

    @app.post("/batch-predict", response_model=BatchPredictionResponse)
    async def batch_predict(request: BatchPredictionRequest, user: dict = Depends(verify_token)):
        features_array = np.array(request.samples)
        predictions = model.predict(features_array)

        return BatchPredictionResponse(
            predictions=[float(p) for p in predictions],
            count=len(predictions)
        )

    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

    @app.get("/models")
    async def get_models():
        return [{
            "name": "test-model",
            "version": "1.0.0",
            "type": "random_forest",
            "features": 10
        }]

    # Create test client
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def test_user():
    """Test user data.

    Returns:
        Dict with user information
    """
    return {
        "username": "testuser",
        "email": "test@example.com",
        "password": "testpassword123"
    }


@pytest.fixture
def auth_token(test_user):
    """Generate valid JWT token for test user.

    Args:
        test_user: Test user fixture

    Returns:
        Valid JWT token string
    """
    payload = {
        "sub": test_user["username"],
        "exp": datetime.utcnow() + timedelta(hours=1)
    }
    token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
    return token


@pytest.fixture
def expired_token(test_user):
    """Generate expired JWT token for testing.

    Args:
        test_user: Test user fixture

    Returns:
        Expired JWT token string
    """
    payload = {
        "sub": test_user["username"],
        "exp": datetime.utcnow() - timedelta(hours=1)  # Expired 1 hour ago
    }
    token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
    return token


@pytest.fixture
def auth_headers(auth_token):
    """Generate authorization headers with valid token.

    Args:
        auth_token: Valid JWT token fixture

    Returns:
        Dict with Authorization header
    """
    return {"Authorization": f"Bearer {auth_token}"}


@pytest.fixture
def sample_features():
    """Generate sample feature vector for predictions.

    Returns:
        List of 10 floats
    """
    return [float(i) for i in range(1, 11)]


@pytest.fixture
def sample_batch():
    """Generate sample batch of feature vectors.

    Returns:
        List of feature vectors
    """
    return [
        [float(i) for i in range(1, 11)],
        [float(i + 10) for i in range(1, 11)],
        [float(i + 20) for i in range(1, 11)]
    ]


@pytest.fixture(autouse=True)
def reset_cache():
    """Reset prediction cache before each test.

    This fixture runs automatically before each test to ensure
    cache doesn't affect test results.
    """
    # Cache is recreated with each client fixture, so no action needed
    yield


# Markers for categorizing tests
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "slow: Slow tests (skipped by default)")
    config.addinivalue_line("markers", "contract: Contract/schema tests")
    config.addinivalue_line("markers", "load: Load/performance tests")


# Hooks for test reporting
@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Hook to capture test results for reporting."""
    outcome = yield
    rep = outcome.get_result()

    # Store result for use in fixtures
    setattr(item, f"rep_{rep.when}", rep)


@pytest.fixture
def test_result(request):
    """Get test execution result.

    Useful for cleanup logic that depends on test outcome.

    Args:
        request: Pytest request fixture

    Returns:
        Test result object
    """
    return request.node
