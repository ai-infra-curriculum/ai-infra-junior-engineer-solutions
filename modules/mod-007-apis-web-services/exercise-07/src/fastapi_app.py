"""FastAPI-based ML model serving API.

Modern API implementation showcasing FastAPI features:
- Automatic request validation with Pydantic
- Automatic OpenAPI documentation
- Async/await support
- Dependency injection
- Background tasks
- Middleware
- Better error handling

This demonstrates the migration from Flask to FastAPI while
maintaining backward compatibility.
"""

from fastapi import (
    FastAPI,
    Depends,
    HTTPException,
    status,
    BackgroundTasks,
    Request
)
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Optional, Dict
import pickle
import numpy as np
import jwt
from datetime import datetime, timedelta
import time
import os
import asyncio
from contextlib import asynccontextmanager

# Import Pydantic models
from models import (
    HealthResponse,
    HealthStatus,
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    ModelInfo,
    LoginRequest,
    LoginResponse,
    ErrorResponse,
    CacheStats
)

# Configuration
SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key-change-in-production')
ALGORITHM = "HS256"
TOKEN_EXPIRATION_HOURS = 24

# Global variables
model = None
prediction_cache: Dict[str, float] = {}
cache_stats = {
    'hits': 0,
    'misses': 0,
    'total': 0
}
start_time = time.time()


# Lifespan context manager for startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    # Startup
    print("🚀 Starting FastAPI ML API...")
    await load_model_async()
    print("✅ Startup complete")

    yield

    # Shutdown
    print("🛑 Shutting down FastAPI ML API...")
    await cleanup()
    print("✅ Shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="ML Model Serving API",
    description="""
    FastAPI-based machine learning model serving API.

    ## Features

    * **Automatic validation** - Request/response validation with Pydantic
    * **Interactive docs** - Swagger UI and ReDoc
    * **Authentication** - JWT token-based auth
    * **Performance** - Async endpoints and caching
    * **Monitoring** - Built-in metrics and health checks

    ## Authentication

    Most endpoints require authentication. Use `/login` to get a token,
    then include it in the `Authorization` header as `Bearer <token>`.
    """,
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security scheme
security = HTTPBearer()


async def load_model_async():
    """Load ML model asynchronously on startup."""
    global model

    model_path = os.getenv('MODEL_PATH', 'model.pkl')

    # Simulate async I/O (e.g., loading from S3, GCS, etc.)
    await asyncio.sleep(0.1)

    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"✅ Model loaded successfully from {model_path}")
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
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise


async def cleanup():
    """Cleanup on shutdown."""
    global prediction_cache, cache_stats

    # Clear cache
    prediction_cache.clear()

    # Log final stats
    total = cache_stats['total']
    hit_rate = cache_stats['hits'] / total if total > 0 else 0.0
    print(f"   Final cache stats: {cache_stats['hits']}/{total} hits ({hit_rate:.2%})")


# Middleware: Add processing time header
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add X-Process-Time header to all responses."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = f"{process_time:.4f}"
    return response


# Middleware: Request ID tracking
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add request ID to response headers."""
    import uuid
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# Dependency: Verify JWT token
async def verify_token(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> dict:
    """Verify and decode JWT token.

    Args:
        credentials: HTTP Bearer credentials from request

    Returns:
        Decoded token payload

    Raises:
        HTTPException: If token is invalid or expired
    """
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired"
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )


# Background task: Log predictions
def log_prediction(
    features: list,
    prediction: float,
    user: str,
    cached: bool,
    request_id: Optional[str] = None
):
    """Log prediction to file (runs in background).

    Args:
        features: Input features
        prediction: Model prediction
        user: Username
        cached: Whether result was cached
        request_id: Optional request ID
    """
    try:
        log_file = os.getenv('PREDICTION_LOG', 'predictions.log')
        with open(log_file, 'a') as f:
            timestamp = datetime.utcnow().isoformat()
            log_entry = {
                'timestamp': timestamp,
                'user': user,
                'features': features,
                'prediction': prediction,
                'cached': cached,
                'request_id': request_id
            }
            f.write(f"{log_entry}\n")
    except Exception as e:
        print(f"⚠️  Failed to log prediction: {e}")


# Background task: Log batch predictions
def log_batch_prediction(
    sample_count: int,
    user: str,
    processing_time_ms: float,
    request_id: Optional[str] = None
):
    """Log batch prediction to file (runs in background).

    Args:
        sample_count: Number of samples processed
        user: Username
        processing_time_ms: Processing time in milliseconds
        request_id: Optional request ID
    """
    try:
        log_file = os.getenv('PREDICTION_LOG', 'predictions.log')
        with open(log_file, 'a') as f:
            timestamp = datetime.utcnow().isoformat()
            log_entry = {
                'timestamp': timestamp,
                'user': user,
                'type': 'batch',
                'sample_count': sample_count,
                'processing_time_ms': processing_time_ms,
                'request_id': request_id
            }
            f.write(f"{log_entry}\n")
    except Exception as e:
        print(f"⚠️  Failed to log batch prediction: {e}")


# Endpoints

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health check",
    description="Check if the API is running and the model is loaded"
)
async def health_check():
    """Health check endpoint.

    Returns current status, timestamp, uptime, and model status.
    """
    uptime = time.time() - start_time
    model_loaded = model is not None

    return HealthResponse(
        status=HealthStatus.HEALTHY if model_loaded else HealthStatus.UNHEALTHY,
        timestamp=datetime.utcnow(),
        uptime_seconds=uptime,
        model_loaded=model_loaded
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Predictions"],
    summary="Make single prediction",
    description="Make a prediction on a single feature vector",
    responses={
        200: {
            "description": "Successful prediction",
            "model": PredictionResponse
        },
        401: {
            "description": "Unauthorized - invalid or missing token",
            "model": ErrorResponse
        },
        422: {
            "description": "Validation Error - invalid input format",
            "model": ErrorResponse
        },
        500: {
            "description": "Internal Server Error",
            "model": ErrorResponse
        }
    }
)
async def predict(
    request: PredictionRequest,
    background_tasks: BackgroundTasks,
    user: dict = Depends(verify_token)
):
    """Make a prediction on input features.

    **Authentication required** - Include JWT token in Authorization header.

    The endpoint accepts a feature vector and returns a prediction.
    Results are cached for faster subsequent requests with identical features.

    Args:
        request: Prediction request with features
        background_tasks: FastAPI background tasks
        user: Authenticated user (from token)

    Returns:
        Prediction response with value and metadata
    """
    try:
        # Check cache
        feature_key = str(request.features)
        if feature_key in prediction_cache:
            cache_stats['hits'] += 1
            cache_stats['total'] += 1

            prediction = prediction_cache[feature_key]

            # Add background task for logging
            background_tasks.add_task(
                log_prediction,
                request.features,
                prediction,
                user['user'],
                True,
                request.request_id
            )

            return PredictionResponse(
                prediction=prediction,
                cached=True,
                model_version="1.0",
                request_id=request.request_id
            )

        # Make prediction
        if model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded"
            )

        # Convert to numpy array and predict
        features_array = np.array([request.features])
        prediction = float(model.predict(features_array)[0])

        # Cache result
        prediction_cache[feature_key] = prediction
        cache_stats['misses'] += 1
        cache_stats['total'] += 1

        # Add background task for logging
        background_tasks.add_task(
            log_prediction,
            request.features,
            prediction,
            user['user'],
            False,
            request.request_id
        )

        return PredictionResponse(
            prediction=prediction,
            cached=False,
            model_version="1.0",
            request_id=request.request_id
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post(
    "/batch-predict",
    response_model=BatchPredictionResponse,
    tags=["Predictions"],
    summary="Make batch predictions",
    description="Make predictions on multiple feature vectors",
    responses={
        200: {
            "description": "Successful batch prediction",
            "model": BatchPredictionResponse
        },
        401: {
            "description": "Unauthorized",
            "model": ErrorResponse
        },
        422: {
            "description": "Validation Error",
            "model": ErrorResponse
        },
        500: {
            "description": "Internal Server Error",
            "model": ErrorResponse
        }
    }
)
async def batch_predict(
    request: BatchPredictionRequest,
    background_tasks: BackgroundTasks,
    user: dict = Depends(verify_token)
):
    """Make predictions on multiple samples.

    **Authentication required** - Include JWT token in Authorization header.

    Accepts up to 100 samples for batch prediction. More efficient than
    making individual requests for multiple predictions.

    Args:
        request: Batch prediction request with samples
        background_tasks: FastAPI background tasks
        user: Authenticated user (from token)

    Returns:
        Batch prediction response with all predictions
    """
    try:
        start = time.time()

        # Check model is loaded
        if model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded"
            )

        # Make predictions
        features_array = np.array(request.samples)
        predictions = model.predict(features_array)

        processing_time = (time.time() - start) * 1000  # milliseconds

        # Add background task for logging
        background_tasks.add_task(
            log_batch_prediction,
            len(request.samples),
            user['user'],
            processing_time,
            request.request_id
        )

        return BatchPredictionResponse(
            predictions=[float(p) for p in predictions],
            count=len(predictions),
            model_version="1.0",
            request_id=request.request_id,
            processing_time_ms=processing_time
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.get(
    "/model-info",
    response_model=ModelInfo,
    tags=["Model"],
    summary="Get model information",
    description="Get metadata about the deployed model"
)
async def model_info():
    """Get information about the deployed model.

    Returns version, type, training date, and performance metrics.
    """
    return ModelInfo(
        version="1.0",
        input_features=10,
        model_type="random_forest",
        training_date="2024-01-15",
        accuracy=0.94,
        framework="scikit-learn"
    )


@app.post(
    "/login",
    response_model=LoginResponse,
    tags=["Authentication"],
    summary="Authenticate",
    description="Login with credentials to receive JWT token",
    responses={
        200: {
            "description": "Successful login",
            "model": LoginResponse
        },
        401: {
            "description": "Invalid credentials",
            "model": ErrorResponse
        },
        422: {
            "description": "Validation Error",
            "model": ErrorResponse
        }
    }
)
async def login(credentials: LoginRequest):
    """Authenticate and receive JWT token.

    Validates credentials and returns a JWT token valid for 24 hours.
    Include this token in the Authorization header for protected endpoints.

    **Note**: This is a simplified example. In production:
    - Validate against a database
    - Use hashed passwords (bcrypt, argon2)
    - Implement rate limiting
    - Add refresh tokens
    - Log authentication attempts

    Args:
        credentials: Username and password

    Returns:
        JWT token and expiration information
    """
    # Simplified auth (in production, validate against database with hashed passwords)
    if credentials.username == "admin" and credentials.password == "password":
        # Create token
        token = jwt.encode(
            {
                "user": credentials.username,
                "exp": datetime.utcnow() + timedelta(hours=TOKEN_EXPIRATION_HOURS)
            },
            SECRET_KEY,
            algorithm=ALGORITHM
        )

        return LoginResponse(
            token=token,
            token_type="bearer",
            expires_in=TOKEN_EXPIRATION_HOURS * 3600  # Convert to seconds
        )

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid credentials"
    )


@app.get(
    "/cache/stats",
    response_model=CacheStats,
    tags=["Cache"],
    summary="Get cache statistics",
    description="Get metrics about the prediction cache"
)
async def cache_stats_endpoint(user: dict = Depends(verify_token)):
    """Get cache statistics.

    **Authentication required** - Include JWT token in Authorization header.

    Returns metrics about cache size, hit rate, and request counts.

    Args:
        user: Authenticated user (from token)

    Returns:
        Cache statistics
    """
    total = cache_stats['total']
    hit_rate = cache_stats['hits'] / total if total > 0 else 0.0

    return CacheStats(
        size=len(prediction_cache),
        hit_rate=hit_rate,
        total_requests=total,
        cache_hits=cache_stats['hits'],
        cache_misses=cache_stats['misses']
    )


@app.post(
    "/cache/clear",
    tags=["Cache"],
    summary="Clear cache",
    description="Clear all cached predictions"
)
async def clear_cache(user: dict = Depends(verify_token)):
    """Clear the prediction cache.

    **Authentication required** - Include JWT token in Authorization header.

    Removes all cached predictions. Use this after model updates.

    Args:
        user: Authenticated user (from token)

    Returns:
        Success message with count of cleared entries
    """
    global prediction_cache

    entries_cleared = len(prediction_cache)
    prediction_cache.clear()

    return {
        "message": "Cache cleared successfully",
        "entries_cleared": entries_cleared
    }


# Custom exception handler
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with standard error format."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "error_code": f"HTTP_{exc.status_code}",
            "timestamp": datetime.utcnow().isoformat()
        }
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv('PORT', 8000))
    workers = int(os.getenv('WORKERS', 1))
    reload = os.getenv('RELOAD', 'False').lower() == 'true'

    print(f"🚀 Starting FastAPI ML API on port {port}")
    print(f"   Workers: {workers}")
    print(f"   Reload: {reload}")
    print(f"   Documentation:")
    print(f"     - Swagger UI: http://localhost:{port}/docs")
    print(f"     - ReDoc: http://localhost:{port}/redoc")
    print()

    uvicorn.run(
        "fastapi_app:app",
        host="0.0.0.0",
        port=port,
        workers=workers,
        reload=reload
    )
