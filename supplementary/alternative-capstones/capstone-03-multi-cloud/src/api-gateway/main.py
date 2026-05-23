"""
Multi-Cloud ML Platform API Gateway
Unified API for model serving across AWS, GCP, and Azure
"""

import asyncio
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum

from fastapi import FastAPI, HTTPException, Depends, Header, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn
from prometheus_client import Counter, Histogram, Gauge, make_asgi_app
import logging

# Import cloud abstraction layer
import sys
sys.path.append('../cloud-abstraction')
from cloud_provider import CloudProvider, CloudFactory, CloudMetrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Multi-Cloud ML Platform API",
    description="Unified API for ML model serving across AWS, GCP, and Azure",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics
request_count = Counter(
    'api_requests_total',
    'Total API requests',
    ['method', 'endpoint', 'cloud_provider', 'status']
)

request_latency = Histogram(
    'api_request_latency_seconds',
    'API request latency',
    ['method', 'endpoint', 'cloud_provider']
)

prediction_count = Counter(
    'model_predictions_total',
    'Total model predictions',
    ['model_name', 'cloud_provider', 'success']
)

active_clouds = Gauge(
    'active_cloud_providers',
    'Number of active cloud providers'
)

# Add Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


# Pydantic Models
class CloudProviderEnum(str, Enum):
    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"
    AUTO = "auto"


class PredictionRequest(BaseModel):
    model_name: str = Field(..., description="Name of the model to use")
    model_version: Optional[str] = Field(None, description="Model version (defaults to latest)")
    input_data: Dict[str, Any] = Field(..., description="Input data for prediction")
    cloud_provider: CloudProviderEnum = Field(
        CloudProviderEnum.AUTO,
        description="Target cloud provider (auto-selects optimal)"
    )
    latency_threshold_ms: Optional[int] = Field(
        100,
        description="Maximum acceptable latency in milliseconds"
    )


class PredictionResponse(BaseModel):
    prediction: Any
    model_name: str
    model_version: str
    cloud_provider: str
    latency_ms: float
    timestamp: datetime
    request_id: str


class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    clouds: Dict[str, bool]
    version: str


class CloudMetricsResponse(BaseModel):
    cloud_provider: str
    latency_p50: float
    latency_p95: float
    latency_p99: float
    error_rate: float
    requests_per_second: float
    cost_per_1000_requests: float


# Cloud provider selector
class CloudSelector:
    """Intelligent cloud provider selection based on metrics and policies"""

    def __init__(self):
        self.cloud_factory = CloudFactory()
        self.metrics_history: Dict[str, List[CloudMetrics]] = {
            "aws": [],
            "gcp": [],
            "azure": []
        }

    async def select_optimal_cloud(
        self,
        latency_threshold_ms: int,
        model_name: str
    ) -> CloudProvider:
        """
        Select optimal cloud provider based on:
        - Current latency
        - Error rates
        - Cost
        - Model availability
        """

        # Get health status of all clouds
        cloud_health = await self._check_cloud_health()
        healthy_clouds = [
            cloud for cloud, is_healthy in cloud_health.items()
            if is_healthy
        ]

        if not healthy_clouds:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="No healthy cloud providers available"
            )

        # Calculate scores for each cloud
        scores = {}
        for cloud_name in healthy_clouds:
            metrics = await self._get_cloud_metrics(cloud_name)

            # Score based on latency (40%), cost (30%), reliability (30%)
            latency_score = max(0, 1 - (metrics.latency_p95 / latency_threshold_ms))
            cost_score = 1 / (1 + metrics.cost_per_1000_requests)
            reliability_score = 1 - metrics.error_rate

            total_score = (
                0.4 * latency_score +
                0.3 * cost_score +
                0.3 * reliability_score
            )

            scores[cloud_name] = total_score

            logger.info(
                f"Cloud {cloud_name} score: {total_score:.3f} "
                f"(latency: {latency_score:.3f}, cost: {cost_score:.3f}, "
                f"reliability: {reliability_score:.3f})"
            )

        # Select cloud with highest score
        best_cloud = max(scores, key=scores.get)
        logger.info(f"Selected cloud provider: {best_cloud}")

        return CloudProvider[best_cloud.upper()]

    async def _check_cloud_health(self) -> Dict[str, bool]:
        """Check health status of all cloud providers"""
        health_status = {}

        for cloud_name in ["aws", "gcp", "azure"]:
            try:
                provider = self.cloud_factory.create_model_client(
                    CloudProvider[cloud_name.upper()]
                )
                is_healthy = await provider.health_check()
                health_status[cloud_name] = is_healthy
            except Exception as e:
                logger.error(f"Health check failed for {cloud_name}: {e}")
                health_status[cloud_name] = False

        active_clouds.set(sum(health_status.values()))
        return health_status

    async def _get_cloud_metrics(self, cloud_name: str) -> CloudMetrics:
        """Get current metrics for a cloud provider"""
        # In production, this would query actual metrics from Prometheus
        # For now, return mock data
        return CloudMetrics(
            cloud_provider=cloud_name,
            latency_p50=45.0,
            latency_p95=87.0,
            latency_p99=120.0,
            error_rate=0.001,
            requests_per_second=150.0,
            cost_per_1000_requests=0.05
        )


# Initialize cloud selector
cloud_selector = CloudSelector()


# API Routes
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Multi-Cloud ML Platform API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    cloud_health = await cloud_selector._check_cloud_health()

    return HealthResponse(
        status="healthy" if any(cloud_health.values()) else "unhealthy",
        timestamp=datetime.utcnow(),
        clouds=cloud_health,
        version="1.0.0"
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    x_request_id: Optional[str] = Header(None)
):
    """
    Make a prediction using specified or optimal cloud provider
    """
    start_time = time.time()
    request_id = x_request_id or f"req_{int(time.time() * 1000)}"

    try:
        # Select cloud provider
        if request.cloud_provider == CloudProviderEnum.AUTO:
            cloud_provider = await cloud_selector.select_optimal_cloud(
                latency_threshold_ms=request.latency_threshold_ms,
                model_name=request.model_name
            )
        else:
            cloud_provider = CloudProvider[request.cloud_provider.value.upper()]

        logger.info(
            f"Processing prediction request {request_id} on {cloud_provider.value}"
        )

        # Get model client for selected cloud
        model_client = cloud_selector.cloud_factory.create_model_client(cloud_provider)

        # Make prediction
        prediction_result = await model_client.predict(
            model_name=request.model_name,
            model_version=request.model_version or "latest",
            input_data=request.input_data
        )

        latency_ms = (time.time() - start_time) * 1000

        # Record metrics
        prediction_count.labels(
            model_name=request.model_name,
            cloud_provider=cloud_provider.value,
            success="true"
        ).inc()

        request_latency.labels(
            method="POST",
            endpoint="/predict",
            cloud_provider=cloud_provider.value
        ).observe(time.time() - start_time)

        request_count.labels(
            method="POST",
            endpoint="/predict",
            cloud_provider=cloud_provider.value,
            status="200"
        ).inc()

        return PredictionResponse(
            prediction=prediction_result["prediction"],
            model_name=request.model_name,
            model_version=prediction_result.get("version", "latest"),
            cloud_provider=cloud_provider.value,
            latency_ms=latency_ms,
            timestamp=datetime.utcnow(),
            request_id=request_id
        )

    except Exception as e:
        logger.error(f"Prediction failed for request {request_id}: {e}")

        prediction_count.labels(
            model_name=request.model_name,
            cloud_provider=request.cloud_provider.value,
            success="false"
        ).inc()

        request_count.labels(
            method="POST",
            endpoint="/predict",
            cloud_provider=request.cloud_provider.value,
            status="500"
        ).inc()

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.get("/models", response_model=List[Dict[str, Any]])
async def list_models():
    """List all available models across clouds"""
    all_models = []

    for cloud in [CloudProvider.AWS, CloudProvider.GCP, CloudProvider.AZURE]:
        try:
            client = cloud_selector.cloud_factory.create_model_client(cloud)
            models = await client.list_models()

            for model in models:
                model["cloud_provider"] = cloud.value

            all_models.extend(models)
        except Exception as e:
            logger.error(f"Failed to list models from {cloud.value}: {e}")

    return all_models


@app.get("/metrics/clouds", response_model=List[CloudMetricsResponse])
async def get_cloud_metrics():
    """Get performance metrics for all cloud providers"""
    metrics = []

    for cloud_name in ["aws", "gcp", "azure"]:
        cloud_metrics = await cloud_selector._get_cloud_metrics(cloud_name)
        metrics.append(CloudMetricsResponse(**cloud_metrics.__dict__))

    return metrics


@app.post("/models/{model_name}/deploy")
async def deploy_model(
    model_name: str,
    clouds: List[CloudProviderEnum],
    model_version: Optional[str] = None
):
    """Deploy model to specified cloud providers"""
    deployment_results = {}

    for cloud in clouds:
        if cloud == CloudProviderEnum.AUTO:
            continue

        try:
            cloud_provider = CloudProvider[cloud.value.upper()]
            client = cloud_selector.cloud_factory.create_model_client(cloud_provider)

            result = await client.deploy_model(
                model_name=model_name,
                model_version=model_version or "latest"
            )

            deployment_results[cloud.value] = {
                "status": "success",
                "endpoint": result.get("endpoint"),
                "version": result.get("version")
            }
        except Exception as e:
            logger.error(f"Deployment to {cloud.value} failed: {e}")
            deployment_results[cloud.value] = {
                "status": "failed",
                "error": str(e)
            }

    return deployment_results


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Internal server error",
            "timestamp": datetime.utcnow().isoformat()
        }
    )


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
