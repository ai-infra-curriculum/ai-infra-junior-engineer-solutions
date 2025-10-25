# Production-Readiness Checklist for Junior Engineers

**Last Updated**: 2025-10-25
**Level**: Junior
**Purpose**: Comprehensive checklist to ensure your code, containers, and deployments are production-ready

---

## Table of Contents

1. [Introduction](#introduction)
2. [Code Quality Checklist](#code-quality-checklist)
3. [Docker Production Checklist](#docker-production-checklist)
4. [Kubernetes Production Checklist](#kubernetes-production-checklist)
5. [Security Basics](#security-basics)
6. [Monitoring and Logging](#monitoring-and-logging)
7. [Documentation Requirements](#documentation-requirements)
8. [Pre-Deployment Checklist](#pre-deployment-checklist)
9. [Post-Deployment Validation](#post-deployment-validation)
10. [Common Production Issues](#common-production-issues)

---

## Introduction

### What is Production-Ready?

Production-ready code is code that is:
- **Reliable**: Works consistently under expected and unexpected conditions
- **Secure**: Protects data and prevents unauthorized access
- **Observable**: Can be monitored and debugged effectively
- **Maintainable**: Easy to update and troubleshoot
- **Performant**: Meets performance requirements efficiently
- **Documented**: Has clear documentation for operators and developers

### Why This Matters

Moving code to production is a critical responsibility. Production issues can:
- Impact real users
- Cost money (downtime, wasted resources)
- Damage reputation
- Create security vulnerabilities
- Cause data loss

### How to Use This Guide

1. **During Development**: Reference relevant sections as you build
2. **Before Deployment**: Complete the full pre-deployment checklist
3. **After Deployment**: Validate using post-deployment checks
4. **Regularly**: Review and update as you learn

---

## Code Quality Checklist

### Python Code Standards

#### ✅ Testing

```python
# Bad: No tests
def predict(data):
    model = load_model()
    return model.predict(data)

# Good: Well-tested
def predict(data):
    """
    Make predictions on input data.

    Args:
        data: Input features as numpy array or pandas DataFrame

    Returns:
        Predictions as numpy array

    Raises:
        ValueError: If data is empty or has wrong shape
    """
    if data is None or len(data) == 0:
        raise ValueError("Input data cannot be empty")

    model = load_model()
    return model.predict(data)

# Test file: tests/test_inference.py
import pytest
import numpy as np
from src.inference import predict

def test_predict_valid_input():
    """Test prediction with valid input."""
    data = np.array([[1, 2, 3], [4, 5, 6]])
    result = predict(data)
    assert result is not None
    assert len(result) == 2

def test_predict_empty_input():
    """Test prediction handles empty input."""
    with pytest.raises(ValueError, match="cannot be empty"):
        predict(np.array([]))

def test_predict_none_input():
    """Test prediction handles None input."""
    with pytest.raises(ValueError):
        predict(None)
```

**Testing Checklist**:
- [ ] Unit tests cover core functions (target >80% coverage)
- [ ] Integration tests validate component interactions
- [ ] Edge cases are tested (empty inputs, nulls, extremes)
- [ ] Error paths are tested (wrong types, missing data)
- [ ] Tests are automated in CI/CD
- [ ] Tests run fast (<5 minutes for unit tests)
- [ ] Test data is in version control or generated deterministically

**Running Tests**:
```bash
# Install test dependencies
pip install pytest pytest-cov

# Run tests with coverage
pytest tests/ -v --cov=src --cov-report=html

# View coverage report
open htmlcov/index.html
```

#### ✅ Linting and Formatting

```python
# Bad: Inconsistent formatting, no type hints
def process_data(data,model_path,output_path):
    result=[]
    for item in data:
        x=transform(item)
        result.append(x)
    return result

# Good: Formatted, type hints, clear
from typing import List, Dict, Any
from pathlib import Path

def process_data(
    data: List[Dict[str, Any]],
    model_path: Path,
    output_path: Path
) -> List[Dict[str, Any]]:
    """
    Process input data through transformation pipeline.

    Args:
        data: List of data dictionaries to process
        model_path: Path to the model file
        output_path: Path to save processed results

    Returns:
        List of transformed data dictionaries
    """
    result = []
    for item in data:
        transformed = transform(item)
        result.append(transformed)

    return result
```

**Linting Checklist**:
- [ ] Code follows PEP 8 style guide
- [ ] Type hints on function signatures
- [ ] No unused imports or variables
- [ ] Line length < 88-100 characters
- [ ] Consistent naming conventions
- [ ] All functions have docstrings
- [ ] No security issues (bandit scan passes)

**Setup Linting**:
```bash
# Install tools
pip install black flake8 mypy bandit isort

# Format code
black src/
isort src/

# Check style
flake8 src/ --max-line-length=100

# Type check
mypy src/ --ignore-missing-imports

# Security scan
bandit -r src/
```

**Pre-commit Hooks** (`.pre-commit-config.yaml`):
```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
        language_version: python3.9

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: [--max-line-length=100]

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.3.0
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
```

#### ✅ Error Handling

```python
# Bad: No error handling
def load_config(path):
    with open(path) as f:
        return json.load(f)

# Good: Comprehensive error handling
import json
import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)

class ConfigError(Exception):
    """Raised when configuration loading fails."""
    pass

def load_config(path: Path) -> Dict[str, Any]:
    """
    Load configuration from JSON file.

    Args:
        path: Path to configuration file

    Returns:
        Configuration dictionary

    Raises:
        ConfigError: If file cannot be read or parsed
    """
    try:
        if not path.exists():
            raise ConfigError(f"Configuration file not found: {path}")

        if not path.is_file():
            raise ConfigError(f"Path is not a file: {path}")

        with open(path, 'r') as f:
            config = json.load(f)

        if not config:
            raise ConfigError("Configuration file is empty")

        logger.info(f"Successfully loaded config from {path}")
        return config

    except json.JSONDecodeError as e:
        raise ConfigError(f"Invalid JSON in config file: {e}") from e
    except PermissionError as e:
        raise ConfigError(f"Permission denied reading config: {e}") from e
    except Exception as e:
        logger.error(f"Unexpected error loading config: {e}")
        raise ConfigError(f"Failed to load config: {e}") from e
```

**Error Handling Checklist**:
- [ ] All I/O operations have error handling
- [ ] External API calls have timeout and retry logic
- [ ] User-facing errors have helpful messages
- [ ] Errors are logged with appropriate context
- [ ] Critical errors trigger alerts
- [ ] Graceful degradation when possible
- [ ] No bare `except:` clauses

#### ✅ Logging

```python
# Bad: Print statements
def process_request(request):
    print(f"Got request: {request}")
    result = do_something(request)
    print(f"Result: {result}")
    return result

# Good: Structured logging
import logging
import json
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def process_request(request: Dict[str, Any]) -> Dict[str, Any]:
    """Process incoming request."""
    request_id = request.get('id', 'unknown')

    logger.info(
        "Processing request",
        extra={
            'request_id': request_id,
            'user_id': request.get('user_id'),
            'action': request.get('action')
        }
    )

    try:
        result = do_something(request)

        logger.info(
            "Request processed successfully",
            extra={
                'request_id': request_id,
                'processing_time_ms': result.get('time_ms')
            }
        )

        return result

    except Exception as e:
        logger.error(
            "Request processing failed",
            extra={
                'request_id': request_id,
                'error': str(e),
                'error_type': type(e).__name__
            },
            exc_info=True
        )
        raise
```

**Logging Best Practices**:
- [ ] Use logging module, not print()
- [ ] Set appropriate log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- [ ] Include context in log messages (request IDs, user IDs, etc.)
- [ ] Log to both file and stdout (for containers)
- [ ] Use structured logging for easier parsing
- [ ] Don't log sensitive data (passwords, tokens, PII)
- [ ] Configure log rotation for file logs

**Log Levels Guide**:
```python
# DEBUG: Detailed information for diagnosing problems
logger.debug(f"Processing item {item_id} with config {config}")

# INFO: Confirmation that things are working
logger.info(f"Started processing batch of {len(items)} items")

# WARNING: Something unexpected but handled
logger.warning(f"API rate limit reached, retrying in {delay}s")

# ERROR: Error that prevents function from completing
logger.error(f"Failed to process item {item_id}: {error}")

# CRITICAL: Serious error that may crash the application
logger.critical(f"Database connection lost, cannot continue")
```

#### ✅ Configuration Management

```python
# Bad: Hardcoded values
def connect_db():
    return psycopg2.connect(
        host="localhost",
        port=5432,
        user="admin",
        password="password123"
    )

# Good: Environment-based configuration
import os
from typing import Optional
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    """Application configuration."""
    # Database
    db_host: str
    db_port: int
    db_user: str
    db_password: str
    db_name: str

    # API
    api_port: int
    api_workers: int

    # Model
    model_path: Path
    batch_size: int

    # Logging
    log_level: str

    @classmethod
    def from_env(cls) -> 'Config':
        """Load configuration from environment variables."""
        return cls(
            # Database
            db_host=os.getenv('DB_HOST', 'localhost'),
            db_port=int(os.getenv('DB_PORT', '5432')),
            db_user=os.getenv('DB_USER', 'postgres'),
            db_password=os.environ['DB_PASSWORD'],  # Required, no default
            db_name=os.getenv('DB_NAME', 'ml_app'),

            # API
            api_port=int(os.getenv('API_PORT', '8000')),
            api_workers=int(os.getenv('API_WORKERS', '4')),

            # Model
            model_path=Path(os.getenv('MODEL_PATH', '/models/model.pkl')),
            batch_size=int(os.getenv('BATCH_SIZE', '32')),

            # Logging
            log_level=os.getenv('LOG_LEVEL', 'INFO')
        )

    def validate(self):
        """Validate configuration."""
        if not self.model_path.exists():
            raise ValueError(f"Model not found at {self.model_path}")

        if self.batch_size < 1:
            raise ValueError("Batch size must be positive")

        if self.api_workers < 1:
            raise ValueError("API workers must be positive")

# Usage
config = Config.from_env()
config.validate()
```

**Configuration Checklist**:
- [ ] No hardcoded credentials or secrets
- [ ] Environment variables for configuration
- [ ] Sensible defaults for development
- [ ] Required values fail fast if missing
- [ ] Configuration is validated on startup
- [ ] Different configs for dev/staging/production
- [ ] Configuration documented in README

**Example `.env` file**:
```bash
# .env (DO NOT COMMIT)
DB_PASSWORD=secure_password_here
API_KEY=your_api_key_here

# .env.example (Safe to commit)
DB_HOST=localhost
DB_PORT=5432
DB_USER=postgres
DB_PASSWORD=your_password_here
DB_NAME=ml_app
API_PORT=8000
API_WORKERS=4
MODEL_PATH=/models/model.pkl
LOG_LEVEL=INFO
```

---

## Docker Production Checklist

### ✅ Dockerfile Best Practices

```dockerfile
# Bad Dockerfile
FROM python:latest
COPY . .
RUN pip install -r requirements.txt
CMD python app.py

# Good Dockerfile
# Use specific version
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

# Copy requirements first (better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=appuser:appuser . .

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Expose port (documentation)
EXPOSE 8000

# Use exec form for proper signal handling
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Dockerfile Checklist**:
- [ ] Use specific base image versions (not `latest`)
- [ ] Use minimal base images (`slim`, `alpine` when appropriate)
- [ ] Multi-stage builds for compiled languages
- [ ] Layer caching optimized (requirements before code)
- [ ] Non-root user for runtime
- [ ] Health check defined
- [ ] Proper signal handling (exec form CMD)
- [ ] No secrets in image layers
- [ ] `.dockerignore` file configured
- [ ] Image size optimized (<500MB for Python apps)

**Multi-Stage Example**:
```dockerfile
# Build stage
FROM python:3.9 as builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Runtime stage
FROM python:3.9-slim

WORKDIR /app

# Copy dependencies from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy application
COPY . .

# Non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

CMD ["python", "app.py"]
```

**`.dockerignore` Example**:
```
# .dockerignore
.git
.gitignore
.github
*.md
tests/
*.pyc
__pycache__
.pytest_cache
.coverage
htmlcov/
.env
.env.*
*.log
data/
models/*.pkl  # Don't copy large model files
notebooks/
docs/
```

### ✅ Health Checks

```python
# app.py - FastAPI example
from fastapi import FastAPI, status
from fastapi.responses import JSONResponse
import psycopg2
import logging

app = FastAPI()
logger = logging.getLogger(__name__)

@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    Returns 200 if service is healthy, 503 if unhealthy.
    """
    checks = {
        "status": "healthy",
        "checks": {}
    }

    # Check database connection
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.close()
        conn.close()
        checks["checks"]["database"] = "ok"
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        checks["status"] = "unhealthy"
        checks["checks"]["database"] = "error"

    # Check model loaded
    try:
        if app.state.model is None:
            raise Exception("Model not loaded")
        checks["checks"]["model"] = "ok"
    except Exception as e:
        logger.error(f"Model health check failed: {e}")
        checks["status"] = "unhealthy"
        checks["checks"]["model"] = "error"

    # Return appropriate status code
    status_code = (
        status.HTTP_200_OK
        if checks["status"] == "healthy"
        else status.HTTP_503_SERVICE_UNAVAILABLE
    )

    return JSONResponse(content=checks, status_code=status_code)

@app.get("/ready")
async def readiness_check():
    """
    Readiness check endpoint.
    Returns 200 if service is ready to accept traffic.
    """
    # Check if initialization is complete
    if not app.state.initialized:
        return JSONResponse(
            content={"status": "not ready"},
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE
        )

    return {"status": "ready"}
```

### ✅ Resource Limits

```yaml
# docker-compose.yml
version: '3.8'

services:
  web:
    image: myapp:latest
    deploy:
      resources:
        limits:
          cpus: '1.0'      # Max 1 CPU
          memory: 1G        # Max 1GB RAM
        reservations:
          cpus: '0.5'       # Reserved 0.5 CPU
          memory: 512M      # Reserved 512MB RAM
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3

    environment:
      - DB_HOST=postgres
      - WORKERS=4

    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 3s
      retries: 3
      start_period: 40s

    ports:
      - "8000:8000"

    networks:
      - app-network

    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

  postgres:
    image: postgres:14
    environment:
      - POSTGRES_PASSWORD_FILE=/run/secrets/db_password
    secrets:
      - db_password
    volumes:
      - postgres-data:/var/lib/postgresql/data
    deploy:
      resources:
        limits:
          memory: 512M

secrets:
  db_password:
    file: ./secrets/db_password.txt

volumes:
  postgres-data:

networks:
  app-network:
    driver: bridge
```

**Docker Compose Checklist**:
- [ ] Resource limits defined
- [ ] Health checks configured
- [ ] Restart policies set
- [ ] Secrets management configured
- [ ] Logging configured with rotation
- [ ] Networks properly isolated
- [ ] Volumes for persistent data
- [ ] Environment variables documented

---

## Kubernetes Production Checklist

### ✅ Deployment Configuration

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-api
  namespace: production
  labels:
    app: ml-api
    version: v1.0.0
    tier: backend
spec:
  replicas: 3  # Multiple replicas for high availability

  # Deployment strategy
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1        # Max pods above desired during update
      maxUnavailable: 0  # Keep all pods available during update

  selector:
    matchLabels:
      app: ml-api

  template:
    metadata:
      labels:
        app: ml-api
        version: v1.0.0
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
        prometheus.io/path: "/metrics"

    spec:
      # Service account for RBAC
      serviceAccountName: ml-api-sa

      # Security context for pod
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 1000

      # Init container for migrations or setup
      initContainers:
      - name: wait-for-db
        image: busybox:1.35
        command: ['sh', '-c', 'until nc -z postgres 5432; do echo waiting for db; sleep 2; done']

      containers:
      - name: ml-api
        image: myregistry.io/ml-api:v1.0.0
        imagePullPolicy: IfNotPresent

        ports:
        - name: http
          containerPort: 8000
          protocol: TCP

        # Environment variables
        env:
        - name: DB_HOST
          value: postgres
        - name: DB_PORT
          value: "5432"
        - name: DB_NAME
          valueFrom:
            configMapKeyRef:
              name: ml-api-config
              key: db_name
        - name: DB_PASSWORD
          valueFrom:
            secretKeyRef:
              name: ml-api-secrets
              key: db_password
        - name: LOG_LEVEL
          value: INFO

        # Resource requests and limits
        resources:
          requests:
            cpu: 500m       # Guaranteed 0.5 CPU
            memory: 512Mi   # Guaranteed 512MB
          limits:
            cpu: 1000m      # Max 1 CPU
            memory: 1Gi     # Max 1GB

        # Liveness probe - restart if unhealthy
        livenessProbe:
          httpGet:
            path: /health
            port: http
          initialDelaySeconds: 60
          periodSeconds: 10
          timeoutSeconds: 3
          failureThreshold: 3

        # Readiness probe - remove from service if not ready
        readinessProbe:
          httpGet:
            path: /ready
            port: http
          initialDelaySeconds: 10
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3

        # Startup probe - for slow-starting containers
        startupProbe:
          httpGet:
            path: /health
            port: http
          initialDelaySeconds: 0
          periodSeconds: 10
          timeoutSeconds: 3
          failureThreshold: 30  # 30 * 10 = 300s max startup time

        # Volume mounts
        volumeMounts:
        - name: config
          mountPath: /app/config
          readOnly: true
        - name: models
          mountPath: /models
          readOnly: true

        # Security context for container
        securityContext:
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
          runAsNonRoot: true
          runAsUser: 1000
          capabilities:
            drop:
            - ALL

      # Volumes
      volumes:
      - name: config
        configMap:
          name: ml-api-config
      - name: models
        persistentVolumeClaim:
          claimName: models-pvc

      # Pod affinity for distribution across nodes
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: app
                  operator: In
                  values:
                  - ml-api
              topologyKey: kubernetes.io/hostname
```

**Kubernetes Deployment Checklist**:
- [ ] Multiple replicas for high availability (minimum 3)
- [ ] Resource requests and limits defined
- [ ] Liveness probe configured
- [ ] Readiness probe configured
- [ ] Startup probe for slow-starting apps
- [ ] Rolling update strategy configured
- [ ] Pod disruption budget defined
- [ ] Security context with non-root user
- [ ] ConfigMaps for configuration
- [ ] Secrets for sensitive data
- [ ] Labels and annotations for observability
- [ ] Pod anti-affinity for distribution

### ✅ Service Configuration

```yaml
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: ml-api
  namespace: production
  labels:
    app: ml-api
  annotations:
    service.beta.kubernetes.io/aws-load-balancer-type: "nlb"
spec:
  type: ClusterIP  # Use LoadBalancer or NodePort as needed
  selector:
    app: ml-api
  ports:
  - name: http
    port: 80
    targetPort: http
    protocol: TCP
  sessionAffinity: None
```

### ✅ ConfigMap and Secrets

```yaml
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: ml-api-config
  namespace: production
data:
  db_name: ml_production
  log_level: INFO
  batch_size: "32"
  workers: "4"

---
# secret.yaml - Create from file, not in git
apiVersion: v1
kind: Secret
metadata:
  name: ml-api-secrets
  namespace: production
type: Opaque
stringData:
  db_password: ""  # Fill from CI/CD or kubectl
  api_key: ""      # Fill from CI/CD or kubectl
```

**Create secrets securely**:
```bash
# From literal values
kubectl create secret generic ml-api-secrets \
  --from-literal=db_password='your_password' \
  --from-literal=api_key='your_api_key' \
  -n production

# From files
kubectl create secret generic ml-api-secrets \
  --from-file=db_password=./secrets/db_password.txt \
  --from-file=api_key=./secrets/api_key.txt \
  -n production
```

### ✅ Pod Disruption Budget

```yaml
# pdb.yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: ml-api-pdb
  namespace: production
spec:
  minAvailable: 2  # At least 2 pods must be available
  selector:
    matchLabels:
      app: ml-api
```

### ✅ Horizontal Pod Autoscaler

```yaml
# hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ml-api-hpa
  namespace: production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ml-api

  minReplicas: 3
  maxReplicas: 10

  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70

  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80

  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300  # Wait 5 min before scaling down
      policies:
      - type: Percent
        value: 50  # Scale down max 50% of pods at a time
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 0
      policies:
      - type: Percent
        value: 100  # Can double pods at once
        periodSeconds: 15
```

---

## Security Basics

### ✅ Secrets Management

**What NOT to Do**:
```python
# ❌ BAD: Hardcoded credentials
API_KEY = "sk-1234567890abcdef"
DATABASE_URL = "postgresql://admin:password123@localhost/db"

# ❌ BAD: Credentials in code repository
# config.py
CREDENTIALS = {
    "username": "admin",
    "password": "super_secret"
}
```

**What to Do**:
```python
# ✅ GOOD: Environment variables
import os

API_KEY = os.environ["API_KEY"]  # Required
DATABASE_URL = os.getenv("DATABASE_URL")  # Optional with None default

# ✅ GOOD: Secrets from file (Kubernetes)
def load_secret(secret_path: str) -> str:
    """Load secret from mounted file."""
    with open(secret_path, 'r') as f:
        return f.read().strip()

db_password = load_secret("/run/secrets/db_password")

# ✅ GOOD: Cloud secrets manager
import boto3

def get_secret(secret_name: str) -> str:
    """Retrieve secret from AWS Secrets Manager."""
    client = boto3.client('secretsmanager')
    response = client.get_secret_value(SecretId=secret_name)
    return response['SecretString']
```

**Security Checklist**:
- [ ] No hardcoded credentials in code
- [ ] No credentials in Git repository
- [ ] `.env` files in `.gitignore`
- [ ] `.env.example` (without secrets) committed
- [ ] Secrets rotated regularly
- [ ] Least privilege principle applied
- [ ] Secrets encrypted at rest and in transit

### ✅ Input Validation

```python
from typing import Optional
from pydantic import BaseModel, Field, validator
from fastapi import FastAPI, HTTPException

app = FastAPI()

class PredictionRequest(BaseModel):
    """Validated prediction request."""

    age: int = Field(..., ge=0, le=120, description="Age in years")
    income: float = Field(..., ge=0, description="Annual income")
    category: str = Field(..., min_length=1, max_length=50)

    @validator('category')
    def validate_category(cls, v):
        """Validate category is from allowed list."""
        allowed = {'A', 'B', 'C', 'D'}
        if v not in allowed:
            raise ValueError(f"Category must be one of {allowed}")
        return v

@app.post("/predict")
async def predict(request: PredictionRequest):
    """
    Make prediction with validated input.
    Pydantic automatically validates the request.
    """
    try:
        result = model.predict({
            'age': request.age,
            'income': request.income,
            'category': request.category
        })
        return {"prediction": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### ✅ HTTPS/TLS

**Development** (self-signed):
```bash
# Generate self-signed certificate
openssl req -x509 -newkey rsa:4096 -nodes \
  -out cert.pem -keyout key.pem -days 365
```

**Production** (Let's Encrypt with Cert-Manager):
```yaml
# certificate.yaml
apiVersion: cert-manager.io/v1
kind: Certificate
metadata:
  name: ml-api-tls
  namespace: production
spec:
  secretName: ml-api-tls-secret
  issuerRef:
    name: letsencrypt-prod
    kind: ClusterIssuer
  dnsNames:
  - api.example.com
  - www.api.example.com
```

**Ingress with TLS**:
```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ml-api-ingress
  namespace: production
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
spec:
  tls:
  - hosts:
    - api.example.com
    secretName: ml-api-tls-secret
  rules:
  - host: api.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: ml-api
            port:
              number: 80
```

### ✅ Network Policies

```yaml
# network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: ml-api-network-policy
  namespace: production
spec:
  podSelector:
    matchLabels:
      app: ml-api

  policyTypes:
  - Ingress
  - Egress

  ingress:
  # Allow traffic from ingress controller
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8000

  egress:
  # Allow DNS
  - to:
    - namespaceSelector:
        matchLabels:
          name: kube-system
    ports:
    - protocol: UDP
      port: 53

  # Allow database access
  - to:
    - podSelector:
        matchLabels:
          app: postgres
    ports:
    - protocol: TCP
      port: 5432

  # Allow external HTTPS
  - to:
    - namespaceSelector: {}
    ports:
    - protocol: TCP
      port: 443
```

---

## Monitoring and Logging

### ✅ Application Metrics

```python
# metrics.py
from prometheus_client import Counter, Histogram, Gauge, make_asgi_app
from fastapi import FastAPI
import time

app = FastAPI()

# Metrics
request_count = Counter(
    'api_requests_total',
    'Total API requests',
    ['method', 'endpoint', 'status']
)

request_duration = Histogram(
    'api_request_duration_seconds',
    'API request duration',
    ['method', 'endpoint']
)

prediction_count = Counter(
    'predictions_total',
    'Total predictions made',
    ['model_version']
)

model_latency = Histogram(
    'model_inference_duration_seconds',
    'Model inference duration',
    ['model_version']
)

active_requests = Gauge(
    'api_active_requests',
    'Number of active requests'
)

@app.middleware("http")
async def metrics_middleware(request, call_next):
    """Track request metrics."""
    active_requests.inc()

    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time

    active_requests.dec()

    request_count.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()

    request_duration.labels(
        method=request.method,
        endpoint=request.url.path
    ).observe(duration)

    return response

@app.post("/predict")
async def predict(data: dict):
    """Make prediction with metrics."""
    model_version = "v1.0.0"

    start_time = time.time()
    result = model.predict(data)
    duration = time.time() - start_time

    prediction_count.labels(model_version=model_version).inc()
    model_latency.labels(model_version=model_version).observe(duration)

    return {"prediction": result}

# Add Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)
```

### ✅ Structured Logging

```python
# structured_logging.py
import logging
import json
import sys
from typing import Any, Dict

class JSONFormatter(logging.Formatter):
    """Format logs as JSON."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data: Dict[str, Any] = {
            'timestamp': self.formatTime(record),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
        }

        # Add extra fields
        if hasattr(record, 'request_id'):
            log_data['request_id'] = record.request_id

        if hasattr(record, 'user_id'):
            log_data['user_id'] = record.user_id

        # Add exception info
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)

        return json.dumps(log_data)

# Configure logging
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(JSONFormatter())

logger = logging.getLogger('ml_api')
logger.setLevel(logging.INFO)
logger.addHandler(handler)

# Usage
logger.info(
    'Request processed',
    extra={
        'request_id': 'req-123',
        'user_id': 'user-456',
        'duration_ms': 150
    }
)
```

### ✅ Monitoring Dashboards

**Key Metrics to Monitor**:

1. **Application Metrics**:
   - Request rate (requests/second)
   - Error rate (errors/total requests)
   - Response time (p50, p95, p99)
   - Active connections

2. **Infrastructure Metrics**:
   - CPU usage
   - Memory usage
   - Disk I/O
   - Network traffic

3. **Business Metrics**:
   - Predictions per minute
   - Model accuracy drift
   - User activity

4. **SLI/SLO Tracking**:
   - Availability (99.9% uptime)
   - Latency (p95 < 200ms)
   - Error rate (<0.1%)

---

## Documentation Requirements

### ✅ README.md

```markdown
# ML Prediction API

Production-ready ML model serving API.

## Quick Start

```bash
# Using Docker
docker-compose up

# Using Kubernetes
kubectl apply -f k8s/
```

## Features

- RESTful API for model predictions
- Automatic request validation
- Prometheus metrics
- Health check endpoints
- Horizontal auto-scaling

## API Documentation

### Endpoints

#### POST /predict
Make a prediction.

**Request**:
```json
{
  "age": 35,
  "income": 50000,
  "category": "A"
}
```

**Response**:
```json
{
  "prediction": 0.85,
  "model_version": "v1.0.0"
}
```

#### GET /health
Health check endpoint.

## Configuration

Environment variables:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| DB_HOST | Yes | - | Database hostname |
| DB_PORT | No | 5432 | Database port |
| DB_PASSWORD | Yes | - | Database password |
| LOG_LEVEL | No | INFO | Logging level |

## Development

```bash
# Install dependencies
pip install -r requirements-dev.txt

# Run tests
pytest

# Run locally
uvicorn app.main:app --reload
```

## Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed instructions.

## Monitoring

- Metrics: http://localhost:8000/metrics
- Grafana: http://localhost:3000

## Troubleshooting

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

## License

MIT
```

### ✅ API Documentation

Use automatic documentation tools:

```python
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(
    title="ML Prediction API",
    description="API for making ML predictions",
    version="1.0.0",
    docs_url="/docs",  # Swagger UI
    redoc_url="/redoc"  # ReDoc
)

class PredictionRequest(BaseModel):
    """Request model for predictions."""

    age: int = Field(..., ge=0, le=120, description="Age in years", example=35)
    income: float = Field(..., ge=0, description="Annual income", example=50000)
    category: str = Field(..., description="Category", example="A")

    class Config:
        schema_extra = {
            "example": {
                "age": 35,
                "income": 50000,
                "category": "A"
            }
        }

@app.post("/predict", tags=["predictions"])
async def predict(request: PredictionRequest):
    """
    Make a prediction.

    - **age**: Customer age (0-120)
    - **income**: Annual income (>= 0)
    - **category**: Customer category (A, B, C, or D)

    Returns prediction score between 0 and 1.
    """
    # Implementation
    pass
```

Access docs at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## Pre-Deployment Checklist

### Phase 1: Code Review

- [ ] All code reviewed and approved
- [ ] Tests passing (>80% coverage)
- [ ] No known bugs
- [ ] Performance tested under load
- [ ] Security scan completed (no high/critical issues)
- [ ] Dependencies up to date
- [ ] Linting passes
- [ ] Type checking passes

### Phase 2: Configuration

- [ ] Environment variables documented
- [ ] Secrets configured in deployment environment
- [ ] Database migrations tested
- [ ] Configuration validated for production
- [ ] Feature flags configured
- [ ] API keys and credentials rotated

### Phase 3: Infrastructure

- [ ] Resource limits appropriate
- [ ] Scaling policies configured
- [ ] Backup strategy in place
- [ ] Disaster recovery plan documented
- [ ] Network policies configured
- [ ] TLS certificates valid

### Phase 4: Observability

- [ ] Logging configured and tested
- [ ] Metrics collection working
- [ ] Dashboards created
- [ ] Alerts configured
- [ ] On-call rotation set up
- [ ] Runbooks created

### Phase 5: Documentation

- [ ] README updated
- [ ] API documentation current
- [ ] Deployment guide complete
- [ ] Troubleshooting guide available
- [ ] Architecture diagrams updated
- [ ] Changelog updated

### Phase 6: Testing

- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] End-to-end tests pass
- [ ] Load tests pass
- [ ] Security tests pass
- [ ] Smoke tests defined for production

### Phase 7: Deployment Plan

- [ ] Deployment steps documented
- [ ] Rollback plan prepared
- [ ] Database backup taken
- [ ] Maintenance window scheduled
- [ ] Stakeholders notified
- [ ] Post-deployment validation plan ready

---

## Post-Deployment Validation

### Immediate Checks (0-5 minutes)

```bash
# Check deployment status
kubectl rollout status deployment/ml-api -n production

# Verify pods are running
kubectl get pods -n production -l app=ml-api

# Check pod logs
kubectl logs -n production -l app=ml-api --tail=100

# Test health endpoint
curl -f https://api.example.com/health

# Check metrics
curl https://api.example.com/metrics | grep api_requests_total
```

**Checklist**:
- [ ] All pods running
- [ ] Health checks passing
- [ ] No error logs
- [ ] Metrics being collected
- [ ] Old pods terminated successfully

### Short-term Validation (5-30 minutes)

```bash
# Monitor error rate
kubectl logs -n production -l app=ml-api -f | grep ERROR

# Check resource usage
kubectl top pods -n production -l app=ml-api

# Test API endpoints
curl -X POST https://api.example.com/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 35, "income": 50000, "category": "A"}'
```

**Checklist**:
- [ ] Error rate < baseline
- [ ] Response times < SLA
- [ ] No memory leaks
- [ ] CPU usage normal
- [ ] All features working

### Extended Monitoring (1-24 hours)

- [ ] Monitor dashboards
- [ ] Check alerts
- [ ] Review logs for anomalies
- [ ] Verify database performance
- [ ] Check external integrations
- [ ] Monitor user feedback
- [ ] Review business metrics

### Rollback Criteria

Rollback immediately if:
- Error rate > 5%
- P95 latency > 2x baseline
- Critical feature broken
- Security vulnerability detected
- Data corruption detected
- Database connection issues
- Memory leaks causing OOM

**Rollback command**:
```bash
# Kubernetes rollback
kubectl rollout undo deployment/ml-api -n production

# Verify rollback
kubectl rollout status deployment/ml-api -n production
```

---

## Common Production Issues

### Issue: Pod Keeps Restarting

**Symptoms**:
```
NAME                      READY   STATUS             RESTARTS   AGE
ml-api-7d8c9f5b6d-abcde   0/1     CrashLoopBackOff   5          3m
```

**Debug**:
```bash
# Check pod logs
kubectl logs ml-api-7d8c9f5b6d-abcde

# Check previous container logs
kubectl logs ml-api-7d8c9f5b6d-abcde --previous

# Describe pod for events
kubectl describe pod ml-api-7d8c9f5b6d-abcde
```

**Common Causes**:
- [ ] Missing environment variables
- [ ] Wrong image version
- [ ] Health check failing too early
- [ ] Insufficient resources
- [ ] Port already in use

### Issue: High Memory Usage

**Debug**:
```bash
# Check memory usage
kubectl top pods -n production

# Check resource limits
kubectl describe pod <pod-name> | grep -A 5 Limits
```

**Common Causes**:
- [ ] Memory leak in application
- [ ] Loading too much data
- [ ] Caching without limits
- [ ] Resource limits too low

**Fix**:
```python
# Add memory profiling
import tracemalloc

tracemalloc.start()

# Your code here

snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')

for stat in top_stats[:10]:
    print(stat)
```

### Issue: Slow Response Times

**Debug**:
```bash
# Check metrics
curl https://api.example.com/metrics | grep duration

# Profile with cProfile
python -m cProfile -o profile.stats app.py
```

**Common Causes**:
- [ ] N+1 database queries
- [ ] No database indexes
- [ ] Synchronous I/O in async code
- [ ] Large payload processing
- [ ] External API timeouts

---

## Final Checklist

Before marking as production-ready:

**Code Quality**:
- [ ] Tests >80% coverage
- [ ] Linting passes
- [ ] Type hints present
- [ ] Documentation complete

**Security**:
- [ ] No hardcoded secrets
- [ ] Input validation
- [ ] HTTPS enabled
- [ ] Security scan clean

**Reliability**:
- [ ] Health checks configured
- [ ] Error handling comprehensive
- [ ] Logging structured
- [ ] Retries for transient failures

**Observability**:
- [ ] Metrics exposed
- [ ] Dashboards created
- [ ] Alerts configured
- [ ] Logs centralized

**Operations**:
- [ ] Runbooks created
- [ ] On-call rotation
- [ ] Rollback plan
- [ ] Backup strategy

**Performance**:
- [ ] Load tested
- [ ] Resource limits appropriate
- [ ] Auto-scaling configured
- [ ] Caching implemented

---

## Additional Resources

### Documentation
- [The Twelve-Factor App](https://12factor.net/)
- [Kubernetes Best Practices](https://kubernetes.io/docs/concepts/configuration/overview/)
- [Docker Security Best Practices](https://docs.docker.com/develop/security-best-practices/)

### Tools
- [Safety](https://github.com/pyupio/safety) - Check dependencies for vulnerabilities
- [Bandit](https://github.com/PyCQA/bandit) - Security linting
- [Trivy](https://github.com/aquasecurity/trivy) - Container security scanning

### Monitoring
- [Prometheus](https://prometheus.io/) - Metrics collection
- [Grafana](https://grafana.com/) - Dashboards
- [Elasticsearch/Kibana](https://www.elastic.co/) - Log aggregation

---

**Remember**: Production readiness is not a one-time checklist. It's an ongoing practice of maintaining quality, security, and reliability.
