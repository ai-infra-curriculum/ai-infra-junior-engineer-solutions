# Implementation Guide: Production ML Deployment with Docker

## Overview

This comprehensive guide walks you through building production-ready ML deployment infrastructure using Docker, covering deployment strategies, monitoring, auto-scaling, and CI/CD integration. You'll learn to deploy ML models with zero downtime, implement A/B testing, monitor model drift, and handle the complete ML model lifecycle in production.

**Duration**: 5-6 hours
**Difficulty**: Advanced
**Prerequisites**: Docker fundamentals, ML basics, Python proficiency

## Table of Contents

1. [Production Deployment Strategies](#1-production-deployment-strategies)
2. [Health Checks and Readiness Probes](#2-health-checks-and-readiness-probes)
3. [Logging and Monitoring Integration](#3-logging-and-monitoring-integration)
4. [Auto-Scaling and Load Balancing](#4-auto-scaling-and-load-balancing)
5. [Rollback Procedures](#5-rollback-procedures)
6. [CI/CD Integration](#6-cicd-integration)
7. [Complete ML Production Pipeline](#7-complete-ml-production-pipeline)

---

## 1. Production Deployment Strategies

### 1.1 Blue-Green Deployment for ML Models

Blue-green deployment allows instant rollback and zero-downtime updates by maintaining two identical production environments.

**Step 1: Create Blue Environment**

```yaml
# docker-compose.blue.yml
version: '3.8'

services:
  ml-api-blue:
    build:
      context: .
      dockerfile: Dockerfile.production
    container_name: ml-api-blue
    environment:
      - MODEL_VERSION=v1.0.0
      - ENVIRONMENT=blue
      - MODEL_PATH=/models/model-v1.onnx
    ports:
      - "8001:8000"
    volumes:
      - ./models/v1:/models:ro
    networks:
      - ml-network
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 2G
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 10s
      timeout: 3s
      retries: 3

networks:
  ml-network:
    driver: bridge
```

**Step 2: Create Green Environment**

```yaml
# docker-compose.green.yml
version: '3.8'

services:
  ml-api-green:
    build:
      context: .
      dockerfile: Dockerfile.production
    container_name: ml-api-green
    environment:
      - MODEL_VERSION=v2.0.0
      - ENVIRONMENT=green
      - MODEL_PATH=/models/model-v2.onnx
    ports:
      - "8002:8000"
    volumes:
      - ./models/v2:/models:ro
    networks:
      - ml-network
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 2G
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 10s
      timeout: 3s
      retries: 3

networks:
  ml-network:
    external: true
```

**Step 3: Load Balancer Configuration**

```nginx
# nginx-blue-green.conf
upstream ml_api {
    # Initially route all traffic to blue
    server ml-api-blue:8000 weight=100;
    server ml-api-green:8000 weight=0;
}

server {
    listen 80;
    server_name ml-api.production.com;

    location / {
        proxy_pass http://ml_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;

        # ML inference timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
    }

    location /health {
        access_log off;
        proxy_pass http://ml_api/health;
    }

    location /metrics {
        proxy_pass http://ml_api/metrics;
    }
}
```

**Step 4: Deployment Script**

```python
# deploy_blue_green.py
import subprocess
import time
import requests
import sys
from typing import Literal

class BlueGreenDeployer:
    def __init__(self, load_balancer_url: str = "http://localhost"):
        self.lb_url = load_balancer_url
        self.nginx_config = "/etc/nginx/conf.d/ml-api.conf"

    def deploy(self, target: Literal['blue', 'green'], model_version: str):
        """Deploy new version to inactive environment"""
        print(f"[1/6] Deploying {target} environment with model {model_version}")

        # Deploy new version
        compose_file = f"docker-compose.{target}.yml"
        cmd = f"docker-compose -f {compose_file} up -d --build"
        subprocess.run(cmd, shell=True, check=True)

        # Wait for service to be healthy
        print(f"[2/6] Waiting for {target} environment to be healthy...")
        if not self.wait_for_health(target):
            print(f"ERROR: {target} environment failed health checks")
            sys.exit(1)

        # Run smoke tests
        print(f"[3/6] Running smoke tests on {target} environment...")
        if not self.run_smoke_tests(target):
            print(f"ERROR: Smoke tests failed on {target} environment")
            sys.exit(1)

        # Perform model validation
        print(f"[4/6] Validating model performance on {target}...")
        if not self.validate_model_performance(target):
            print(f"ERROR: Model validation failed on {target} environment")
            sys.exit(1)

        # Switch traffic
        print(f"[5/6] Switching traffic to {target} environment...")
        self.switch_traffic(target)

        # Monitor for issues
        print(f"[6/6] Monitoring {target} environment for 60 seconds...")
        if not self.monitor_deployment(target, duration=60):
            print(f"WARNING: Issues detected, rolling back to previous version")
            other = 'green' if target == 'blue' else 'blue'
            self.switch_traffic(other)
            sys.exit(1)

        print(f"SUCCESS: Deployment to {target} completed successfully")

    def wait_for_health(self, target: str, timeout: int = 120) -> bool:
        """Wait for service to pass health checks"""
        port = 8001 if target == 'blue' else 8002
        url = f"http://localhost:{port}/health"

        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    print(f"  ✓ Health check passed for {target}")
                    return True
            except requests.RequestException:
                pass
            time.sleep(5)

        return False

    def run_smoke_tests(self, target: str) -> bool:
        """Run basic smoke tests"""
        port = 8001 if target == 'blue' else 8002
        base_url = f"http://localhost:{port}"

        tests = [
            ("/health", 200),
            ("/ready", 200),
            ("/metrics", 200),
        ]

        for endpoint, expected_status in tests:
            try:
                response = requests.get(f"{base_url}{endpoint}", timeout=10)
                if response.status_code != expected_status:
                    print(f"  ✗ {endpoint} returned {response.status_code}")
                    return False
                print(f"  ✓ {endpoint} passed")
            except Exception as e:
                print(f"  ✗ {endpoint} failed: {e}")
                return False

        return True

    def validate_model_performance(self, target: str) -> bool:
        """Validate model prediction performance"""
        port = 8001 if target == 'blue' else 8002
        url = f"http://localhost:{port}/predict"

        # Sample test data
        test_data = {
            "features": [0.5, 1.2, -0.3, 2.1]
        }

        try:
            response = requests.post(url, json=test_data, timeout=30)
            if response.status_code != 200:
                print(f"  ✗ Prediction failed with status {response.status_code}")
                return False

            result = response.json()
            latency = result.get('latency_ms', 0)

            # Validate latency threshold
            if latency > 1000:  # 1 second max
                print(f"  ✗ Latency too high: {latency}ms")
                return False

            print(f"  ✓ Model prediction successful (latency: {latency}ms)")
            return True

        except Exception as e:
            print(f"  ✗ Prediction test failed: {e}")
            return False

    def switch_traffic(self, target: str):
        """Update load balancer to route traffic to target"""
        # Update nginx config
        config = f"""
upstream ml_api {{
    server ml-api-{target}:8000 weight=100;
    server ml-api-{'green' if target == 'blue' else 'blue'}:8000 weight=0;
}}
        """

        # Reload nginx
        subprocess.run("docker exec nginx nginx -s reload", shell=True, check=True)
        print(f"  ✓ Traffic switched to {target}")

    def monitor_deployment(self, target: str, duration: int = 60) -> bool:
        """Monitor deployment for errors"""
        port = 8001 if target == 'blue' else 8002
        url = f"http://localhost:{port}/metrics"

        start_time = time.time()
        baseline_errors = None

        while time.time() - start_time < duration:
            try:
                response = requests.get(url, timeout=5)
                metrics = response.text

                # Parse error count
                error_count = self.parse_error_count(metrics)

                if baseline_errors is None:
                    baseline_errors = error_count
                elif error_count > baseline_errors + 10:
                    print(f"  ✗ Error rate increased: {error_count} errors")
                    return False

                time.sleep(10)
            except Exception as e:
                print(f"  ✗ Monitoring failed: {e}")
                return False

        print(f"  ✓ No issues detected during monitoring period")
        return True

    def parse_error_count(self, metrics: str) -> int:
        """Parse error count from Prometheus metrics"""
        for line in metrics.split('\n'):
            if 'prediction_errors_total' in line and not line.startswith('#'):
                return int(float(line.split()[-1]))
        return 0

if __name__ == "__main__":
    deployer = BlueGreenDeployer()
    deployer.deploy('green', 'v2.0.0')
```

**Step 5: Execute Blue-Green Deployment**

```bash
# Deploy to green (inactive) environment
python deploy_blue_green.py

# If successful, green is now active
# To deploy next version, target blue
python deploy_blue_green.py blue v3.0.0
```

### 1.2 Canary Deployment for Gradual Rollout

Canary deployment reduces risk by gradually shifting traffic to new model versions.

**Step 1: Canary Configuration**

```yaml
# docker-compose.canary.yml
version: '3.8'

services:
  ml-api-production:
    image: ml-api:v1.0.0
    deploy:
      replicas: 9  # 90% of traffic
      resources:
        limits:
          cpus: '1.0'
          memory: 2G
    labels:
      - "version=v1.0.0"
      - "deployment=production"

  ml-api-canary:
    image: ml-api:v2.0.0
    deploy:
      replicas: 1  # 10% of traffic
      resources:
        limits:
          cpus: '1.0'
          memory: 2G
    labels:
      - "version=v2.0.0"
      - "deployment=canary"

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx-canary.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - ml-api-production
      - ml-api-canary
```

**Step 2: Canary Deployment Script**

```python
# deploy_canary.py
import subprocess
import time
import requests
from typing import List, Dict

class CanaryDeployer:
    def __init__(self):
        self.stages = [
            {"replicas_prod": 9, "replicas_canary": 1, "wait": 300},   # 10%
            {"replicas_prod": 7, "replicas_canary": 3, "wait": 300},   # 30%
            {"replicas_prod": 5, "replicas_canary": 5, "wait": 600},   # 50%
            {"replicas_prod": 3, "replicas_canary": 7, "wait": 600},   # 70%
            {"replicas_prod": 0, "replicas_canary": 10, "wait": 300},  # 100%
        ]

    def deploy(self, canary_version: str):
        """Execute canary deployment"""
        print(f"Starting canary deployment for version {canary_version}")

        for i, stage in enumerate(self.stages, 1):
            percentage = int((stage['replicas_canary'] / 10) * 100)
            print(f"\n[Stage {i}/5] Shifting {percentage}% traffic to canary")

            # Scale services
            self.scale_services(stage['replicas_prod'], stage['replicas_canary'])

            # Monitor canary metrics
            print(f"Monitoring canary for {stage['wait']} seconds...")
            if not self.monitor_canary(stage['wait']):
                print("ERROR: Canary showing degraded performance, rolling back")
                self.rollback()
                return False

            print(f"✓ Stage {i} successful")

        print("\n✓ Canary deployment completed successfully")
        return True

    def scale_services(self, prod_replicas: int, canary_replicas: int):
        """Scale production and canary services"""
        cmds = [
            f"docker service scale ml-api-production={prod_replicas}",
            f"docker service scale ml-api-canary={canary_replicas}"
        ]

        for cmd in cmds:
            subprocess.run(cmd, shell=True, check=True)

        # Wait for scaling to complete
        time.sleep(30)

    def monitor_canary(self, duration: int) -> bool:
        """Monitor canary metrics and compare to production"""
        metrics_url = "http://localhost/metrics"

        start_time = time.time()
        while time.time() - start_time < duration:
            try:
                # Get metrics
                canary_metrics = self.get_service_metrics("canary")
                prod_metrics = self.get_service_metrics("production")

                # Compare error rates
                if canary_metrics['error_rate'] > prod_metrics['error_rate'] * 1.5:
                    print(f"✗ Canary error rate too high: {canary_metrics['error_rate']}")
                    return False

                # Compare latency
                if canary_metrics['p95_latency'] > prod_metrics['p95_latency'] * 1.2:
                    print(f"✗ Canary latency too high: {canary_metrics['p95_latency']}ms")
                    return False

                time.sleep(30)
            except Exception as e:
                print(f"Warning: Monitoring error: {e}")

        return True

    def get_service_metrics(self, deployment: str) -> Dict:
        """Get metrics for specific deployment"""
        # Implementation would query Prometheus
        return {
            'error_rate': 0.01,
            'p95_latency': 150
        }

    def rollback(self):
        """Rollback canary deployment"""
        print("Rolling back to production version...")
        self.scale_services(10, 0)

if __name__ == "__main__":
    deployer = CanaryDeployer()
    deployer.deploy("v2.0.0")
```

### 1.3 A/B Testing for Model Comparison

A/B testing allows comparing multiple model versions simultaneously.

**A/B Testing Implementation**

```python
# ab_testing.py
from flask import Flask, request, jsonify
import random
import hashlib
from typing import Literal

app = Flask(__name__)

class ABTestRouter:
    def __init__(self):
        self.models = {
            'A': {'endpoint': 'http://ml-model-a:8000', 'traffic': 50},
            'B': {'endpoint': 'http://ml-model-b:8000', 'traffic': 50}
        }

    def route_request(self, user_id: str) -> Literal['A', 'B']:
        """Consistently route user to same model variant"""
        # Hash user_id for consistent routing
        hash_value = int(hashlib.md5(user_id.encode()).hexdigest(), 16)

        # Route based on hash
        if hash_value % 100 < self.models['A']['traffic']:
            return 'A'
        return 'B'

    def update_traffic_split(self, model_a_percentage: int):
        """Dynamically adjust traffic split"""
        self.models['A']['traffic'] = model_a_percentage
        self.models['B']['traffic'] = 100 - model_a_percentage

router = ABTestRouter()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    user_id = request.headers.get('X-User-ID', str(random.random()))

    # Route to appropriate model
    variant = router.route_request(user_id)
    model_endpoint = router.models[variant]['endpoint']

    # Forward request to model
    import requests
    response = requests.post(f"{model_endpoint}/predict", json=data)

    # Add variant information to response
    result = response.json()
    result['variant'] = variant

    return jsonify(result)

@app.route('/ab/config', methods=['POST'])
def update_ab_config():
    """Update A/B test configuration"""
    data = request.json
    model_a_percentage = data.get('model_a_percentage', 50)

    router.update_traffic_split(model_a_percentage)

    return jsonify({
        'status': 'updated',
        'config': router.models
    })
```

---

## 2. Health Checks and Readiness Probes

### 2.1 Comprehensive Health Check Implementation

```python
# app/health.py
from fastapi import FastAPI, status, Response
from fastapi.responses import JSONResponse
import psutil
import time
import torch
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)
app = FastAPI()

class HealthChecker:
    def __init__(self):
        self.startup_time = time.time()
        self.model_loaded = False
        self.last_prediction_time = None

    def set_model_loaded(self, loaded: bool):
        self.model_loaded = loaded

    def update_prediction_time(self):
        self.last_prediction_time = time.time()

    def check_model(self) -> Dict[str, Any]:
        """Check if model is loaded and functioning"""
        return {
            'loaded': self.model_loaded,
            'time_since_last_prediction':
                time.time() - self.last_prediction_time
                if self.last_prediction_time else None
        }

    def check_resources(self) -> Dict[str, Any]:
        """Check system resources"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        return {
            'cpu': {
                'percent': cpu_percent,
                'healthy': cpu_percent < 90
            },
            'memory': {
                'percent': memory.percent,
                'available_mb': memory.available / 1024 / 1024,
                'healthy': memory.percent < 90
            },
            'disk': {
                'percent': disk.percent,
                'free_gb': disk.free / 1024 / 1024 / 1024,
                'healthy': disk.percent < 90
            }
        }

    def check_gpu(self) -> Dict[str, Any]:
        """Check GPU status if available"""
        if not torch.cuda.is_available():
            return {'available': False}

        try:
            gpu_memory_allocated = torch.cuda.memory_allocated() / 1024 / 1024
            gpu_memory_reserved = torch.cuda.memory_reserved() / 1024 / 1024

            return {
                'available': True,
                'device_count': torch.cuda.device_count(),
                'memory_allocated_mb': gpu_memory_allocated,
                'memory_reserved_mb': gpu_memory_reserved,
                'healthy': gpu_memory_reserved < 8000  # 8GB threshold
            }
        except Exception as e:
            logger.error(f"GPU check failed: {e}")
            return {'available': True, 'healthy': False, 'error': str(e)}

health_checker = HealthChecker()

@app.get("/health")
async def health_check():
    """Basic liveness check - is the service running?"""
    return {
        'status': 'healthy',
        'timestamp': time.time(),
        'uptime_seconds': time.time() - health_checker.startup_time
    }

@app.get("/health/ready")
async def readiness_check():
    """Detailed readiness check - can service handle traffic?"""

    # Perform all checks
    model_status = health_checker.check_model()
    resource_status = health_checker.check_resources()
    gpu_status = health_checker.check_gpu()

    # Determine overall readiness
    checks_passed = (
        model_status['loaded'] and
        resource_status['cpu']['healthy'] and
        resource_status['memory']['healthy'] and
        resource_status['disk']['healthy'] and
        (not gpu_status['available'] or gpu_status['healthy'])
    )

    response_data = {
        'ready': checks_passed,
        'checks': {
            'model': model_status,
            'resources': resource_status,
            'gpu': gpu_status
        },
        'timestamp': time.time()
    }

    if checks_passed:
        return response_data
    else:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=response_data
        )

@app.get("/health/live")
async def liveness_check():
    """Liveness check - should container be restarted?"""

    # Check for deadlock or hung state
    uptime = time.time() - health_checker.startup_time

    # If model not loaded after 5 minutes, something is wrong
    if uptime > 300 and not health_checker.model_loaded:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                'alive': False,
                'reason': 'Model failed to load',
                'uptime_seconds': uptime
            }
        )

    return {
        'alive': True,
        'uptime_seconds': uptime,
        'model_loaded': health_checker.model_loaded
    }

@app.get("/health/startup")
async def startup_check():
    """Startup probe - has initialization completed?"""

    startup_complete = (
        health_checker.model_loaded and
        time.time() - health_checker.startup_time > 10
    )

    if startup_complete:
        return {'started': True}
    else:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                'started': False,
                'model_loaded': health_checker.model_loaded,
                'uptime': time.time() - health_checker.startup_time
            }
        )
```

### 2.2 Docker and Kubernetes Health Check Configuration

**Docker Compose**

```yaml
# docker-compose.yml
version: '3.8'

services:
  ml-api:
    build: .
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health/ready"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s  # Allow 60s for model loading
    deploy:
      restart_policy:
        condition: on-failure
        delay: 10s
        max_attempts: 3
        window: 120s
```

**Kubernetes Deployment**

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-inference
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: ml-api
        image: ml-api:v1.0.0
        ports:
        - containerPort: 8000

        # Startup probe - allows long initialization
        startupProbe:
          httpGet:
            path: /health/startup
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 10
          failureThreshold: 30  # 5 minutes total

        # Liveness probe - restart if fails
        livenessProbe:
          httpGet:
            path: /health/live
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3

        # Readiness probe - remove from load balancer if fails
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 5
          timeoutSeconds: 3
          successThreshold: 1
          failureThreshold: 3
```

---

## 3. Logging and Monitoring Integration

### 3.1 Structured Logging Implementation

```python
# app/logging_config.py
import logging
import json
import sys
from datetime import datetime
from typing import Any, Dict
import traceback

class JSONFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging"""

    def format(self, record: logging.LogRecord) -> str:
        log_data: Dict[str, Any] = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }

        # Add exception information
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }

        # Add custom fields from extra
        if hasattr(record, 'request_id'):
            log_data['request_id'] = record.request_id
        if hasattr(record, 'user_id'):
            log_data['user_id'] = record.user_id
        if hasattr(record, 'model_version'):
            log_data['model_version'] = record.model_version
        if hasattr(record, 'latency_ms'):
            log_data['latency_ms'] = record.latency_ms
        if hasattr(record, 'prediction'):
            log_data['prediction'] = record.prediction

        return json.dumps(log_data)

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Configure structured logging"""

    logger = logging.getLogger("ml_api")
    logger.setLevel(getattr(logging, log_level.upper()))

    # Console handler with JSON formatting
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)

    return logger

# Usage in main application
logger = setup_logging()

# Example log statements
logger.info("Model loaded successfully", extra={
    'model_version': 'v2.1.0',
    'model_size_mb': 145
})

logger.info("Prediction completed", extra={
    'request_id': 'req-12345',
    'user_id': 'user-789',
    'model_version': 'v2.1.0',
    'latency_ms': 45,
    'prediction': 0.87
})
```

### 3.2 Centralized Logging with EFK Stack

```yaml
# docker-compose.logging.yml
version: '3.8'

services:
  # Elasticsearch for log storage
  elasticsearch:
    image: elasticsearch:8.11.0
    environment:
      - discovery.type=single-node
      - "ES_JAVA_OPTS=-Xms1g -Xmx1g"
      - xpack.security.enabled=false
    ports:
      - "9200:9200"
    volumes:
      - elasticsearch-data:/usr/share/elasticsearch/data
    healthcheck:
      test: ["CMD-SHELL", "curl -f http://localhost:9200/_cluster/health || exit 1"]
      interval: 30s
      timeout: 10s
      retries: 5

  # Fluentd for log aggregation
  fluentd:
    build:
      context: ./fluentd
      dockerfile: Dockerfile
    ports:
      - "24224:24224"
      - "24224:24224/udp"
    volumes:
      - ./fluentd/conf:/fluentd/etc
      - fluentd-logs:/fluentd/log
    depends_on:
      elasticsearch:
        condition: service_healthy
    environment:
      - FLUENTD_CONF=fluent.conf

  # Kibana for visualization
  kibana:
    image: kibana:8.11.0
    ports:
      - "5601:5601"
    environment:
      - ELASTICSEARCH_HOSTS=http://elasticsearch:9200
    depends_on:
      elasticsearch:
        condition: service_healthy

  # ML API with logging
  ml-api:
    build: .
    logging:
      driver: "fluentd"
      options:
        fluentd-address: localhost:24224
        tag: ml-api
        fluentd-async: "true"
    depends_on:
      - fluentd

volumes:
  elasticsearch-data:
  fluentd-logs:
```

**Fluentd Configuration**

```ruby
# fluentd/conf/fluent.conf
<source>
  @type forward
  port 24224
  bind 0.0.0.0
</source>

# Parse JSON logs
<filter ml-api>
  @type parser
  key_name log
  <parse>
    @type json
  </parse>
</filter>

# Add metadata
<filter ml-api>
  @type record_transformer
  <record>
    hostname "#{Socket.gethostname}"
    tag ${tag}
  </record>
</filter>

# Send to Elasticsearch
<match ml-api>
  @type elasticsearch
  host elasticsearch
  port 9200
  logstash_format true
  logstash_prefix ml-api
  include_timestamp true
  <buffer>
    @type file
    path /fluentd/log/buffer
    flush_interval 10s
    retry_max_interval 30s
    chunk_limit_size 2M
  </buffer>
</match>
```

### 3.3 Prometheus Monitoring

```python
# app/metrics.py
from prometheus_client import Counter, Histogram, Gauge, Summary, Info
from prometheus_client import generate_latest, REGISTRY
from fastapi import FastAPI, Response
import time
from functools import wraps

# Define metrics
PREDICTION_COUNT = Counter(
    'ml_predictions_total',
    'Total number of predictions',
    ['model_version', 'status', 'model_type']
)

PREDICTION_LATENCY = Histogram(
    'ml_prediction_latency_seconds',
    'Prediction latency in seconds',
    ['model_version'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

PREDICTION_SCORE = Summary(
    'ml_prediction_score',
    'Prediction confidence scores',
    ['model_version']
)

MODEL_LOADED = Gauge(
    'ml_model_loaded',
    'Is model loaded (1=yes, 0=no)',
    ['model_version']
)

ACTIVE_REQUESTS = Gauge(
    'ml_active_requests',
    'Number of requests currently being processed'
)

ERROR_COUNT = Counter(
    'ml_errors_total',
    'Total number of errors',
    ['error_type', 'model_version']
)

MODEL_INFO = Info(
    'ml_model_info',
    'Information about the loaded model'
)

# GPU Metrics
GPU_MEMORY_ALLOCATED = Gauge(
    'ml_gpu_memory_allocated_bytes',
    'GPU memory currently allocated',
    ['device_id']
)

GPU_UTILIZATION = Gauge(
    'ml_gpu_utilization_percent',
    'GPU utilization percentage',
    ['device_id']
)

# Model drift metrics
PREDICTION_DISTRIBUTION = Histogram(
    'ml_prediction_distribution',
    'Distribution of prediction values',
    ['model_version'],
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

FEATURE_STATS = Summary(
    'ml_feature_statistics',
    'Statistics of input features',
    ['feature_name', 'model_version']
)

def track_prediction_metrics(model_version: str):
    """Decorator to track prediction metrics"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            ACTIVE_REQUESTS.inc()
            start_time = time.time()

            try:
                result = await func(*args, **kwargs)

                # Track success
                PREDICTION_COUNT.labels(
                    model_version=model_version,
                    status='success',
                    model_type='classifier'
                ).inc()

                # Track latency
                latency = time.time() - start_time
                PREDICTION_LATENCY.labels(model_version=model_version).observe(latency)

                # Track prediction score
                if 'prediction' in result:
                    PREDICTION_SCORE.labels(model_version=model_version).observe(
                        result['prediction']
                    )
                    PREDICTION_DISTRIBUTION.labels(model_version=model_version).observe(
                        result['prediction']
                    )

                return result

            except Exception as e:
                # Track errors
                ERROR_COUNT.labels(
                    error_type=type(e).__name__,
                    model_version=model_version
                ).inc()

                PREDICTION_COUNT.labels(
                    model_version=model_version,
                    status='error',
                    model_type='classifier'
                ).inc()

                raise
            finally:
                ACTIVE_REQUESTS.dec()

        return wrapper
    return decorator

app = FastAPI()

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(
        content=generate_latest(REGISTRY),
        media_type="text/plain"
    )

# Update GPU metrics periodically
import torch

def update_gpu_metrics():
    """Update GPU-related metrics"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            GPU_MEMORY_ALLOCATED.labels(device_id=str(i)).set(
                torch.cuda.memory_allocated(i)
            )

            # GPU utilization would require nvidia-ml-py3
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                GPU_UTILIZATION.labels(device_id=str(i)).set(util.gpu)
            except ImportError:
                pass
```

**Prometheus Configuration**

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    cluster: 'ml-production'
    environment: 'production'

# Alerting configuration
alerting:
  alertmanagers:
    - static_configs:
        - targets: ['alertmanager:9093']

# Alert rules
rule_files:
  - 'alerts.yml'

# Scrape configurations
scrape_configs:
  - job_name: 'ml-api'
    static_configs:
      - targets: ['ml-api:8000']
    metrics_path: '/metrics'
    scrape_interval: 10s

  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']
```

**Alert Rules**

```yaml
# alerts.yml
groups:
  - name: ml_api_alerts
    interval: 30s
    rules:
      # High error rate
      - alert: HighErrorRate
        expr: |
          rate(ml_predictions_total{status="error"}[5m]) /
          rate(ml_predictions_total[5m]) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High prediction error rate"
          description: "Error rate is {{ $value | humanizePercentage }} over last 5 minutes"

      # High latency
      - alert: HighPredictionLatency
        expr: |
          histogram_quantile(0.95,
            rate(ml_prediction_latency_seconds_bucket[5m])
          ) > 1.0
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High prediction latency"
          description: "P95 latency is {{ $value }}s"

      # Model not loaded
      - alert: ModelNotLoaded
        expr: ml_model_loaded == 0
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "ML model not loaded"
          description: "Model {{ $labels.model_version }} is not loaded"

      # GPU memory high
      - alert: GPUMemoryHigh
        expr: |
          ml_gpu_memory_allocated_bytes / (1024*1024*1024) > 7
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "GPU memory usage high"
          description: "GPU {{ $labels.device_id }} using {{ $value }}GB"

      # Potential model drift
      - alert: PredictionDistributionDrift
        expr: |
          abs(
            avg_over_time(ml_prediction_score_sum[1h]) /
            avg_over_time(ml_prediction_score_count[1h]) -
            avg_over_time(ml_prediction_score_sum[1h] offset 24h) /
            avg_over_time(ml_prediction_score_count[1h] offset 24h)
          ) > 0.1
        for: 30m
        labels:
          severity: warning
        annotations:
          summary: "Potential model drift detected"
          description: "Average prediction changed by {{ $value }}"
```

### 3.4 Grafana Dashboards

```json
// grafana/dashboards/ml-api-dashboard.json
{
  "dashboard": {
    "title": "ML API Production Dashboard",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(ml_predictions_total[5m])",
            "legendFormat": "{{model_version}} - {{status}}"
          }
        ]
      },
      {
        "title": "Latency Percentiles",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.50, rate(ml_prediction_latency_seconds_bucket[5m]))",
            "legendFormat": "p50"
          },
          {
            "expr": "histogram_quantile(0.95, rate(ml_prediction_latency_seconds_bucket[5m]))",
            "legendFormat": "p95"
          },
          {
            "expr": "histogram_quantile(0.99, rate(ml_prediction_latency_seconds_bucket[5m]))",
            "legendFormat": "p99"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(ml_predictions_total{status='error'}[5m]) / rate(ml_predictions_total[5m])",
            "legendFormat": "Error Rate"
          }
        ]
      },
      {
        "title": "GPU Memory Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "ml_gpu_memory_allocated_bytes / (1024*1024*1024)",
            "legendFormat": "GPU {{device_id}}"
          }
        ]
      },
      {
        "title": "Prediction Distribution",
        "type": "heatmap",
        "targets": [
          {
            "expr": "rate(ml_prediction_distribution_bucket[5m])",
            "legendFormat": "{{le}}"
          }
        ]
      },
      {
        "title": "Active Requests",
        "type": "graph",
        "targets": [
          {
            "expr": "ml_active_requests",
            "legendFormat": "Active Requests"
          }
        ]
      }
    ]
  }
}
```

---

## 4. Auto-Scaling and Load Balancing

### 4.1 Docker Swarm Auto-Scaling

```bash
# deploy_swarm.sh
#!/bin/bash

# Initialize swarm
docker swarm init

# Create overlay network
docker network create --driver overlay ml-network

# Deploy ML service with constraints
docker service create \
  --name ml-api \
  --replicas 3 \
  --network ml-network \
  --publish published=8000,target=8000 \
  --limit-cpu 2.0 \
  --limit-memory 4G \
  --reserve-cpu 1.0 \
  --reserve-memory 2G \
  --update-parallelism 1 \
  --update-delay 30s \
  --update-failure-action rollback \
  --rollback-parallelism 1 \
  --rollback-delay 10s \
  --env MODEL_VERSION=v1.0.0 \
  --mount type=bind,source=/models,target=/models,readonly \
  --health-cmd "curl -f http://localhost:8000/health || exit 1" \
  --health-interval 30s \
  --health-timeout 10s \
  --health-retries 3 \
  --health-start-period 60s \
  ml-api:latest

# Auto-scaling script
cat > autoscale.sh << 'EOF'
#!/bin/bash

SERVICE_NAME="ml-api"
MIN_REPLICAS=2
MAX_REPLICAS=10
CPU_THRESHOLD=70
MEMORY_THRESHOLD=80

while true; do
    # Get current replicas
    CURRENT_REPLICAS=$(docker service ls --filter name=$SERVICE_NAME --format "{{.Replicas}}" | cut -d'/' -f1)

    # Get average CPU usage
    AVG_CPU=$(docker service ps $SERVICE_NAME -q | xargs -I {} docker stats {} --no-stream --format "{{.CPUPerc}}" | sed 's/%//' | awk '{sum+=$1; count++} END {print sum/count}')

    echo "Current replicas: $CURRENT_REPLICAS, Average CPU: $AVG_CPU%"

    # Scale up if CPU high
    if (( $(echo "$AVG_CPU > $CPU_THRESHOLD" | bc -l) )); then
        if [ "$CURRENT_REPLICAS" -lt "$MAX_REPLICAS" ]; then
            NEW_REPLICAS=$((CURRENT_REPLICAS + 1))
            echo "Scaling up to $NEW_REPLICAS replicas"
            docker service scale $SERVICE_NAME=$NEW_REPLICAS
        fi
    fi

    # Scale down if CPU low
    if (( $(echo "$AVG_CPU < 30" | bc -l) )); then
        if [ "$CURRENT_REPLICAS" -gt "$MIN_REPLICAS" ]; then
            NEW_REPLICAS=$((CURRENT_REPLICAS - 1))
            echo "Scaling down to $NEW_REPLICAS replicas"
            docker service scale $SERVICE_NAME=$NEW_REPLICAS
        fi
    fi

    sleep 60
done
EOF

chmod +x autoscale.sh
```

### 4.2 Kubernetes Horizontal Pod Autoscaler

```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ml-api-hpa
  namespace: production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ml-inference
  minReplicas: 3
  maxReplicas: 20
  metrics:
    # CPU-based scaling
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70

    # Memory-based scaling
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80

    # Custom metric: requests per second
    - type: Pods
      pods:
        metric:
          name: http_requests_per_second
        target:
          type: AverageValue
          averageValue: "1000"

    # Custom metric: prediction latency
    - type: Pods
      pods:
        metric:
          name: prediction_latency_p95
        target:
          type: AverageValue
          averageValue: "500m"  # 500ms

  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300  # Wait 5 minutes before scaling down
      policies:
        - type: Percent
          value: 25  # Scale down max 25% at a time
          periodSeconds: 60
        - type: Pods
          value: 2  # Scale down max 2 pods at a time
          periodSeconds: 60
      selectPolicy: Min  # Choose the policy that scales down least

    scaleUp:
      stabilizationWindowSeconds: 60  # Wait 1 minute before scaling up
      policies:
        - type: Percent
          value: 50  # Scale up max 50% at a time
          periodSeconds: 30
        - type: Pods
          value: 4  # Scale up max 4 pods at a time
          periodSeconds: 30
      selectPolicy: Max  # Choose the policy that scales up most
```

### 4.3 NGINX Load Balancer Configuration

```nginx
# nginx-advanced.conf
upstream ml_api_backend {
    # Load balancing method
    least_conn;  # Route to server with fewest active connections

    # Backend servers
    server ml-api-1:8000 max_fails=3 fail_timeout=30s weight=1;
    server ml-api-2:8000 max_fails=3 fail_timeout=30s weight=1;
    server ml-api-3:8000 max_fails=3 fail_timeout=30s weight=1;

    # Health check (requires nginx-plus or custom module)
    # health_check interval=10s fails=3 passes=2;

    # Keepalive connections
    keepalive 32;
}

# Rate limiting
limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;
limit_conn_zone $binary_remote_addr zone=conn_limit:10m;

# Connection caching
proxy_cache_path /var/cache/nginx levels=1:2 keys_zone=ml_cache:10m max_size=1g inactive=60m use_temp_path=off;

server {
    listen 80;
    server_name ml-api.production.com;

    # Redirect HTTP to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name ml-api.production.com;

    # SSL configuration
    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;

    # Security headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;

    # Prediction endpoint
    location /predict {
        # Rate limiting
        limit_req zone=api_limit burst=20 nodelay;
        limit_conn conn_limit 10;

        # Proxy settings
        proxy_pass http://ml_api_backend;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Timeouts for ML inference
        proxy_connect_timeout 60s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;

        # Buffer settings
        proxy_buffering on;
        proxy_buffer_size 4k;
        proxy_buffers 8 4k;
        proxy_busy_buffers_size 8k;

        # Retry logic
        proxy_next_upstream error timeout invalid_header http_500 http_502 http_503;
        proxy_next_upstream_tries 2;
        proxy_next_upstream_timeout 30s;
    }

    # Health check endpoint (not cached, no rate limit)
    location /health {
        access_log off;
        proxy_pass http://ml_api_backend;
        proxy_connect_timeout 5s;
        proxy_read_timeout 5s;
    }

    # Metrics endpoint (restricted access)
    location /metrics {
        allow 10.0.0.0/8;  # Internal network only
        deny all;

        proxy_pass http://ml_api_backend;
    }

    # Static content caching
    location /static/ {
        proxy_cache ml_cache;
        proxy_cache_valid 200 60m;
        proxy_cache_use_stale error timeout updating http_500 http_502 http_503 http_504;
        proxy_cache_lock on;

        proxy_pass http://ml_api_backend;
    }
}
```

---

## 5. Rollback Procedures

### 5.1 Automated Rollback Script

```python
# rollback.py
import subprocess
import requests
import time
import argparse
from typing import Optional

class RollbackManager:
    def __init__(self, compose_file: str = "docker-compose.yml"):
        self.compose_file = compose_file
        self.health_url = "http://localhost:8000/health/ready"

    def get_current_version(self) -> str:
        """Get currently deployed version"""
        try:
            result = subprocess.run(
                f"docker-compose -f {self.compose_file} ps -q ml-api",
                shell=True,
                capture_output=True,
                text=True
            )
            container_id = result.stdout.strip()

            if container_id:
                result = subprocess.run(
                    f"docker inspect {container_id} --format='{{{{.Config.Image}}}}'",
                    shell=True,
                    capture_output=True,
                    text=True
                )
                return result.stdout.strip()

            return "unknown"
        except Exception as e:
            print(f"Error getting current version: {e}")
            return "unknown"

    def rollback_to_previous_version(self, previous_version: str):
        """Rollback to previous version"""
        print(f"[1/5] Rolling back to version: {previous_version}")

        # Update compose file to use previous version
        subprocess.run(
            f"sed -i 's/image: ml-api:.*/image: {previous_version}/' {self.compose_file}",
            shell=True,
            check=True
        )

        # Pull previous image
        print("[2/5] Pulling previous image...")
        subprocess.run(
            f"docker pull {previous_version}",
            shell=True,
            check=True
        )

        # Stop current containers
        print("[3/5] Stopping current containers...")
        subprocess.run(
            f"docker-compose -f {self.compose_file} stop ml-api",
            shell=True,
            check=True
        )

        # Start containers with previous version
        print("[4/5] Starting containers with previous version...")
        subprocess.run(
            f"docker-compose -f {self.compose_file} up -d ml-api",
            shell=True,
            check=True
        )

        # Verify health
        print("[5/5] Verifying health...")
        if not self.wait_for_health(timeout=120):
            print("ERROR: Health check failed after rollback")
            return False

        print("SUCCESS: Rollback completed successfully")
        return True

    def wait_for_health(self, timeout: int = 120) -> bool:
        """Wait for service to be healthy"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(self.health_url, timeout=5)
                if response.status_code == 200:
                    return True
            except requests.RequestException:
                pass
            time.sleep(5)
        return False

    def create_backup(self):
        """Create backup of current deployment"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        backup_dir = f"/backups/deployment_{timestamp}"

        print(f"Creating backup at {backup_dir}")

        # Backup docker-compose file
        subprocess.run(
            f"mkdir -p {backup_dir} && cp {self.compose_file} {backup_dir}/",
            shell=True,
            check=True
        )

        # Backup environment files
        subprocess.run(
            f"cp .env* {backup_dir}/ 2>/dev/null || true",
            shell=True
        )

        # Export container configs
        result = subprocess.run(
            f"docker-compose -f {self.compose_file} config",
            shell=True,
            capture_output=True,
            text=True
        )

        with open(f"{backup_dir}/resolved-compose.yml", 'w') as f:
            f.write(result.stdout)

        print(f"Backup created at {backup_dir}")
        return backup_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Rollback ML API deployment')
    parser.add_argument('--version', required=True, help='Version to rollback to')
    parser.add_argument('--compose-file', default='docker-compose.yml', help='Docker compose file')
    parser.add_argument('--backup', action='store_true', help='Create backup before rollback')

    args = parser.parse_args()

    manager = RollbackManager(args.compose_file)

    if args.backup:
        manager.create_backup()

    manager.rollback_to_previous_version(args.version)
```

### 5.2 Kubernetes Rollback

```bash
# k8s-rollback.sh
#!/bin/bash

NAMESPACE="production"
DEPLOYMENT="ml-inference"

# View rollout history
echo "Rollout history:"
kubectl rollout history deployment/$DEPLOYMENT -n $NAMESPACE

# Check current status
echo -e "\nCurrent status:"
kubectl rollout status deployment/$DEPLOYMENT -n $NAMESPACE

# Rollback to previous version
echo -e "\nRolling back to previous version..."
kubectl rollout undo deployment/$DEPLOYMENT -n $NAMESPACE

# Wait for rollback to complete
kubectl rollout status deployment/$DEPLOYMENT -n $NAMESPACE

# Verify pods are running
echo -e "\nVerifying pods:"
kubectl get pods -n $NAMESPACE -l app=ml-inference

# Check health
echo -e "\nChecking health:"
POD=$(kubectl get pods -n $NAMESPACE -l app=ml-inference -o jsonpath='{.items[0].metadata.name}')
kubectl exec $POD -n $NAMESPACE -- curl -f http://localhost:8000/health

echo -e "\n✓ Rollback complete"
```

---

## 6. CI/CD Integration

### 6.1 Complete GitHub Actions Pipeline

```yaml
# .github/workflows/ml-deployment.yml
name: ML API CI/CD Pipeline

on:
  push:
    branches: [main, develop]
    tags: ['v*']
  pull_request:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  # Job 1: Run tests
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install -r requirements-dev.txt

      - name: Run unit tests
        run: |
          pytest tests/unit --cov=app --cov-report=xml

      - name: Run integration tests
        run: |
          docker-compose -f docker-compose.test.yml up -d
          sleep 30
          pytest tests/integration
          docker-compose -f docker-compose.test.yml down

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml

  # Job 2: Build and scan image
  build:
    needs: test
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
      security-events: write

    steps:
      - name: Checkout code
        uses: actions/checkout@v3

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v2

      - name: Log in to Container Registry
        uses: docker/login-action@v2
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v4
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
          tags: |
            type=ref,event=branch
            type=ref,event=pr
            type=semver,pattern={{version}}
            type=semver,pattern={{major}}.{{minor}}
            type=sha,prefix={{branch}}-

      - name: Build and push Docker image
        uses: docker/build-push-action@v4
        with:
          context: .
          file: ./Dockerfile.production
          push: ${{ github.event_name != 'pull_request' }}
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
          build-args: |
            MODEL_VERSION=${{ github.sha }}

      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}
          format: 'sarif'
          output: 'trivy-results.sarif'
          severity: 'CRITICAL,HIGH'

      - name: Upload Trivy results to GitHub Security
        uses: github/codeql-action/upload-sarif@v2
        if: always()
        with:
          sarif_file: 'trivy-results.sarif'

      - name: Fail on critical vulnerabilities
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}
          format: 'table'
          exit-code: '1'
          severity: 'CRITICAL'

  # Job 3: Deploy to staging
  deploy-staging:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/develop'
    environment:
      name: staging
      url: https://ml-api-staging.example.com

    steps:
      - name: Checkout code
        uses: actions/checkout@v3

      - name: Configure kubectl
        uses: azure/k8s-set-context@v3
        with:
          kubeconfig: ${{ secrets.KUBE_CONFIG_STAGING }}

      - name: Deploy to staging
        run: |
          kubectl set image deployment/ml-inference \
            ml-api=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }} \
            -n staging

          kubectl rollout status deployment/ml-inference -n staging --timeout=5m

      - name: Run smoke tests
        run: |
          python tests/smoke_test.py --environment staging

  # Job 4: Deploy to production
  deploy-production:
    needs: deploy-staging
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    environment:
      name: production
      url: https://ml-api.example.com

    steps:
      - name: Checkout code
        uses: actions/checkout@v3

      - name: Configure kubectl
        uses: azure/k8s-set-context@v3
        with:
          kubeconfig: ${{ secrets.KUBE_CONFIG_PRODUCTION }}

      - name: Create backup
        run: |
          kubectl get deployment ml-inference -n production -o yaml > backup-deployment.yaml

      - name: Deploy to production (canary)
        run: |
          # Deploy canary
          kubectl apply -f k8s/canary-deployment.yaml

          # Wait and monitor
          sleep 300

          # Check canary metrics
          python scripts/check_canary_metrics.py

          # If successful, promote canary
          kubectl set image deployment/ml-inference \
            ml-api=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }} \
            -n production

          kubectl rollout status deployment/ml-inference -n production --timeout=10m

      - name: Verify deployment
        run: |
          python tests/smoke_test.py --environment production
          python tests/integration_test.py --environment production

      - name: Rollback on failure
        if: failure()
        run: |
          kubectl apply -f backup-deployment.yaml
          kubectl rollout status deployment/ml-inference -n production

      - name: Notify deployment
        if: always()
        uses: 8398a7/action-slack@v3
        with:
          status: ${{ job.status }}
          text: |
            Production deployment ${{ job.status }}
            Version: ${{ github.sha }}
            Author: ${{ github.actor }}
          webhook_url: ${{ secrets.SLACK_WEBHOOK }}
```

---

## 7. Complete ML Production Pipeline

### 7.1 End-to-End Production Stack

```yaml
# docker-compose.production.yml
version: '3.8'

services:
  # NGINX Load Balancer
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/production.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
      - nginx-cache:/var/cache/nginx
    depends_on:
      - ml-api
    restart: always
    networks:
      - frontend
      - backend

  # ML API (multiple replicas)
  ml-api:
    image: ml-api:${VERSION:-latest}
    deploy:
      replicas: 5
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G
      update_config:
        parallelism: 1
        delay: 30s
        failure_action: rollback
        monitor: 60s
      restart_policy:
        condition: on-failure
        delay: 10s
        max_attempts: 3
        window: 120s
    environment:
      - MODEL_VERSION=${MODEL_VERSION}
      - LOG_LEVEL=info
      - REDIS_URL=redis://redis:6379
      - POSTGRES_URL=postgresql://ml:${DB_PASSWORD}@postgres:5432/mldb
      - MLFLOW_TRACKING_URI=http://mlflow:5000
    volumes:
      - model-cache:/models
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health/ready"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
    networks:
      - backend
    logging:
      driver: "fluentd"
      options:
        fluentd-address: fluentd:24224
        tag: ml-api

  # Redis Cache
  redis:
    image: redis:7-alpine
    command: redis-server --maxmemory 2gb --maxmemory-policy allkeys-lru --appendonly yes
    volumes:
      - redis-data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 3s
      retries: 3
    networks:
      - backend
    restart: always

  # PostgreSQL Database
  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_USER=ml
      - POSTGRES_PASSWORD=${DB_PASSWORD}
      - POSTGRES_DB=mldb
      - POSTGRES_INITDB_ARGS=--data-checksums
    volumes:
      - postgres-data:/var/lib/postgresql/data
      - ./init-db.sql:/docker-entrypoint-initdb.d/init.sql:ro
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ml"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - backend
    restart: always

  # MLflow Tracking Server
  mlflow:
    image: ghcr.io/mlflow/mlflow:latest
    ports:
      - "5000:5000"
    command: >
      mlflow server
      --backend-store-uri postgresql://ml:${DB_PASSWORD}@postgres:5432/mldb
      --default-artifact-root s3://mlflow-artifacts/
      --host 0.0.0.0
    environment:
      - AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
      - AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
    depends_on:
      postgres:
        condition: service_healthy
    networks:
      - backend
    restart: always

  # Prometheus
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - ./prometheus/alerts.yml:/etc/prometheus/alerts.yml:ro
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--storage.tsdb.retention.time=30d'
      - '--web.enable-lifecycle'
    networks:
      - backend
    restart: always

  # Grafana
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
      - GF_INSTALL_PLUGINS=grafana-piechart-panel
    volumes:
      - grafana-data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards:ro
      - ./grafana/datasources:/etc/grafana/provisioning/datasources:ro
    depends_on:
      - prometheus
    networks:
      - backend
    restart: always

  # Elasticsearch
  elasticsearch:
    image: elasticsearch:8.11.0
    environment:
      - discovery.type=single-node
      - "ES_JAVA_OPTS=-Xms1g -Xmx1g"
      - xpack.security.enabled=false
    volumes:
      - elasticsearch-data:/usr/share/elasticsearch/data
    healthcheck:
      test: ["CMD-SHELL", "curl -f http://localhost:9200/_cluster/health || exit 1"]
      interval: 30s
      timeout: 10s
      retries: 5
    networks:
      - backend
    restart: always

  # Fluentd
  fluentd:
    build: ./fluentd
    ports:
      - "24224:24224"
      - "24224:24224/udp"
    volumes:
      - ./fluentd/conf:/fluentd/etc:ro
    depends_on:
      elasticsearch:
        condition: service_healthy
    networks:
      - backend
    restart: always

  # Kibana
  kibana:
    image: kibana:8.11.0
    ports:
      - "5601:5601"
    environment:
      - ELASTICSEARCH_HOSTS=http://elasticsearch:9200
    depends_on:
      elasticsearch:
        condition: service_healthy
    networks:
      - backend
    restart: always

networks:
  frontend:
    driver: bridge
  backend:
    driver: bridge

volumes:
  nginx-cache:
  model-cache:
  redis-data:
  postgres-data:
  prometheus-data:
  grafana-data:
  elasticsearch-data:
```

### 7.2 Model Deployment Pipeline

```python
# model_deployment_pipeline.py
import mlflow
import mlflow.pytorch
import requests
import time
from typing import Dict, Any, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLDeploymentPipeline:
    def __init__(
        self,
        mlflow_uri: str,
        staging_api: str,
        production_api: str
    ):
        self.mlflow_uri = mlflow_uri
        self.staging_api = staging_api
        self.production_api = production_api
        mlflow.set_tracking_uri(mlflow_uri)

    def deploy_model(
        self,
        model_name: str,
        version: int,
        deployment_strategy: str = "canary"
    ):
        """Complete model deployment pipeline"""

        logger.info(f"Starting deployment of {model_name} v{version}")

        # Step 1: Validate model
        if not self.validate_model(model_name, version):
            logger.error("Model validation failed")
            return False

        # Step 2: Deploy to staging
        if not self.deploy_to_staging(model_name, version):
            logger.error("Staging deployment failed")
            return False

        # Step 3: Run staging tests
        if not self.run_staging_tests():
            logger.error("Staging tests failed")
            return False

        # Step 4: Deploy to production
        if deployment_strategy == "canary":
            success = self.deploy_canary(model_name, version)
        elif deployment_strategy == "blue-green":
            success = self.deploy_blue_green(model_name, version)
        else:
            success = self.deploy_rolling(model_name, version)

        if not success:
            logger.error("Production deployment failed")
            return False

        # Step 5: Monitor deployment
        if not self.monitor_deployment(duration=3600):  # 1 hour
            logger.error("Deployment monitoring detected issues, rolling back")
            self.rollback()
            return False

        # Step 6: Promote model
        self.promote_model(model_name, version)

        logger.info(f"Deployment of {model_name} v{version} completed successfully")
        return True

    def validate_model(self, model_name: str, version: int) -> bool:
        """Validate model before deployment"""
        logger.info(f"Validating model {model_name} v{version}")

        client = mlflow.tracking.MlflowClient()

        # Get model version
        mv = client.get_model_version(model_name, version)

        # Check required metrics
        run = client.get_run(mv.run_id)
        metrics = run.data.metrics

        required_metrics = {
            'accuracy': 0.90,
            'precision': 0.85,
            'recall': 0.85
        }

        for metric, threshold in required_metrics.items():
            if metric not in metrics:
                logger.error(f"Missing required metric: {metric}")
                return False

            if metrics[metric] < threshold:
                logger.error(
                    f"Metric {metric}={metrics[metric]:.4f} below threshold {threshold}"
                )
                return False

        logger.info("Model validation passed")
        return True

    def deploy_to_staging(self, model_name: str, version: int) -> bool:
        """Deploy model to staging environment"""
        logger.info("Deploying to staging")

        # Transition model to Staging
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Staging"
        )

        # Update staging deployment
        # (Implementation depends on deployment platform)

        return True

    def run_staging_tests(self) -> bool:
        """Run tests in staging environment"""
        logger.info("Running staging tests")

        # Run smoke tests
        response = requests.get(f"{self.staging_api}/health")
        if response.status_code != 200:
            return False

        # Run prediction tests
        test_data = {"features": [1.0, 2.0, 3.0, 4.0]}
        response = requests.post(
            f"{self.staging_api}/predict",
            json=test_data,
            timeout=30
        )

        if response.status_code != 200:
            return False

        # Validate response structure
        result = response.json()
        if 'prediction' not in result:
            return False

        logger.info("Staging tests passed")
        return True

    def deploy_canary(self, model_name: str, version: int) -> bool:
        """Deploy using canary strategy"""
        logger.info("Deploying via canary strategy")

        # Canary deployment implementation
        # (Using previously defined CanaryDeployer)

        return True

    def monitor_deployment(self, duration: int = 3600) -> bool:
        """Monitor deployment for issues"""
        logger.info(f"Monitoring deployment for {duration} seconds")

        start_time = time.time()
        baseline_error_rate = None
        baseline_latency = None

        while time.time() - start_time < duration:
            # Get current metrics
            metrics = self.get_production_metrics()

            if baseline_error_rate is None:
                baseline_error_rate = metrics['error_rate']
                baseline_latency = metrics['p95_latency']
            else:
                # Check for degradation
                if metrics['error_rate'] > baseline_error_rate * 1.5:
                    logger.error(f"Error rate increased: {metrics['error_rate']}")
                    return False

                if metrics['p95_latency'] > baseline_latency * 1.3:
                    logger.error(f"Latency increased: {metrics['p95_latency']}")
                    return False

            time.sleep(60)

        logger.info("Monitoring completed successfully")
        return True

    def get_production_metrics(self) -> Dict[str, Any]:
        """Get production metrics from Prometheus"""
        # Query Prometheus for metrics
        return {
            'error_rate': 0.01,
            'p95_latency': 150,
            'throughput': 1000
        }

    def promote_model(self, model_name: str, version: int):
        """Promote model to production"""
        logger.info(f"Promoting {model_name} v{version} to production")

        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production",
            archive_existing_versions=True
        )

    def rollback(self):
        """Rollback to previous version"""
        logger.info("Rolling back deployment")
        # Rollback implementation
        pass

if __name__ == "__main__":
    pipeline = MLDeploymentPipeline(
        mlflow_uri="http://mlflow:5000",
        staging_api="http://staging-ml-api:8000",
        production_api="http://ml-api:8000"
    )

    pipeline.deploy_model(
        model_name="sentiment-classifier",
        version=5,
        deployment_strategy="canary"
    )
```

---

## Summary

This implementation guide covered:

1. **Deployment Strategies**: Blue-green, canary, and A/B testing for ML models
2. **Health Checks**: Comprehensive liveness, readiness, and startup probes
3. **Logging & Monitoring**: Structured logging with EFK stack, Prometheus metrics, and Grafana dashboards
4. **Auto-Scaling**: Docker Swarm and Kubernetes HPA configurations
5. **Rollback Procedures**: Automated rollback scripts and procedures
6. **CI/CD Integration**: Complete GitHub Actions pipeline for ML deployment
7. **Production Pipeline**: End-to-end ML deployment infrastructure

**Key Takeaways**:
- Always validate models before production deployment
- Implement multiple deployment strategies for different scenarios
- Monitor model performance and drift continuously
- Automate rollback procedures for rapid recovery
- Use structured logging for troubleshooting
- Scale based on actual load and performance metrics

**Next Steps**:
1. Practice each deployment strategy in a test environment
2. Set up monitoring dashboards for your specific ML models
3. Implement automated testing for model quality
4. Create runbooks for common production scenarios
5. Move to Module 006 for Kubernetes orchestration

**Resources**:
- [Docker Production Best Practices](https://docs.docker.com/config/containers/resource_constraints/)
- [MLOps Principles](https://ml-ops.org/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Prometheus Best Practices](https://prometheus.io/docs/practices/)
