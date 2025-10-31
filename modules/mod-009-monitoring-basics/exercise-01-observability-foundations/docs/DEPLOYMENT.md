# Deployment Guide: Observability Foundations Lab

## Overview

This guide covers deploying the instrumented inference gateway across different environments: **Development**, **Staging**, and **Production**. Each environment has specific configurations optimized for different use cases.

---

## Table of Contents

1. [Development Environment](#development-environment)
2. [Staging Environment](#staging-environment)
3. [Production Environment](#production-environment)
4. [Kubernetes Deployment](#kubernetes-deployment)
5. [Monitoring Stack Deployment](#monitoring-stack-deployment)
6. [Configuration Management](#configuration-management)
7. [Deployment Strategies](#deployment-strategies)

---

## Development Environment

### Purpose
- Local development and testing
- Full observability stack for debugging
- Fast iteration cycles
- Minimal resource requirements

### Configuration

**`.env.development`**:
```bash
# Application
APP_NAME=inference-gateway-dev
APP_VERSION=dev
LOG_LEVEL=DEBUG
WORKERS=2

# Model
MODEL_NAME=resnet50
MODEL_WARMUP=true
MODEL_DEVICE=cpu

# Observability
ENABLE_METRICS=true
ENABLE_TRACING=true
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
LOG_FORMAT=text  # Console-friendly for development

# Performance
MAX_QUEUE_SIZE=10
REQUEST_TIMEOUT=30

# Development
RELOAD=true
DEBUG=true
```

### Deployment Steps

```bash
# 1. Copy development environment
cp .env.development .env

# 2. Install dependencies
python -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt

# 3. Run locally without Docker
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 4. Or run with Docker Compose
docker-compose -f docker-compose.dev.yml up -d

# 5. Access services
# - App: http://localhost:8000
# - Prometheus: http://localhost:9090
# - Jaeger: http://localhost:16686
```

### Development Docker Compose

**`docker-compose.dev.yml`**:
```yaml
version: '3.8'

services:
  inference-gateway:
    build:
      context: .
      dockerfile: Dockerfile.dev
    volumes:
      - ./app:/app/app  # Mount code for hot reload
    environment:
      - LOG_LEVEL=DEBUG
      - RELOAD=true
      - DEBUG=true
    ports:
      - "8000:8000"
    command: uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

  prometheus:
    image: prom/prometheus:v2.48.0
    volumes:
      - ./config/prometheus-dev.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"

  jaeger:
    image: jaegertracing/all-in-one:1.53
    ports:
      - "16686:16686"  # UI
      - "4318:4318"    # OTLP receiver
```

### Cost
**~$0/month** (runs locally)

---

## Staging Environment

### Purpose
- Pre-production testing
- Load testing and performance validation
- Integration testing with other services
- SLO validation

### Configuration

**`.env.staging`**:
```bash
# Application
APP_NAME=inference-gateway-staging
APP_VERSION=1.0.0
LOG_LEVEL=INFO
WORKERS=4

# Model
MODEL_NAME=resnet50
MODEL_WARMUP=true
MODEL_DEVICE=cuda  # GPU if available

# Observability
ENABLE_METRICS=true
ENABLE_TRACING=true
OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger-collector.monitoring:4318
LOG_FORMAT=json

# Performance
MAX_QUEUE_SIZE=100
REQUEST_TIMEOUT=30

# Production-like settings
RELOAD=false
DEBUG=false
```

### AWS ECS Deployment (Staging)

**`ecs-task-definition-staging.json`**:
```json
{
  "family": "inference-gateway-staging",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "4096",
  "containerDefinitions": [
    {
      "name": "inference-gateway",
      "image": "YOUR_ECR_REPO/inference-gateway:staging",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {"name": "APP_NAME", "value": "inference-gateway-staging"},
        {"name": "LOG_LEVEL", "value": "INFO"},
        {"name": "ENABLE_METRICS", "value": "true"},
        {"name": "ENABLE_TRACING", "value": "true"},
        {"name": "MODEL_WARMUP", "value": "true"}
      ],
      "secrets": [
        {
          "name": "OTEL_EXPORTER_OTLP_ENDPOINT",
          "valueFrom": "arn:aws:secretsmanager:REGION:ACCOUNT:secret:jaeger-endpoint"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/inference-gateway-staging",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3,
        "startPeriod": 60
      }
    }
  ]
}
```

### Deployment Commands

```bash
# 1. Build and push Docker image
docker build -t inference-gateway:staging .
docker tag inference-gateway:staging YOUR_ECR_REPO/inference-gateway:staging
docker push YOUR_ECR_REPO/inference-gateway:staging

# 2. Register task definition
aws ecs register-task-definition --cli-input-json file://ecs-task-definition-staging.json

# 3. Update ECS service
aws ecs update-service \
  --cluster inference-cluster-staging \
  --service inference-gateway \
  --task-definition inference-gateway-staging:LATEST \
  --force-new-deployment

# 4. Wait for deployment
aws ecs wait services-stable \
  --cluster inference-cluster-staging \
  --services inference-gateway
```

### Cost
**~$70-100/month**
- ECS Fargate: 2 vCPU, 4GB RAM = ~$60/month
- ALB: ~$20/month
- CloudWatch Logs: ~$5-10/month

---

## Production Environment

### Purpose
- Serve production traffic
- High availability and reliability
- Complete observability and monitoring
- Auto-scaling based on load

### Configuration

**`.env.production`**:
```bash
# Application
APP_NAME=inference-gateway
APP_VERSION=1.0.0
LOG_LEVEL=INFO
WORKERS=8

# Model
MODEL_NAME=resnet50
MODEL_WARMUP=true
MODEL_DEVICE=cuda

# Observability
ENABLE_METRICS=true
ENABLE_TRACING=true
OTEL_EXPORTER_OTLP_ENDPOINT=https://jaeger-collector.prod.internal:4318
LOG_FORMAT=json

# Performance
MAX_QUEUE_SIZE=500
REQUEST_TIMEOUT=30
GRACEFUL_SHUTDOWN_TIMEOUT=30

# Production
RELOAD=false
DEBUG=false
```

### Kubernetes Production Deployment

**`k8s/production/deployment.yaml`**:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: inference-gateway
  namespace: ml-inference
  labels:
    app: inference-gateway
    environment: production
    team: ml-platform
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: inference-gateway
  template:
    metadata:
      labels:
        app: inference-gateway
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
        prometheus.io/path: "/metrics"
        version: "1.0.0"
    spec:
      serviceAccountName: inference-gateway
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 1000
      containers:
      - name: inference-gateway
        image: YOUR_ECR_REPO/inference-gateway:1.0.0
        imagePullPolicy: IfNotPresent
        ports:
        - name: http
          containerPort: 8000
          protocol: TCP
        env:
        - name: APP_NAME
          value: "inference-gateway"
        - name: LOG_LEVEL
          value: "INFO"
        - name: ENABLE_METRICS
          value: "true"
        - name: ENABLE_TRACING
          value: "true"
        - name: OTEL_EXPORTER_OTLP_ENDPOINT
          valueFrom:
            configMapKeyRef:
              name: observability-config
              key: jaeger-endpoint
        - name: WORKERS
          value: "8"
        resources:
          requests:
            memory: "4Gi"
            cpu: "2000m"
          limits:
            memory: "8Gi"
            cpu: "4000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 5
          timeoutSeconds: 5
          successThreshold: 1
          failureThreshold: 3
---
apiVersion: v1
kind: Service
metadata:
  name: inference-gateway
  namespace: ml-inference
  labels:
    app: inference-gateway
  annotations:
    service.beta.kubernetes.io/aws-load-balancer-type: "nlb"
spec:
  type: LoadBalancer
  selector:
    app: inference-gateway
  ports:
  - name: http
    port: 80
    targetPort: 8000
    protocol: TCP
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: inference-gateway-hpa
  namespace: ml-inference
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: inference-gateway
  minReplicas: 3
  maxReplicas: 20
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
  - type: Pods
    pods:
      metric:
        name: http_requests_per_second
      target:
        type: AverageValue
        averageValue: "1000"
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100
        periodSeconds: 30
```

### Production Deployment Process

```bash
# 1. Build production image
docker build -t inference-gateway:1.0.0 -f Dockerfile .

# 2. Tag and push to ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin YOUR_ECR_REPO
docker tag inference-gateway:1.0.0 YOUR_ECR_REPO/inference-gateway:1.0.0
docker push YOUR_ECR_REPO/inference-gateway:1.0.0

# 3. Create namespace
kubectl create namespace ml-inference

# 4. Create ConfigMaps and Secrets
kubectl apply -f k8s/production/configmap.yaml
kubectl apply -f k8s/production/secrets.yaml

# 5. Deploy application
kubectl apply -f k8s/production/deployment.yaml

# 6. Verify deployment
kubectl rollout status deployment/inference-gateway -n ml-inference

# 7. Check pods
kubectl get pods -n ml-inference -l app=inference-gateway

# 8. Test endpoints
GATEWAY_URL=$(kubectl get svc inference-gateway -n ml-inference -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
curl http://$GATEWAY_URL/health
curl http://$GATEWAY_URL/ready
```

### Cost
**~$350-500/month** (AWS)
- EKS cluster: ~$75/month
- 3 EC2 instances (m5.xlarge): ~$250/month
- Load Balancer: ~$20/month
- CloudWatch: ~$10-20/month
- Data transfer: ~$10-20/month

---

## Kubernetes Deployment

### Complete Kubernetes Manifests

**`k8s/production/configmap.yaml`**:
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: observability-config
  namespace: ml-inference
data:
  jaeger-endpoint: "http://jaeger-collector.monitoring.svc.cluster.local:4318"
  prometheus-endpoint: "http://prometheus.monitoring.svc.cluster.local:9090"
  log-level: "INFO"
```

**`k8s/production/service-monitor.yaml`** (for Prometheus Operator):
```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: inference-gateway
  namespace: ml-inference
  labels:
    app: inference-gateway
spec:
  selector:
    matchLabels:
      app: inference-gateway
  endpoints:
  - port: http
    path: /metrics
    interval: 15s
    scrapeTimeout: 10s
```

---

## Monitoring Stack Deployment

### Deploy Prometheus, Grafana, Jaeger

**Option 1: Helm Charts (Recommended)**

```bash
# Add Helm repos
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add grafana https://grafana.github.io/helm-charts
helm repo add jaegertracing https://jaegertracing.github.io/helm-charts
helm repo update

# Create monitoring namespace
kubectl create namespace monitoring

# Deploy Prometheus + Grafana stack
helm install kube-prometheus-stack prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false \
  --set grafana.enabled=true \
  --set grafana.adminPassword=admin

# Deploy Jaeger
helm install jaeger jaegertracing/jaeger \
  --namespace monitoring \
  --set provisionDataStore.cassandra=false \
  --set allInOne.enabled=true \
  --set storage.type=memory

# Verify deployments
kubectl get pods -n monitoring
```

**Option 2: Docker Compose (Development/Staging)**

See `docker-compose.yml` in project root.

---

## Configuration Management

### Managing Secrets

**AWS Secrets Manager**:
```bash
# Store secrets
aws secretsmanager create-secret \
  --name inference-gateway/prod/config \
  --secret-string '{
    "otel_endpoint": "https://jaeger-collector.prod.internal:4318",
    "api_key": "secret-api-key"
  }'

# Reference in Kubernetes
apiVersion: v1
kind: Secret
metadata:
  name: inference-gateway-secrets
  namespace: ml-inference
type: Opaque
stringData:
  otel-endpoint: "https://jaeger-collector.prod.internal:4318"
```

### Environment-Specific Configurations

**Kustomize** (Kubernetes):
```bash
k8s/
├── base/
│   ├── deployment.yaml
│   ├── service.yaml
│   └── kustomization.yaml
├── overlays/
│   ├── development/
│   │   └── kustomization.yaml
│   ├── staging/
│   │   └── kustomization.yaml
│   └── production/
│       └── kustomization.yaml

# Deploy to production
kubectl apply -k k8s/overlays/production
```

---

## Deployment Strategies

### 1. Blue-Green Deployment

```bash
# Deploy new version (green)
kubectl apply -f k8s/production/deployment-v2.yaml

# Test green deployment
curl http://inference-gateway-v2.ml-inference.svc.cluster.local/health

# Switch traffic (update service selector)
kubectl patch service inference-gateway -n ml-inference \
  -p '{"spec":{"selector":{"version":"2.0.0"}}}'

# Rollback if needed
kubectl patch service inference-gateway -n ml-inference \
  -p '{"spec":{"selector":{"version":"1.0.0"}}}'
```

### 2. Canary Deployment

```yaml
apiVersion: flagger.app/v1beta1
kind: Canary
metadata:
  name: inference-gateway
  namespace: ml-inference
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: inference-gateway
  service:
    port: 80
  analysis:
    interval: 1m
    threshold: 5
    maxWeight: 50
    stepWeight: 10
    metrics:
    - name: request-success-rate
      thresholdRange:
        min: 99
      interval: 1m
    - name: request-duration
      thresholdRange:
        max: 500
      interval: 1m
```

### 3. Rolling Update (Default)

Already configured in the Deployment YAML above with:
```yaml
strategy:
  type: RollingUpdate
  rollingUpdate:
    maxSurge: 1
    maxUnavailable: 0
```

---

## Rollback Procedures

### Kubernetes Rollback

```bash
# View rollout history
kubectl rollout history deployment/inference-gateway -n ml-inference

# Rollback to previous version
kubectl rollout undo deployment/inference-gateway -n ml-inference

# Rollback to specific revision
kubectl rollout undo deployment/inference-gateway -n ml-inference --to-revision=2

# Monitor rollback
kubectl rollout status deployment/inference-gateway -n ml-inference
```

### ECS Rollback

```bash
# List task definitions
aws ecs list-task-definitions --family-prefix inference-gateway

# Update service to previous task definition
aws ecs update-service \
  --cluster inference-cluster \
  --service inference-gateway \
  --task-definition inference-gateway:PREVIOUS_VERSION
```

---

## Post-Deployment Validation

### Smoke Tests

```bash
#!/bin/bash
# smoke-test.sh

GATEWAY_URL=$1

echo "Running smoke tests for $GATEWAY_URL"

# Test 1: Health check
echo "Test 1: Health check"
curl -f $GATEWAY_URL/health || exit 1

# Test 2: Readiness check
echo "Test 2: Readiness check"
curl -f $GATEWAY_URL/ready || exit 1

# Test 3: Prediction
echo "Test 3: Prediction"
curl -X POST $GATEWAY_URL/predict \
  -F "file=@test_image.jpg" \
  -f || exit 1

# Test 4: Metrics endpoint
echo "Test 4: Metrics endpoint"
curl -f $GATEWAY_URL/metrics | grep http_requests_total || exit 1

echo "All smoke tests passed!"
```

### SLO Validation

```bash
# Query Prometheus for SLO compliance
curl -s 'http://prometheus:9090/api/v1/query?query=slo:availability:ratio_rate30d' | \
  jq -r '.data.result[0].value[1]'

# Expected: > 0.995 (99.5%)
```

---

## Summary

| Environment | Purpose | Replicas | Resources | Cost/Month |
|-------------|---------|----------|-----------|------------|
| Development | Local testing | 1 | Minimal | $0 |
| Staging | Pre-prod validation | 2 | Medium | $70-100 |
| Production | Live traffic | 3-20 (auto-scale) | High | $350-500 |

**Deployment Best Practices**:
- ✅ Use infrastructure as code (Terraform, Kubernetes manifests)
- ✅ Automate deployments with CI/CD pipelines
- ✅ Implement health checks for all environments
- ✅ Use blue-green or canary deployments for production
- ✅ Monitor SLOs after every deployment
- ✅ Have rollback procedures ready
- ✅ Run smoke tests post-deployment
- ✅ Use separate observability stacks per environment
