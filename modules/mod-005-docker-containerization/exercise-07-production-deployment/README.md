# Exercise 07: Production Deployment

Master production-ready deployment strategies for containerized ML applications including orchestration, monitoring, logging, and CI/CD integration.

## Learning Objectives

- Deploy containerized ML applications to production
- Implement health checks and readiness probes
- Set up centralized logging and monitoring
- Configure load balancing and auto-scaling
- Integrate CI/CD pipelines for automated deployments
- Implement zero-downtime deployments
- Handle secrets and configuration management
- Set up disaster recovery and backups

## Production Deployment Patterns

### 1. Blue-Green Deployment

Deploy new version alongside old, then switch traffic:

```bash
# Deploy green version
docker-compose -f docker-compose.green.yml up -d

# Test green version
curl http://green.myapp.com/health

# Switch traffic (update load balancer)
# If issues, instantly rollback to blue
```

**Benefits**: Zero downtime, instant rollback
**Drawbacks**: Requires 2x resources temporarily

### 2. Canary Deployment

Gradually shift traffic to new version:

```bash
# Deploy canary with 10% traffic
docker service update --replicas 1 ml-api-canary

# Monitor metrics
# If successful, increase to 50%, then 100%
docker service update --replicas 5 ml-api-canary
```

**Benefits**: Risk mitigation, gradual validation
**Drawbacks**: Requires traffic splitting capability

### 3. Rolling Update

Update containers one at a time:

```bash
docker service update \
  --update-parallelism 1 \
  --update-delay 30s \
  --image myapp:v2 \
  ml-api
```

**Benefits**: No extra resources needed
**Drawbacks**: Slower deployment, mixed versions running

## Health Checks

### Application Health

```python
# app/health.py
from fastapi import FastAPI, status
from fastapi.responses import JSONResponse
import psutil
import time

app = FastAPI()

startup_time = time.time()

@app.get("/health")
async def health_check():
    """Basic health check - is app running?"""
    return {"status": "healthy", "timestamp": time.time()}

@app.get("/health/ready")
async def readiness_check():
    """Readiness check - can app handle traffic?"""
    checks = {
        "database": check_database(),
        "model_loaded": check_model(),
        "disk_space": psutil.disk_usage('/').percent < 90,
        "memory": psutil.virtual_memory().percent < 90
    }

    if all(checks.values()):
        return {"status": "ready", "checks": checks}
    else:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "not_ready", "checks": checks}
        )

@app.get("/health/live")
async def liveness_check():
    """Liveness check - should container be restarted?"""
    # Check if app is deadlocked or hung
    uptime = time.time() - startup_time
    return {"status": "alive", "uptime_seconds": uptime}
```

### Docker Compose Health Checks

```yaml
services:
  ml-api:
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health/ready"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s  # Allow time for model loading
```

### Kubernetes Probes

```yaml
apiVersion: v1
kind: Pod
spec:
  containers:
  - name: ml-api
    livenessProbe:
      httpGet:
        path: /health/live
        port: 8000
      initialDelaySeconds: 30
      periodSeconds: 10
    readinessProbe:
      httpGet:
        path: /health/ready
        port: 8000
      initialDelaySeconds: 60
      periodSeconds: 5
```

## Logging

### Centralized Logging Stack (EFK)

```yaml
version: '3.8'

services:
  # Application
  ml-api:
    logging:
      driver: "fluentd"
      options:
        fluentd-address: localhost:24224
        tag: ml-api

  # Fluentd log aggregator
  fluentd:
    image: fluent/fluentd:v1.16-1
    ports:
      - "24224:24224"
    volumes:
      - ./fluentd/fluent.conf:/fluentd/etc/fluent.conf
      - fluentd-data:/fluentd/log

  # Elasticsearch for log storage
  elasticsearch:
    image: elasticsearch:8.11.0
    environment:
      - discovery.type=single-node
      - "ES_JAVA_OPTS=-Xms512m -Xmx512m"
    volumes:
      - es-data:/usr/share/elasticsearch/data

  # Kibana for log visualization
  kibana:
    image: kibana:8.11.0
    ports:
      - "5601:5601"
    depends_on:
      - elasticsearch

volumes:
  fluentd-data:
  es-data:
```

### Structured Logging

```python
import logging
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }

        # Add extra fields
        if hasattr(record, 'request_id'):
            log_data['request_id'] = record.request_id
        if hasattr(record, 'user_id'):
            log_data['user_id'] = record.user_id

        return json.dumps(log_data)

# Configure logger
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Usage
logger.info("Model prediction completed", extra={
    "request_id": "abc123",
    "model_version": "v2.1.0",
    "latency_ms": 45
})
```

## Monitoring

### Prometheus + Grafana

```yaml
# docker-compose.monitoring.yml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    ports:
      - "9090:9090"
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--storage.tsdb.retention.time=30d'

  grafana:
    image: grafana/grafana:latest
    volumes:
      - grafana-data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./grafana/datasources:/etc/grafana/provisioning/datasources
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_INSTALL_PLUGINS=grafana-piechart-panel

  # ML API with Prometheus metrics
  ml-api:
    environment:
      - PROMETHEUS_MULTIPROC_DIR=/tmp/prometheus
    ports:
      - "8000:8000"
      - "8001:8001"  # Metrics endpoint

volumes:
  prometheus-data:
  grafana-data:
```

### Application Metrics

```python
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from fastapi import FastAPI
import time

app = FastAPI()

# Metrics
prediction_counter = Counter('predictions_total', 'Total predictions', ['model_version', 'status'])
prediction_latency = Histogram('prediction_latency_seconds', 'Prediction latency', ['model_version'])
model_load_gauge = Gauge('model_loaded', 'Is model loaded', ['model_version'])
active_requests = Gauge('active_requests', 'Number of active requests')

@app.post("/predict")
async def predict(data: dict):
    active_requests.inc()
    start_time = time.time()

    try:
        result = await model.predict(data)
        prediction_counter.labels(model_version='v1', status='success').inc()
        return result
    except Exception as e:
        prediction_counter.labels(model_version='v1', status='error').inc()
        raise
    finally:
        latency = time.time() - start_time
        prediction_latency.labels(model_version='v1').observe(latency)
        active_requests.dec()

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

## Secrets Management

### Docker Secrets

```bash
# Create secret
echo "db_password_here" | docker secret create db_password -

# Use in docker-compose
services:
  ml-api:
    secrets:
      - db_password
    environment:
      - DATABASE_PASSWORD_FILE=/run/secrets/db_password

secrets:
  db_password:
    external: true
```

### HashiCorp Vault

```python
import hvac

# Connect to Vault
client = hvac.Client(url='http://vault:8200', token=os.getenv('VAULT_TOKEN'))

# Read secrets
secrets = client.secrets.kv.v2.read_secret_version(path='ml-api/config')
db_password = secrets['data']['data']['db_password']
api_key = secrets['data']['data']['api_key']
```

### Environment-based Config

```bash
# .env.production (NOT committed to git)
DATABASE_URL=postgresql://user:pass@db:5432/mldb
MODEL_S3_BUCKET=ml-models-prod
API_KEY=secret_key_here
LOG_LEVEL=info

# Load with docker-compose
services:
  ml-api:
    env_file:
      - .env.production
```

## CI/CD Integration

### GitHub Actions Pipeline

```yaml
# .github/workflows/deploy.yml
name: Build and Deploy

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Run tests
        run: |
          docker-compose -f docker-compose.test.yml up --abort-on-container-exit
          docker-compose -f docker-compose.test.yml down

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Build Docker image
        run: |
          docker build -t ml-api:${{ github.sha }} .
          docker tag ml-api:${{ github.sha }} ml-api:latest

      - name: Scan for vulnerabilities
        run: |
          docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
            aquasec/trivy image ml-api:${{ github.sha }}

      - name: Push to registry
        run: |
          echo "${{ secrets.DOCKER_PASSWORD }}" | docker login -u "${{ secrets.DOCKER_USERNAME }}" --password-stdin
          docker push ml-api:${{ github.sha }}
          docker push ml-api:latest

  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - name: Deploy to production
        run: |
          # SSH to production server and update containers
          ssh deploy@production "cd /app && docker-compose pull && docker-compose up -d"
```

### GitLab CI/CD

```yaml
# .gitlab-ci.yml
stages:
  - test
  - build
  - deploy

test:
  stage: test
  script:
    - docker-compose -f docker-compose.test.yml up --abort-on-container-exit
  only:
    - merge_requests
    - main

build:
  stage: build
  script:
    - docker build -t $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA .
    - docker tag $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA $CI_REGISTRY_IMAGE:latest
    - docker login -u $CI_REGISTRY_USER -p $CI_REGISTRY_PASSWORD $CI_REGISTRY
    - docker push $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA
    - docker push $CI_REGISTRY_IMAGE:latest
  only:
    - main

deploy_staging:
  stage: deploy
  script:
    - kubectl set image deployment/ml-api ml-api=$CI_REGISTRY_IMAGE:$CI_COMMIT_SHA -n staging
    - kubectl rollout status deployment/ml-api -n staging
  only:
    - main
  environment:
    name: staging

deploy_production:
  stage: deploy
  script:
    - kubectl set image deployment/ml-api ml-api=$CI_REGISTRY_IMAGE:$CI_COMMIT_SHA -n production
    - kubectl rollout status deployment/ml-api -n production
  only:
    - main
  when: manual
  environment:
    name: production
```

## Load Balancing

### NGINX Load Balancer

```nginx
# nginx.conf
upstream ml_api {
    least_conn;  # Route to server with fewest connections

    server ml-api-1:8000 max_fails=3 fail_timeout=30s;
    server ml-api-2:8000 max_fails=3 fail_timeout=30s;
    server ml-api-3:8000 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    server_name api.example.com;

    location / {
        proxy_pass http://ml_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;

        # Timeouts for ML inference
        proxy_connect_timeout 60s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
    }

    location /health {
        access_log off;
        proxy_pass http://ml_api/health;
    }
}
```

### Docker Compose with NGINX

```yaml
version: '3.8'

services:
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - ml-api-1
      - ml-api-2
      - ml-api-3

  ml-api-1:
    image: ml-api:latest
    environment:
      - INSTANCE_ID=1

  ml-api-2:
    image: ml-api:latest
    environment:
      - INSTANCE_ID=2

  ml-api-3:
    image: ml-api:latest
    environment:
      - INSTANCE_ID=3
```

## Auto-Scaling

### Docker Swarm Auto-Scaling

```bash
# Create service with auto-scaling
docker service create \
  --name ml-api \
  --replicas 2 \
  --limit-cpu 1 \
  --limit-memory 2G \
  --reserve-cpu 0.5 \
  --reserve-memory 1G \
  ml-api:latest

# Manual scaling
docker service scale ml-api=5

# Monitor and auto-scale (external script)
# Scale based on CPU/memory/request rate
```

### Kubernetes Horizontal Pod Autoscaler

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ml-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ml-api
  minReplicas: 2
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
  - type: Pods
    pods:
      metric:
        name: http_requests_per_second
      target:
        type: AverageValue
        averageValue: "1000"
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300  # Wait 5 min before scaling down
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 0
      policies:
      - type: Percent
        value: 100
        periodSeconds: 30
      - type: Pods
        value: 2
        periodSeconds: 30
```

## Backup and Disaster Recovery

### Volume Backups

```bash
#!/bin/bash
# backup_volumes.sh

BACKUP_DIR="/backups/$(date +%Y%m%d)"
mkdir -p "$BACKUP_DIR"

# Backup named volumes
for volume in $(docker volume ls -q); do
    echo "Backing up volume: $volume"
    docker run --rm \
        -v "$volume":/data \
        -v "$BACKUP_DIR":/backup \
        alpine \
        tar czf "/backup/${volume}.tar.gz" -C /data .
done

# Upload to S3
aws s3 sync "$BACKUP_DIR" "s3://my-backups/docker-volumes/$(date +%Y%m%d)/"
```

### Database Backups

```bash
#!/bin/bash
# backup_database.sh

BACKUP_FILE="postgres_$(date +%Y%m%d_%H%M%S).sql.gz"

docker exec postgres pg_dump -U mluser mldb | gzip > "/backups/$BACKUP_FILE"

# Upload to S3
aws s3 cp "/backups/$BACKUP_FILE" "s3://my-backups/postgres/"

# Retain only last 30 days locally
find /backups -name "postgres_*.sql.gz" -mtime +30 -delete
```

### Restore Procedures

```bash
# Restore volume
docker run --rm \
    -v ml-data:/data \
    -v /backups:/backup \
    alpine \
    tar xzf /backup/ml-data.tar.gz -C /data

# Restore database
gunzip < /backups/postgres_20241024_120000.sql.gz | \
    docker exec -i postgres psql -U mluser mldb
```

## Production Checklist

### Pre-Deployment
- [ ] All tests passing
- [ ] Security scan completed (no critical vulnerabilities)
- [ ] Environment variables configured
- [ ] Secrets properly managed
- [ ] Resource limits set
- [ ] Health checks configured
- [ ] Monitoring dashboards created
- [ ] Alerts configured
- [ ] Backup strategy tested
- [ ] Rollback plan documented

### During Deployment
- [ ] Deploy to staging first
- [ ] Run smoke tests
- [ ] Monitor error rates
- [ ] Check resource usage
- [ ] Verify health checks passing
- [ ] Test critical user flows

### Post-Deployment
- [ ] Monitor for 1-2 hours
- [ ] Check logs for errors
- [ ] Verify metrics are normal
- [ ] Test rollback procedure
- [ ] Document any issues
- [ ] Update runbooks

## Common Production Issues

### Issue: Container Out of Memory

```bash
# Symptoms
docker logs ml-api
# OOMKilled

# Solution 1: Increase memory limit
docker-compose.yml:
  ml-api:
    deploy:
      resources:
        limits:
          memory: 4G

# Solution 2: Fix memory leak
# Profile Python memory usage
import tracemalloc
tracemalloc.start()
# ... run code
snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')
```

### Issue: Slow Response Times

```bash
# Check container resources
docker stats ml-api

# Check logs for bottlenecks
docker logs ml-api --tail 1000 | grep -i "slow\|timeout"

# Profile with cProfile
python -m cProfile -o profile.stats app.py

# Analyze with snakeviz
snakeviz profile.stats
```

### Issue: Database Connection Pool Exhausted

```python
# Solution: Configure connection pooling
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=20,          # Max connections
    max_overflow=10,       # Allow 10 more if pool full
    pool_timeout=30,       # Wait 30s for connection
    pool_recycle=3600,     # Recycle connections after 1h
    pool_pre_ping=True     # Verify connection before use
)
```

## Solutions

See `solutions/` directory for:
- `production_deploy.py` - Complete deployment orchestration tool
- `health_monitor.py` - Production health monitoring dashboard
- `log_aggregator.py` - Centralized logging aggregator
- `backup_manager.sh` - Automated backup and restore scripts
- `docker-compose.prod.yml` - Production Docker Compose configuration
- `k8s/` - Kubernetes manifests for production deployment

## Quick Start

```bash
cd solutions/

# Deploy full production stack
python production_deploy.py --environment production --strategy rolling

# Monitor deployment health
python health_monitor.py --environment production --interval 5

# Create backup
./backup_manager.sh backup --type full

# Restore from backup
./backup_manager.sh restore --backup-id 20241024_120000
```

## Resources

- [Docker Production Best Practices](https://docs.docker.com/config/containers/resource_constraints/)
- [12 Factor App Methodology](https://12factor.net/)
- [Kubernetes Production Best Practices](https://kubernetes.io/docs/setup/best-practices/)
- [Prometheus Best Practices](https://prometheus.io/docs/practices/naming/)
- [EFK Stack Documentation](https://www.elastic.co/what-is/elk-stack)

## Next Steps

After mastering production deployment:
1. Move to **Module 006: Kubernetes & Orchestration** for advanced container orchestration
2. Explore **Module 007: CI/CD Pipelines** for automated delivery
3. Learn **Module 008: Cloud Platforms** for AWS/GCP/Azure deployments
