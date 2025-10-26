# Step-by-Step Implementation Guide: Production Deployment

## Overview

Deploy containerized ML applications to production. Learn deployment strategies, health checks, logging, monitoring, CI/CD integration, and production best practices for ML services.

**Time**: 3-4 hours | **Difficulty**: Advanced

---

## Learning Objectives

✅ Implement production-ready deployment configurations
✅ Configure health checks and readiness probes
✅ Set up centralized logging
✅ Implement monitoring and alerting
✅ Deploy with CI/CD pipelines
✅ Handle zero-downtime deployments
✅ Implement auto-scaling and load balancing

---

## Production docker-compose.yml

```yaml
version: '3.8'

services:
  # ML API (Load Balanced)
  api:
    image: registry.company.com/ml-api:${VERSION}
    deploy:
      replicas: 3
      restart_policy:
        condition: on-failure
        max_attempts: 3
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
    environment:
      - ENV=production
      - LOG_LEVEL=info
      - WORKERS=4
    secrets:
      - db_password
      - api_key
    networks:
      - frontend
      - backend
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

  # NGINX Reverse Proxy
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - api
    networks:
      - frontend
    restart: unless-stopped

  # PostgreSQL
  db:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=mldb
      - POSTGRES_USER=mluser
    secrets:
      - db_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./backups:/backups
    networks:
      - backend
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U mluser"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Redis Cache
  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    networks:
      - backend
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s

  # Prometheus Monitoring
  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus
    ports:
      - "9090:9090"
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
    networks:
      - monitoring
    restart: unless-stopped

  # Grafana Dashboards
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD__FILE=/run/secrets/grafana_password
    secrets:
      - grafana_password
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards:ro
    depends_on:
      - prometheus
    networks:
      - monitoring
    restart: unless-stopped

networks:
  frontend:
  backend:
    internal: true
  monitoring:

volumes:
  postgres_data:
  redis_data:
  prometheus_data:
  grafana_data:

secrets:
  db_password:
    external: true
  api_key:
    external: true
  grafana_password:
    external: true
```

---

## Health Checks

### Application Health Check

```python
# app.py
from fastapi import FastAPI, status

app = FastAPI()

@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """Liveness probe"""
    return {"status": "healthy"}

@app.get("/ready", status_code=status.HTTP_200_OK)
async def readiness_check():
    """Readiness probe - check dependencies"""
    # Check database
    try:
        await db.execute("SELECT 1")
    except Exception:
        return {"status": "not ready"}, 503

    # Check model loaded
    if not model_loaded:
        return {"status": "not ready"}, 503

    return {"status": "ready"}
```

---

## Logging Configuration

### Centralized Logging

```yaml
services:
  api:
    logging:
      driver: "fluentd"
      options:
        fluentd-address: localhost:24224
        tag: ml-api

  # Fluentd log aggregator
  fluentd:
    image: fluent/fluentd:latest
    ports:
      - "24224:24224"
    volumes:
      - ./fluentd/fluent.conf:/fluentd/etc/fluent.conf
      - fluentd_logs:/fluentd/log
```

---

## CI/CD Integration

### GitHub Actions

```yaml
# .github/workflows/deploy.yml
name: Deploy to Production

on:
  push:
    tags:
      - 'v*'

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Build and push
        run: |
          docker build -t registry.company.com/ml-api:${{ github.ref_name }} .
          docker push registry.company.com/ml-api:${{ github.ref_name }}

      - name: Deploy
        run: |
          export VERSION=${{ github.ref_name }}
          docker-compose -f docker-compose.prod.yml up -d
```

---

## Zero-Downtime Deployment

### Rolling Update

```bash
# Update one service at a time
docker-compose up -d --no-deps --scale api=4 --no-recreate api
docker-compose up -d --no-deps --scale api=3 --no-recreate api
```

### Blue-Green Deployment

```bash
# Deploy to green
docker-compose -f docker-compose.green.yml up -d

# Test green
curl http://green.company.com/health

# Switch traffic (update load balancer)
# ...

# Remove blue
docker-compose -f docker-compose.blue.yml down
```

---

## Monitoring

### Prometheus Metrics

```python
# Add metrics to API
from prometheus_client import Counter, Histogram, generate_latest

request_count = Counter('requests_total', 'Total requests')
request_duration = Histogram('request_duration_seconds', 'Request duration')

@app.middleware("http")
async def metrics_middleware(request, call_next):
    request_count.inc()
    with request_duration.time():
        response = await call_next(request)
    return response

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

---

## Backup Strategy

```bash
# Automated backup script
#!/bin/bash

DATE=$(date +%Y%m%d_%H%M%S)

# Backup database
docker exec postgres pg_dump -U mluser mldb | gzip > backups/db_${DATE}.sql.gz

# Backup volumes
docker run --rm \
  -v ml_data:/data \
  -v $(pwd)/backups:/backup \
  alpine tar czf /backup/data_${DATE}.tar.gz /data

# Upload to S3
aws s3 cp backups/ s3://company-backups/ml-api/ --recursive
```

---

## Production Checklist

✅ Health checks configured
✅ Resource limits set
✅ Secrets managed securely
✅ Logging centralized
✅ Monitoring and alerting active
✅ Backups automated
✅ SSL/TLS configured
✅ Auto-restart policies set
✅ Load balancing configured
✅ CI/CD pipeline tested
✅ Rollback procedure documented
✅ Disaster recovery plan ready

---

**Production deployment complete!** 🚀
