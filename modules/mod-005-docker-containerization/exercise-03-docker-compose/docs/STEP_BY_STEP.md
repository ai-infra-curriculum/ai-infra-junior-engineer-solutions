# Step-by-Step Implementation Guide: Docker Compose for ML Services

## Overview

Orchestrate multi-container ML applications with Docker Compose. Learn to define ML stacks with web APIs, databases, message queues, monitoring, and manage complex service dependencies.

**Time**: 2-3 hours | **Difficulty**: Intermediate

---

## Learning Objectives

✅ Define multi-service ML applications with docker-compose.yml
✅ Configure service dependencies and networks
✅ Manage volumes for data persistence
✅ Use environment variables and secrets
✅ Scale services dynamically
✅ Implement health checks and restart policies
✅ Orchestrate ML training and serving pipelines

---

## Quick Start: ML API Stack

### docker-compose.yml

```yaml
version: '3.8'

services:
  # ML API Service
  api:
    build: ./api
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:password@db:5432/mldb
      - REDIS_URL=redis://redis:6379
    depends_on:
      - db
      - redis
    volumes:
      - ./models:/app/models:ro
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 3s
      retries: 3

  # PostgreSQL Database
  db:
    image: postgres:15-alpine
    environment:
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=mldb
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    restart: unless-stopped

  # Redis Cache
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

  # MLflow Tracking Server
  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.8.0
    ports:
      - "5000:5000"
    environment:
      - BACKEND_STORE_URI=postgresql://postgres:password@db:5432/mldb
      - ARTIFACT_ROOT=/mlflow/artifacts
    volumes:
      - mlflow_data:/mlflow
    depends_on:
      - db
    command: >
      mlflow server
      --backend-store-uri postgresql://postgres:password@db:5432/mldb
      --default-artifact-root /mlflow/artifacts
      --host 0.0.0.0
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
  mlflow_data:
```

### Commands

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f api

# Stop all services
docker-compose down

# Stop and remove volumes
docker-compose down -v

# Rebuild and restart
docker-compose up -d --build

# Scale service
docker-compose up -d --scale api=3
```

---

## Complete ML Training Stack

```yaml
version: '3.8'

services:
  # Training Worker
  trainer:
    build:
      context: ./training
      dockerfile: Dockerfile.gpu
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    volumes:
      - ./data:/workspace/data:ro
      - ./checkpoints:/workspace/checkpoints
      - ./logs:/workspace/logs
    environment:
      - MLFLOW_TRACKING_URI=http://mlflow:5000
      - CUDA_VISIBLE_DEVICES=0
    depends_on:
      - mlflow
      - redis
    command: python train.py --config configs/train.yaml

  # Jupyter Lab
  jupyter:
    image: jupyter/pytorch-notebook:latest
    ports:
      - "8888:8888"
    volumes:
      - ./notebooks:/home/jovyan/work
      - ./data:/home/jovyan/data:ro
    environment:
      - JUPYTER_ENABLE_LAB=yes
    restart: unless-stopped

  # TensorBoard
  tensorboard:
    image: tensorflow/tensorflow:latest
    ports:
      - "6006:6006"
    volumes:
      - ./logs:/logs:ro
    command: tensorboard --logdir=/logs --host=0.0.0.0
    restart: unless-stopped

  # MLflow
  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.8.0
    ports:
      - "5000:5000"
    volumes:
      - mlflow_data:/mlflow
      - ./models:/mlflow/models
    environment:
      - BACKEND_STORE_URI=sqlite:///mlflow/mlflow.db
      - ARTIFACT_ROOT=/mlflow/artifacts
    command: >
      mlflow server
      --backend-store-uri sqlite:///mlflow/mlflow.db
      --default-artifact-root /mlflow/artifacts
      --host 0.0.0.0
    restart: unless-stopped

  # Redis for task queue
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

volumes:
  mlflow_data:
  redis_data:
```

---

## Best Practices

### 1. Environment Variables

```yaml
# .env file
DATABASE_URL=postgresql://postgres:password@localhost:5432/mldb
REDIS_URL=redis://localhost:6379
MODEL_PATH=/app/models/best_model.pth

# docker-compose.yml
services:
  api:
    env_file: .env
    # or
    environment:
      - DATABASE_URL=${DATABASE_URL}
```

### 2. Health Checks

```yaml
services:
  api:
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
```

### 3. Resource Limits

```yaml
services:
  trainer:
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 16G
        reservations:
          memory: 8G
```

### 4. Networks

```yaml
networks:
  frontend:
  backend:

services:
  api:
    networks:
      - frontend
      - backend

  db:
    networks:
      - backend
```

---

## Useful Commands

```bash
# Start in background
docker-compose up -d

# View running services
docker-compose ps

# Execute command in service
docker-compose exec api bash

# View logs
docker-compose logs -f api

# Restart service
docker-compose restart api

# Stop specific service
docker-compose stop api

# Remove stopped containers
docker-compose rm

# Pull latest images
docker-compose pull

# Validate config
docker-compose config

# Scale workers
docker-compose up -d --scale worker=4
```

---

**Docker Compose mastered!** 📦
