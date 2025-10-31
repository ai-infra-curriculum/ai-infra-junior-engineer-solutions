# Module 005: Docker & Containerization

## Overview

Master Docker and containerization for ML infrastructure. Learn to containerize ML applications, optimize Docker images, manage multi-container deployments, and implement container orchestration for production ML systems.

## Learning Objectives

- ✅ Understand container fundamentals and Docker architecture
- ✅ Build optimized Docker images for ML applications
- ✅ Manage container networking and storage
- ✅ Orchestrate multi-container ML applications
- ✅ Implement container security best practices
- ✅ Optimize Docker performance for ML workloads
- ✅ Deploy containerized ML systems to production

## Module Structure

### Exercise 01: Docker Fundamentals
- Docker architecture and concepts
- Basic Docker commands
- Container lifecycle management
- Docker images and registries
- Dockerfile basics

### Exercise 02: Building ML Docker Images
- Dockerfile optimization for ML
- Multi-stage builds
- Layer caching strategies
- GPU support in containers
- Image size optimization

### Exercise 03: Docker Compose & Multi-Container Apps
- Docker Compose fundamentals
- Multi-container ML applications
- Service dependencies
- Environment configuration
- Development workflows

### Exercise 04: Docker Networking
- Container networking modes
- Custom networks
- Service discovery
- Port mapping and exposure
- Network security

### Exercise 05: Docker Volumes & Data Management
- Volume types and usage
- Bind mounts vs volumes
- Data persistence strategies
- Backup and restore
- Volume drivers

### Exercise 06: Container Security
- Security best practices
- Image scanning
- User permissions
- Secrets management
- Runtime security

### Exercise 07: Production Deployment
- Container orchestration intro
- Load balancing
- Health checks
- Logging and monitoring
- CI/CD with Docker

## Prerequisites

- Linux command line proficiency
- Basic understanding of ML workflows
- Python programming skills
- Familiarity with ML frameworks

## Tools & Technologies

- Docker Engine
- Docker Compose
- Docker Hub / Container Registry
- NVIDIA Container Toolkit (for GPU)
- Docker BuildKit
- Dive (image analysis)
- Trivy (security scanning)

## Docker Installation

### Ubuntu/Debian

```bash
# Remove old versions
sudo apt-get remove docker docker-engine docker.io containerd runc

# Install dependencies
sudo apt-get update
sudo apt-get install -y \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

# Add Docker's official GPG key
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
    sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# Set up repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker Engine
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Verify installation
sudo docker run hello-world

# Add user to docker group (optional)
sudo usermod -aG docker $USER
newgrp docker
```

### NVIDIA Container Toolkit (for GPU support)

```bash
# Add NVIDIA repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install NVIDIA Container Toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker daemon
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Test GPU access
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

## Key Concepts

### 1. Container vs Virtual Machine

```
Virtual Machines:
┌─────────────────────────────┐
│     Application Layer       │
├─────────────────────────────┤
│      Guest OS (Full)        │
├─────────────────────────────┤
│       Hypervisor            │
├─────────────────────────────┤
│       Host OS               │
├─────────────────────────────┤
│       Hardware              │
└─────────────────────────────┘

Containers:
┌─────────────────────────────┐
│     Application Layer       │
├─────────────────────────────┤
│    Container Runtime        │
├─────────────────────────────┤
│       Host OS               │
├─────────────────────────────┤
│       Hardware              │
└─────────────────────────────┘
```

**Benefits of Containers:**
- Lightweight (MBs vs GBs)
- Fast startup (seconds vs minutes)
- Better resource utilization
- Consistent environments
- Easy scaling

### 2. Docker Architecture

```
┌──────────────────────────────────────────┐
│          Docker Client (CLI)             │
└────────────────┬─────────────────────────┘
                 │ Docker API
┌────────────────▼─────────────────────────┐
│         Docker Daemon (dockerd)          │
│  ┌──────────────────────────────────┐   │
│  │         Container Runtime         │   │
│  │    ┌────────┐  ┌────────┐       │   │
│  │    │Container│  │Container│       │   │
│  │    └────────┘  └────────┘       │   │
│  └──────────────────────────────────┘   │
└────────────────┬─────────────────────────┘
                 │
┌────────────────▼─────────────────────────┐
│          Image Registry                  │
│     (Docker Hub, ECR, GCR, etc.)        │
└──────────────────────────────────────────┘
```

### 3. Docker Images and Layers

Docker images are built in layers:

```
┌─────────────────────────────────┐
│  Application Layer (Python app) │  ← Layer 4 (Added)
├─────────────────────────────────┤
│  Dependencies (pip packages)    │  ← Layer 3 (Added)
├─────────────────────────────────┤
│  Python Runtime                 │  ← Layer 2 (Base)
├─────────────────────────────────┤
│  Base OS (Ubuntu/Alpine)        │  ← Layer 1 (Base)
└─────────────────────────────────┘

Each layer is cached and reused!
```

### 4. Container Lifecycle

```
┌─────────┐   docker create   ┌─────────┐
│  Image  │ ─────────────────→│ Created │
└─────────┘                    └────┬────┘
                                    │ docker start
                               ┌────▼────┐
                               │ Running │
                               └────┬────┘
                                    │ docker stop
                               ┌────▼────┐
                               │ Stopped │
                               └────┬────┘
                                    │ docker rm
                               ┌────▼────┐
                               │ Removed │
                               └─────────┘
```

## Essential Docker Commands

### Image Management

```bash
# Pull an image
docker pull ubuntu:22.04

# List images
docker images
docker image ls

# Build an image
docker build -t my-app:v1.0 .

# Tag an image
docker tag my-app:v1.0 myregistry/my-app:v1.0

# Push to registry
docker push myregistry/my-app:v1.0

# Remove image
docker rmi my-app:v1.0

# Remove all unused images
docker image prune -a

# Inspect image
docker image inspect my-app:v1.0

# View image history
docker history my-app:v1.0
```

### Container Management

```bash
# Run container
docker run -d --name my-container my-app:v1.0

# Run with options
docker run -d \
    --name my-container \
    -p 8000:8000 \
    -v /data:/app/data \
    -e ENV_VAR=value \
    --restart unless-stopped \
    my-app:v1.0

# List running containers
docker ps

# List all containers
docker ps -a

# Stop container
docker stop my-container

# Start container
docker start my-container

# Restart container
docker restart my-container

# Remove container
docker rm my-container

# Remove all stopped containers
docker container prune

# View logs
docker logs my-container
docker logs -f my-container  # Follow logs

# Execute command in container
docker exec -it my-container bash

# Inspect container
docker inspect my-container

# View container stats
docker stats my-container

# Copy files
docker cp my-container:/app/file.txt ./
docker cp ./file.txt my-container:/app/
```

### System Management

```bash
# View Docker info
docker info

# View disk usage
docker system df

# Clean up everything
docker system prune -a --volumes

# View events
docker events

# View version
docker version
```

## Dockerfile Best Practices

### 1. Multi-Stage Builds

```dockerfile
# Stage 1: Build
FROM python:3.10 AS builder

WORKDIR /build

# Install dependencies
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM python:3.10-slim

WORKDIR /app

# Copy only what's needed from builder
COPY --from=builder /root/.local /root/.local
COPY . .

# Make sure scripts are in PATH
ENV PATH=/root/.local/bin:$PATH

CMD ["python", "app.py"]
```

### 2. Layer Optimization

```dockerfile
# ❌ Bad: Creates unnecessary layers
FROM ubuntu:22.04
RUN apt-get update
RUN apt-get install -y python3
RUN apt-get install -y python3-pip
RUN apt-get clean

# ✅ Good: Minimizes layers
FROM ubuntu:22.04
RUN apt-get update && \
    apt-get install -y \
        python3 \
        python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
```

### 3. Caching Optimization

```dockerfile
# ❌ Bad: Changes to code invalidate dependency cache
FROM python:3.10
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt

# ✅ Good: Dependencies cached separately
FROM python:3.10
WORKDIR /app

# Copy and install dependencies first (cached)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code (changes frequently)
COPY . .
```

### 4. Security Best Practices

```dockerfile
FROM python:3.10-slim

# Create non-root user
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install dependencies as root
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY --chown=appuser:appuser . .

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

CMD ["python", "app.py"]
```

## Docker Compose

### Basic Structure

```yaml
version: '3.8'

services:
  # ML API Service
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/models/model.pt
      - LOG_LEVEL=info
    volumes:
      - ./models:/models
    depends_on:
      - redis
      - postgres
    restart: unless-stopped

  # Redis Cache
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data

  # PostgreSQL Database
  postgres:
    image: postgres:15
    environment:
      POSTGRES_USER: mluser
      POSTGRES_PASSWORD: mlpassword
      POSTGRES_DB: mldb
    volumes:
      - postgres-data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  # Prometheus Monitoring
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'

volumes:
  redis-data:
  postgres-data:
  prometheus-data:
```

### Common Commands

```bash
# Start services
docker-compose up -d

# Stop services
docker-compose down

# View logs
docker-compose logs -f api

# Scale services
docker-compose up -d --scale api=3

# Rebuild and start
docker-compose up -d --build

# View status
docker-compose ps

# Execute command
docker-compose exec api bash
```

## GPU Support in Docker

### Dockerfile with GPU

```dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Install Python
RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118

WORKDIR /app
COPY . .

CMD ["python3", "train.py"]
```

### Running with GPU

```bash
# Single GPU
docker run --gpus all my-ml-app

# Specific GPU
docker run --gpus device=0 my-ml-app

# Multiple GPUs
docker run --gpus '"device=0,1"' my-ml-app

# Docker Compose with GPU
version: '3.8'

services:
  ml-training:
    image: my-ml-app
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

## Performance Optimization

### 1. Image Size Optimization

```bash
# Use smaller base images
FROM python:3.10-slim      # ~150MB
# vs
FROM python:3.10           # ~900MB

# Use Alpine for minimal size
FROM python:3.10-alpine    # ~50MB

# Multi-stage builds
# Final image only contains runtime dependencies
```

### 2. Build Cache Optimization

```dockerfile
# Use BuildKit for better caching
# DOCKER_BUILDKIT=1 docker build .

# Use cache mounts
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt
```

### 3. Layer Optimization

```bash
# View layer sizes
docker history my-app:latest

# Analyze with dive
dive my-app:latest
```

## Container Networking Patterns

### 1. Bridge Network (Default)

```bash
# Create custom bridge network
docker network create ml-network

# Run containers on network
docker run -d --name api --network ml-network my-api
docker run -d --name db --network ml-network postgres

# Containers can communicate via service names
# api can reach db at hostname "db"
```

### 2. Host Network

```bash
# Use host network (better performance)
docker run --network host my-app

# No port mapping needed
# Container uses host's network stack directly
```

### 3. Service Discovery

```yaml
# docker-compose.yml enables automatic service discovery
services:
  api:
    build: .
    environment:
      - DATABASE_URL=postgresql://postgres:5432/db
      # Can use service name "postgres" as hostname!

  postgres:
    image: postgres:15
```

## Data Persistence

### Volume Types

```bash
# Named volumes (managed by Docker)
docker volume create ml-data
docker run -v ml-data:/data my-app

# Bind mounts (host directory)
docker run -v /host/path:/container/path my-app

# tmpfs (in-memory, fast)
docker run --tmpfs /tmp my-app
```

### Backup and Restore

```bash
# Backup volume
docker run --rm \
    -v ml-data:/data \
    -v $(pwd):/backup \
    ubuntu tar czf /backup/data-backup.tar.gz -C /data .

# Restore volume
docker run --rm \
    -v ml-data:/data \
    -v $(pwd):/backup \
    ubuntu tar xzf /backup/data-backup.tar.gz -C /data
```

## Security Best Practices

### 1. Image Security

```bash
# Scan images for vulnerabilities
docker scan my-app:latest

# Use Trivy
trivy image my-app:latest

# Use specific image versions (not :latest)
FROM python:3.10.12-slim

# Verify image signatures
docker trust sign my-app:v1.0
```

### 2. Runtime Security

```dockerfile
# Run as non-root user
USER 1000:1000

# Read-only filesystem
docker run --read-only my-app

# Drop capabilities
docker run --cap-drop=ALL --cap-add=NET_BIND_SERVICE my-app

# Use security options
docker run --security-opt=no-new-privileges my-app
```

### 3. Secrets Management

```bash
# Docker secrets (Swarm mode)
docker secret create db_password password.txt
docker service create --secret db_password my-app

# Environment variables from file
docker run --env-file .env my-app

# Use external secret managers
# - AWS Secrets Manager
# - HashiCorp Vault
# - Kubernetes Secrets
```

## Monitoring and Logging

### Container Logging

```bash
# View logs
docker logs my-container

# Follow logs
docker logs -f my-container

# Limit log output
docker logs --tail 100 my-container

# Configure log driver
docker run --log-driver=json-file \
    --log-opt max-size=10m \
    --log-opt max-file=3 \
    my-app
```

### Monitoring with cAdvisor

```yaml
services:
  cadvisor:
    image: gcr.io/cadvisor/cadvisor:latest
    ports:
      - "8080:8080"
    volumes:
      - /:/rootfs:ro
      - /var/run:/var/run:ro
      - /sys:/sys:ro
      - /var/lib/docker:/var/lib/docker:ro
```

## Resources

- [Docker Documentation](https://docs.docker.com/)
- [Dockerfile Best Practices](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)
- [Container Security](https://cheatsheetseries.owasp.org/cheatsheets/Docker_Security_Cheat_Sheet.html)

## Next Steps

After completing this module:
1. Practice building Docker images for ML applications
2. Deploy multi-container ML systems with Docker Compose
3. Learn Kubernetes for container orchestration
4. Implement CI/CD pipelines with Docker
5. Explore advanced topics: service mesh, serverless containers

---

**Master containerization for ML infrastructure! 🐳**
