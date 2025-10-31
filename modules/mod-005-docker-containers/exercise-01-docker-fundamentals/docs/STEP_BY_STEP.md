# Step-by-Step Implementation Guide: Docker Fundamentals

## Overview

Master Docker fundamentals for containerizing ML applications. Learn Docker architecture, essential commands, container lifecycle management, image building, registries, and troubleshooting for ML infrastructure.

**Time**: 3-4 hours | **Difficulty**: Beginner to Intermediate

---

## Learning Objectives

✅ Understand Docker architecture and components
✅ Master essential Docker commands
✅ Manage container lifecycle (create, start, stop, remove)
✅ Build and tag Docker images
✅ Work with Docker registries (Docker Hub, private registries)
✅ Inspect and debug containers
✅ Optimize images for ML workloads

---

## Prerequisites

```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add user to docker group (avoid sudo)
sudo usermod -aG docker $USER
newgrp docker

# Verify installation
docker --version
docker run hello-world

# Check Docker info
docker info
docker version
```

---

## Phase 1: Essential Docker Commands (60 minutes)

### Container Management

```bash
# Run container
docker run ubuntu:22.04 echo "Hello Docker"

# Run interactive container
docker run -it ubuntu:22.04 /bin/bash

# Run in background (detached)
docker run -d nginx:latest

# Run with name
docker run --name my-nginx -d nginx:latest

# Run with port mapping
docker run -d -p 8080:80 nginx:latest

# Run with environment variables
docker run -e ML_ENV=production -d my-ml-app

# Run with resource limits
docker run -m 2g --cpus 2 my-ml-app

# List running containers
docker ps

# List all containers (including stopped)
docker ps -a

# Stop container
docker stop container_id

# Start stopped container
docker start container_id

# Restart container
docker restart container_id

# Remove container
docker rm container_id

# Remove running container (force)
docker rm -f container_id

# Remove all stopped containers
docker container prune
```

### Image Management

```bash
# List images
docker images

# Pull image from Docker Hub
docker pull python:3.10-slim

# Pull specific version
docker pull pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# Tag image
docker tag my-image:latest myrepo/my-image:v1.0

# Push to registry
docker push myrepo/my-image:v1.0

# Remove image
docker rmi image_id

# Remove unused images
docker image prune

# Remove all unused images
docker image prune -a

# Inspect image
docker inspect python:3.10

# View image history
docker history python:3.10
```

---

## Phase 2: Building ML Docker Images (60 minutes)

### Basic Dockerfile for ML

```dockerfile
# Dockerfile for ML Application
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/app/models

# Health check
HEALTHCHECK --interval=30s --timeout=3s \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["python", "app.py"]
```

### Building and Running

```bash
# Build image
docker build -t ml-app:latest .

# Build with build args
docker build --build-arg PYTHON_VERSION=3.10 -t ml-app:latest .

# Build with no cache
docker build --no-cache -t ml-app:latest .

# Run the image
docker run -d -p 8000:8000 --name ml-service ml-app:latest

# View logs
docker logs ml-service

# Follow logs
docker logs -f ml-service

# View last 100 lines
docker logs --tail 100 ml-service
```

---

## Phase 3: Container Inspection & Debugging (45 minutes)

### Inspection Commands

```bash
# Execute command in running container
docker exec ml-service ls /app

# Interactive shell in running container
docker exec -it ml-service /bin/bash

# Inspect container details
docker inspect ml-service

# View container stats
docker stats ml-service

# View resource usage of all containers
docker stats

# View processes in container
docker top ml-service

# Copy files from container
docker cp ml-service:/app/model.pkl ./local/

# Copy files to container
docker cp ./data.csv ml-service:/app/data/
```

### Debugging

```bash
# View container logs
docker logs ml-service

# Check exit code of stopped container
docker inspect --format='{{.State.ExitCode}}' ml-service

# View recent events
docker events --since '10m'

# Pause/unpause container
docker pause ml-service
docker unpause ml-service

# Commit container to image (for debugging)
docker commit ml-service debug-image:latest
```

---

## Phase 4: Working with Registries (30 minutes)

### Docker Hub

```bash
# Login to Docker Hub
docker login

# Tag for Docker Hub
docker tag ml-app:latest username/ml-app:v1.0

# Push to Docker Hub
docker push username/ml-app:v1.0

# Pull from Docker Hub
docker pull username/ml-app:v1.0
```

### Private Registry

```bash
# Run local registry
docker run -d -p 5000:5000 --name registry registry:2

# Tag for private registry
docker tag ml-app:latest localhost:5000/ml-app:v1.0

# Push to private registry
docker push localhost:5000/ml-app:v1.0

# Pull from private registry
docker pull localhost:5000/ml-app:v1.0
```

---

## Phase 5: Best Practices for ML Images (30 minutes)

### Multi-stage Builds

```dockerfile
# Multi-stage build for smaller images
FROM python:3.10 AS builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Final stage
FROM python:3.10-slim

WORKDIR /app

# Copy only necessary files from builder
COPY --from=builder /root/.local /root/.local
COPY . .

ENV PATH=/root/.local/bin:$PATH

CMD ["python", "app.py"]
```

### Layer Optimization

```dockerfile
# Bad: Changes to code rebuild everything
COPY . .
RUN pip install -r requirements.txt

# Good: Dependencies cached separately
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
```

### .dockerignore

```
# .dockerignore
__pycache__
*.pyc
*.pyo
.git
.gitignore
.env
.venv
venv/
*.md
.DS_Store
.pytest_cache
*.log
models/*.pth  # Don't include large model files
data/
```

---

## Common Use Cases for ML

### 1. Jupyter Notebook Container

```bash
docker run -d \
    -p 8888:8888 \
    -v $(pwd):/workspace \
    --name jupyter \
    jupyter/scipy-notebook
```

### 2. PyTorch GPU Container

```bash
docker run --gpus all -it \
    -v $(pwd):/workspace \
    pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime \
    /bin/bash
```

### 3. TensorFlow Container

```bash
docker run --gpus all -d \
    -p 6006:6006 \
    -v $(pwd)/logs:/logs \
    tensorflow/tensorflow:2.12.0-gpu \
    tensorboard --logdir=/logs --host=0.0.0.0
```

---

## Troubleshooting

### Container Won't Start

```bash
# Check logs
docker logs container_name

# Inspect exit code
docker inspect --format='{{.State.ExitCode}}' container_name

# Common exit codes:
# 0: Success
# 1: Application error
# 137: Killed (OOM)
# 139: Segmentation fault
```

### Out of Disk Space

```bash
# Check disk usage
docker system df

# Clean up
docker system prune -a --volumes

# Remove specific volumes
docker volume prune
```

### Permission Denied

```bash
# Add user to docker group
sudo usermod -aG docker $USER

# Or run with sudo
sudo docker ps
```

---

## Best Practices

✅ Use official base images when possible
✅ Keep images small (use -slim or -alpine variants)
✅ Use multi-stage builds
✅ Don't run as root inside containers
✅ Use .dockerignore to exclude unnecessary files
✅ Tag images with versions, not just :latest
✅ Clean up regularly (docker system prune)
✅ Use health checks for production containers

---

## Quick Reference

```bash
# Lifecycle
docker run        # Create and start
docker start      # Start stopped
docker stop       # Stop running
docker restart    # Restart
docker rm         # Remove

# Information
docker ps         # List running
docker ps -a      # List all
docker logs       # View logs
docker inspect    # Details
docker stats      # Resource usage

# Images
docker images     # List
docker pull       # Download
docker build      # Create
docker push       # Upload
docker rmi        # Remove

# Exec
docker exec       # Run command
docker exec -it   # Interactive shell

# Cleanup
docker system prune    # Clean all
docker container prune # Remove stopped
docker image prune     # Remove unused images
```

---

**Docker fundamentals mastered!** 🐳
