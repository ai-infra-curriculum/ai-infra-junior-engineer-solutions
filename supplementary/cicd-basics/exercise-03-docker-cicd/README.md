# Exercise 03: Docker Image CI/CD

## Overview

Learn to containerize ML applications and automate Docker image building, testing, and publishing through CI/CD pipelines. This exercise covers Dockerfiles, multi-stage builds, image optimization, and automated container workflows.

## Learning Objectives

- Write Dockerfiles for ML applications
- Implement multi-stage builds for optimization
- Build and test Docker images locally
- Set up automated image building in CI/CD
- Publish images to container registries
- Implement security scanning for images
- Optimize image size and build times
- Use Docker Compose for multi-container apps

## Prerequisites

- Docker installed (version 20.10+)
- Docker Compose installed
- GitHub account (for GitHub Container Registry)
- Completed Exercise 01 (Git Workflows)
- Basic understanding of containers

## Project Structure

```
exercise-03-docker-cicd/
├── dockerfiles/
│   ├── Dockerfile.basic              # Simple Dockerfile
│   ├── Dockerfile.optimized          # Optimized multi-stage build
│   ├── Dockerfile.gpu                # GPU-enabled image
│   ├── Dockerfile.dev                # Development image
│   └── .dockerignore                 # Files to exclude
├── app/
│   ├── main.py                       # ML API application
│   ├── model.py                      # Model loading/inference
│   ├── requirements.txt              # Python dependencies
│   └── config.py                     # Configuration
├── scripts/
│   ├── build.sh                      # Build images
│   ├── test-image.sh                 # Test Docker images
│   ├── push.sh                       # Push to registry
│   └── run-local.sh                  # Run containers locally
├── .github/
│   └── workflows/
│       ├── docker-build.yml          # Build on push
│       ├── docker-publish.yml        # Publish on release
│       └── docker-scan.yml           # Security scanning
├── docker-compose.yml                # Multi-container setup
├── docker-compose.dev.yml            # Development setup
├── docs/
│   ├── DOCKERFILE_GUIDE.md          # Dockerfile best practices
│   ├── REGISTRY_GUIDE.md            # Working with registries
│   └── OPTIMIZATION.md              # Image optimization tips
└── README.md                         # This file
```

## Docker Basics

### What is Docker?

Docker is a platform for developing, shipping, and running applications in containers. Containers package an application with all its dependencies, ensuring consistency across environments.

**Key Concepts:**
- **Image**: Read-only template with application code and dependencies
- **Container**: Running instance of an image
- **Dockerfile**: Instructions to build an image
- **Registry**: Storage for Docker images (Docker Hub, GitHub Container Registry, etc.)
- **Layer**: Each instruction in Dockerfile creates a layer (cached for efficiency)

### Why Docker for ML?

✅ **Reproducibility**: Same environment everywhere
✅ **Dependency Management**: All dependencies packaged together
✅ **Isolation**: No conflicts with system packages
✅ **Scalability**: Easy to deploy multiple instances
✅ **Portability**: Run anywhere Docker is supported
✅ **CI/CD Integration**: Automate building and deployment

## Dockerfile Fundamentals

### Basic Structure

```dockerfile
# Base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run application
CMD ["python", "main.py"]
```

### Common Instructions

| Instruction | Purpose | Example |
|------------|---------|---------|
| `FROM` | Base image | `FROM python:3.11-slim` |
| `WORKDIR` | Set working directory | `WORKDIR /app` |
| `COPY` | Copy files from host | `COPY . /app` |
| `ADD` | Copy + extract archives | `ADD model.tar.gz /models` |
| `RUN` | Execute commands | `RUN pip install numpy` |
| `ENV` | Set environment variables | `ENV MODEL_PATH=/models` |
| `EXPOSE` | Document port | `EXPOSE 8000` |
| `CMD` | Default command | `CMD ["python", "app.py"]` |
| `ENTRYPOINT` | Fixed command | `ENTRYPOINT ["python"]` |
| `ARG` | Build-time variables | `ARG VERSION=1.0` |
| `LABEL` | Metadata | `LABEL maintainer="you@example.com"` |
| `USER` | Set user | `USER appuser` |
| `VOLUME` | Create mount point | `VOLUME /data` |

## Multi-Stage Builds

Multi-stage builds reduce image size by separating build and runtime environments.

### Example: ML Application

```dockerfile
# Stage 1: Builder
FROM python:3.11 AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim

WORKDIR /app

# Copy only Python packages from builder
COPY --from=builder /root/.local /root/.local

# Copy application
COPY app/ .

# Update PATH
ENV PATH=/root/.local/bin:$PATH

# Run application
CMD ["python", "main.py"]
```

**Benefits:**
- Smaller final image (no build tools)
- Faster deployment (less data to transfer)
- More secure (fewer attack surfaces)

## Docker Best Practices for ML

### 1. Use Appropriate Base Images

```dockerfile
# Production: Slim images
FROM python:3.11-slim

# GPU support
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Minimal
FROM python:3.11-alpine  # Be careful with ML libs
```

### 2. Optimize Layer Caching

```dockerfile
# ✅ Good: Dependencies cached separately
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .

# ❌ Bad: Changes in code invalidate dependency cache
COPY . .
RUN pip install -r requirements.txt
```

### 3. Minimize Layers

```dockerfile
# ✅ Good: Single layer for related commands
RUN apt-get update && \
    apt-get install -y pkg1 pkg2 && \
    rm -rf /var/lib/apt/lists/*

# ❌ Bad: Multiple layers
RUN apt-get update
RUN apt-get install -y pkg1
RUN apt-get install -y pkg2
```

### 4. Use .dockerignore

```
# .dockerignore
**/.git
**/__pycache__
**/*.pyc
**/.pytest_cache
**/venv
**/env
**/.env
**/tests
**/docs
**/*.md
.gitignore
```

### 5. Don't Run as Root

```dockerfile
# Create non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser
```

### 6. Health Checks

```dockerfile
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
```

### 7. Use Build Arguments

```dockerfile
ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim

ARG MODEL_VERSION=1.0.0
ENV MODEL_VERSION=${MODEL_VERSION}
```

## ML-Specific Docker Patterns

### Pattern 1: Model as Artifact

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ .

# Download model during build (not recommended for large models)
ARG MODEL_URL
RUN wget -O /app/model.pkl ${MODEL_URL}

# Or mount model at runtime
VOLUME /models

CMD ["python", "serve.py"]
```

### Pattern 2: Model in Image

```dockerfile
# For small models (<100MB)
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ .
COPY models/ /app/models/  # Include model in image

CMD ["python", "serve.py"]
```

### Pattern 3: Model from Volume

```dockerfile
# For large models - mount at runtime
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ .

# Model loaded from mounted volume
VOLUME /models

ENV MODEL_PATH=/models/model.pkl

CMD ["python", "serve.py"]
```

## Building Images

### Basic Build

```bash
# Build with tag
docker build -t my-ml-app:latest -f dockerfiles/Dockerfile.basic .

# Build with build args
docker build \
    --build-arg PYTHON_VERSION=3.11 \
    --build-arg MODEL_VERSION=1.0.0 \
    -t my-ml-app:1.0.0 .

# Build for specific platform
docker build --platform linux/amd64 -t my-ml-app:latest .
```

### Multi-stage Build

```bash
docker build -f dockerfiles/Dockerfile.optimized -t my-ml-app:optimized .
```

### Build with Cache

```bash
# Use BuildKit for better caching
DOCKER_BUILDKIT=1 docker build -t my-ml-app:latest .

# Use cache from registry
docker build --cache-from my-registry/my-ml-app:latest -t my-ml-app:latest .
```

## Running Containers

### Basic Run

```bash
# Run container
docker run -p 8000:8000 my-ml-app:latest

# Run with environment variables
docker run -e MODEL_PATH=/models/model.pkl -p 8000:8000 my-ml-app:latest

# Run with volume mount
docker run -v $(pwd)/models:/models -p 8000:8000 my-ml-app:latest

# Run in detached mode
docker run -d --name ml-api -p 8000:8000 my-ml-app:latest

# Run with resource limits
docker run --memory="4g" --cpus="2" -p 8000:8000 my-ml-app:latest
```

### GPU Support

```bash
# Run with GPU
docker run --gpus all -p 8000:8000 my-ml-app:gpu

# Run with specific GPU
docker run --gpus '"device=0,1"' -p 8000:8000 my-ml-app:gpu
```

## Docker Compose

### Basic Configuration

```yaml
version: '3.8'

services:
  api:
    build:
      context: .
      dockerfile: dockerfiles/Dockerfile.optimized
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/models/model.pkl
    volumes:
      - ./models:/models
    restart: unless-stopped

  postgres:
    image: postgres:15
    environment:
      - POSTGRES_PASSWORD=secret
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

### Commands

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f api

# Stop services
docker-compose down

# Rebuild and start
docker-compose up --build

# Scale service
docker-compose up -d --scale api=3
```

## CI/CD for Docker Images

### GitHub Actions Workflow

```yaml
name: Build and Push Docker Image

on:
  push:
    branches: [ main ]
    tags: [ 'v*' ]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Log in to GitHub Container Registry
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ghcr.io/${{ github.repository }}
          tags: |
            type=ref,event=branch
            type=semver,pattern={{version}}
            type=semver,pattern={{major}}.{{minor}}
            type=sha

      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          context: .
          file: dockerfiles/Dockerfile.optimized
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

## Container Registries

### Docker Hub

```bash
# Login
docker login

# Tag image
docker tag my-ml-app:latest username/my-ml-app:latest

# Push
docker push username/my-ml-app:latest

# Pull
docker pull username/my-ml-app:latest
```

### GitHub Container Registry

```bash
# Login
echo $GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin

# Tag
docker tag my-ml-app:latest ghcr.io/username/my-ml-app:latest

# Push
docker push ghcr.io/username/my-ml-app:latest
```

### AWS ECR

```bash
# Login
aws ecr get-login-password --region us-east-1 | \
    docker login --username AWS --password-stdin 123456789.dkr.ecr.us-east-1.amazonaws.com

# Tag
docker tag my-ml-app:latest 123456789.dkr.ecr.us-east-1.amazonaws.com/my-ml-app:latest

# Push
docker push 123456789.dkr.ecr.us-east-1.amazonaws.com/my-ml-app:latest
```

## Security Best Practices

### 1. Scan for Vulnerabilities

```bash
# Using Trivy
trivy image my-ml-app:latest

# Using Docker Scout
docker scout cves my-ml-app:latest

# Using Snyk
snyk container test my-ml-app:latest
```

### 2. Use Official Base Images

```dockerfile
# ✅ Good: Official Python image
FROM python:3.11-slim

# ❌ Bad: Unknown source
FROM randomuser/python:latest
```

### 3. Keep Images Updated

```bash
# Regularly rebuild with latest base
docker build --pull -t my-ml-app:latest .
```

### 4. Don't Store Secrets in Images

```dockerfile
# ❌ Bad: Secrets in image
ENV API_KEY=secret123

# ✅ Good: Pass at runtime
# docker run -e API_KEY=$API_KEY my-ml-app:latest
```

### 5. Use Content Trust

```bash
# Enable content trust
export DOCKER_CONTENT_TRUST=1

# Sign and push
docker push my-ml-app:latest
```

## Image Optimization

### Size Optimization

**Techniques:**
1. Use slim/alpine base images
2. Multi-stage builds
3. Remove unnecessary files
4. Combine RUN commands
5. Use .dockerignore
6. Clean package manager caches

**Example:**

```dockerfile
FROM python:3.11-slim

# Install dependencies efficiently
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir numpy scikit-learn

# Result: Much smaller image
```

### Build Time Optimization

**Techniques:**
1. Order instructions by frequency of change
2. Use build cache effectively
3. Use BuildKit
4. Parallelize with multi-stage builds
5. Use layer caching from registry

## Common Commands Reference

```bash
# Build
docker build -t image:tag .

# Run
docker run -p 8000:8000 image:tag

# List images
docker images

# Remove image
docker rmi image:tag

# List containers
docker ps -a

# Stop container
docker stop container_id

# Remove container
docker rm container_id

# View logs
docker logs container_id

# Execute command in container
docker exec -it container_id bash

# Inspect
docker inspect container_id

# Stats
docker stats

# Clean up
docker system prune -a
```

## Exercises

### Exercise 1: Build Basic Image
Create a Dockerfile for the sample ML API and build it.

### Exercise 2: Optimize Image Size
Use multi-stage build to reduce image size by >50%.

### Exercise 3: Add Health Checks
Implement health check endpoint and Dockerfile HEALTHCHECK.

### Exercise 4: Set Up Docker Compose
Create docker-compose.yml for API + database.

### Exercise 5: Automate CI/CD
Create GitHub Actions workflow to build and push images.

### Exercise 6: Implement Security Scanning
Add vulnerability scanning to CI pipeline.

## Resources

- [Docker Documentation](https://docs.docker.com/)
- [Dockerfile Best Practices](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/)
- [Docker for ML](https://madewithml.com/courses/mlops/docker/)
- [Multi-stage Builds](https://docs.docker.com/build/building/multi-stage/)
- [GitHub Actions Docker](https://docs.github.com/en/actions/publishing-packages/publishing-docker-images)

## Next Steps

After completing this exercise:

1. ✅ Understand Docker fundamentals
2. ✅ Write production-ready Dockerfiles
3. ✅ Optimize image size and build time
4. ✅ Implement automated image building
5. ✅ Publish images to registries

**Move on to**: Exercise 04 - Kubernetes Deployment Automation
