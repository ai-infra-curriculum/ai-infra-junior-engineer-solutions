# Exercise 01: Docker Fundamentals

## Overview

Master Docker fundamentals including container lifecycle management, image operations, Docker registries, and essential commands for ML infrastructure work.

## Learning Objectives

- ✅ Understand Docker architecture and concepts
- ✅ Manage container lifecycle effectively
- ✅ Build and manage Docker images
- ✅ Work with Docker registries
- ✅ Use Docker CLI efficiently
- ✅ Debug containerized applications
- ✅ Implement basic containerization patterns

## Topics Covered

### 1. Docker Architecture

**Components:**
- Docker Client (CLI)
- Docker Daemon (dockerd)
- Container Runtime (containerd/runc)
- Docker Registry
- Images and Containers

**Key Concepts:**
- Images: Read-only templates
- Containers: Running instances of images
- Layers: Incremental filesystem changes
- Registry: Storage for images

### 2. Container Lifecycle

```bash
# Create container (doesn't start it)
docker create --name my-container ubuntu:22.04

# Start container
docker start my-container

# Run (create + start)
docker run ubuntu:22.04 echo "Hello Docker"

# Run interactively
docker run -it ubuntu:22.04 bash

# Run in background (detached)
docker run -d nginx:latest

# Stop container
docker stop my-container

# Kill container (force stop)
docker kill my-container

# Remove container
docker rm my-container

# Remove running container
docker rm -f my-container
```

### 3. Essential Docker Commands

#### Container Management

```bash
# List running containers
docker ps

# List all containers
docker ps -a

# Container details
docker inspect my-container

# Container logs
docker logs my-container
docker logs -f my-container  # Follow

# Execute command in container
docker exec my-container ls /app
docker exec -it my-container bash

# Copy files
docker cp my-container:/app/file.txt ./
docker cp ./file.txt my-container:/app/

# View resource usage
docker stats my-container

# View processes
docker top my-container

# Attach to container
docker attach my-container

# Pause/unpause container
docker pause my-container
docker unpause my-container

# Restart container
docker restart my-container
```

#### Image Management

```bash
# Pull image
docker pull ubuntu:22.04

# List images
docker images
docker image ls

# Build image
docker build -t my-app:v1.0 .

# Tag image
docker tag my-app:v1.0 registry.com/my-app:v1.0

# Push to registry
docker push registry.com/my-app:v1.0

# Remove image
docker rmi my-app:v1.0

# Remove unused images
docker image prune

# Save image to file
docker save my-app:v1.0 -o my-app.tar

# Load image from file
docker load -i my-app.tar

# Image history
docker history my-app:v1.0

# Image inspection
docker image inspect my-app:v1.0
```

### 4. Running Containers with Options

```bash
# Name container
docker run --name web-server nginx

# Port mapping
docker run -p 8080:80 nginx  # Host:Container

# Environment variables
docker run -e API_KEY=secret my-app

# Volume mounting
docker run -v /host/path:/container/path my-app

# Working directory
docker run -w /app my-app

# User
docker run --user 1000:1000 my-app

# Memory limit
docker run -m 512m my-app

# CPU limit
docker run --cpus=2 my-app

# Restart policy
docker run --restart unless-stopped my-app

# Network
docker run --network my-network my-app

# Hostname
docker run --hostname my-host my-app

# Add host entry
docker run --add-host api.local:192.168.1.10 my-app
```

### 5. Building Docker Images

#### Basic Dockerfile

```dockerfile
# Base image
FROM python:3.10-slim

# Metadata
LABEL maintainer="your-email@example.com"
LABEL version="1.0"

# Set working directory
WORKDIR /app

# Copy files
COPY requirements.txt .
COPY app.py .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 8000

# Set environment variable
ENV APP_ENV=production

# Default command
CMD ["python", "app.py"]
```

#### Build Commands

```bash
# Build image
docker build -t my-app:v1.0 .

# Build with custom Dockerfile
docker build -f Dockerfile.dev -t my-app:dev .

# Build with build arguments
docker build --build-arg VERSION=1.0 -t my-app .

# Build without cache
docker build --no-cache -t my-app .

# Build with progress output
docker build --progress=plain -t my-app .

# Build specific stage
docker build --target builder -t my-app:builder .
```

### 6. Docker Registry Operations

#### Docker Hub

```bash
# Login to Docker Hub
docker login

# Tag for Docker Hub
docker tag my-app:v1.0 username/my-app:v1.0

# Push to Docker Hub
docker push username/my-app:v1.0

# Pull from Docker Hub
docker pull username/my-app:v1.0
```

#### Private Registry

```bash
# Login to private registry
docker login registry.company.com

# Tag for private registry
docker tag my-app:v1.0 registry.company.com/my-app:v1.0

# Push to private registry
docker push registry.company.com/my-app:v1.0

# Pull from private registry
docker pull registry.company.com/my-app:v1.0
```

### 7. Debugging Containers

```bash
# View container logs
docker logs container-name

# Follow logs in real-time
docker logs -f container-name

# Show last N lines
docker logs --tail 100 container-name

# Show logs with timestamps
docker logs -t container-name

# Inspect container
docker inspect container-name

# Check container processes
docker top container-name

# Monitor resource usage
docker stats container-name

# Access container shell
docker exec -it container-name bash

# View container filesystem changes
docker diff container-name

# Export container filesystem
docker export container-name > container.tar
```

### 8. System Management

```bash
# View Docker info
docker info

# Check Docker version
docker version

# View system-wide information
docker system df

# Clean up unused resources
docker system prune

# Clean up everything
docker system prune -a --volumes

# View events
docker events

# View daemon logs
journalctl -u docker
```

---

## Practical Examples

### Example 1: Simple Web Server

```bash
# Run nginx web server
docker run -d \
    --name web-server \
    -p 8080:80 \
    nginx:latest

# Test
curl http://localhost:8080

# View logs
docker logs web-server

# Stop and remove
docker stop web-server
docker rm web-server
```

### Example 2: Python Application

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

CMD ["python", "app.py"]
```

```bash
# Build
docker build -t python-app:v1.0 .

# Run
docker run -d \
    --name python-app \
    -p 8000:8000 \
    -e DEBUG=true \
    python-app:v1.0

# Check logs
docker logs -f python-app
```

### Example 3: Data Science Container

```bash
# Run Jupyter notebook
docker run -d \
    --name jupyter \
    -p 8888:8888 \
    -v $(pwd)/notebooks:/notebooks \
    jupyter/scipy-notebook

# Get token
docker logs jupyter | grep token
```

---

## Project: Docker Management Tool

Build a comprehensive Docker management tool that automates common operations.

### Requirements

**Features:**
1. Container lifecycle management
2. Image operations
3. Automated cleanup
4. Health monitoring
5. Log aggregation
6. Resource reporting

### Implementation

See `solutions/` directory for:
1. **`docker_manager.sh`** - Container management automation
2. **`image_cleaner.sh`** - Automated image cleanup
3. **`container_monitor.sh`** - Container health monitoring
4. **`registry_sync.sh`** - Registry synchronization tool

---

## Practice Problems

### Problem 1: Container Inspector

Create a script that:
- Lists all containers with detailed info
- Shows resource usage per container
- Identifies containers with issues
- Generates health report
- Provides cleanup recommendations

### Problem 2: Image Analyzer

Build a tool that:
- Analyzes Docker images for size
- Lists layers and their sizes
- Identifies optimization opportunities
- Compares image versions
- Generates optimization report

### Problem 3: Container Lifecycle Manager

Implement a manager that:
- Handles container creation with templates
- Manages container lifecycle
- Implements health checks
- Handles automatic restarts
- Provides status dashboard

---

## Best Practices

### 1. Container Naming

```bash
# Use descriptive names
docker run --name ml-api-v1 my-api

# Include environment
docker run --name ml-api-prod my-api

# Use labels for organization
docker run --label env=prod --label team=ml my-api
```

### 2. Resource Limits

```bash
# Set memory limits
docker run -m 512m my-app

# Set CPU limits
docker run --cpus=1.5 my-app

# Set all limits
docker run \
    -m 1g \
    --memory-swap 2g \
    --cpus=2 \
    --cpu-shares=1024 \
    my-app
```

### 3. Logging Configuration

```bash
# Configure log driver
docker run \
    --log-driver json-file \
    --log-opt max-size=10m \
    --log-opt max-file=3 \
    my-app
```

### 4. Health Checks

```dockerfile
# In Dockerfile
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
```

```bash
# View health status
docker inspect --format='{{.State.Health.Status}}' container-name
```

---

## Common Issues and Solutions

### Issue 1: Permission Denied

```bash
# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker
```

### Issue 2: Port Already in Use

```bash
# Find process using port
sudo lsof -i :8080

# Use different port
docker run -p 8081:8080 my-app
```

### Issue 3: Container Won't Stop

```bash
# Force stop
docker kill container-name

# Check for zombie containers
docker ps -a
```

### Issue 4: Out of Disk Space

```bash
# Clean up
docker system prune -a --volumes

# Check disk usage
docker system df
```

---

## Validation

Test your Docker skills:

```bash
# 1. Pull and run container
docker pull nginx:latest
docker run -d -p 8080:80 --name web nginx:latest
curl http://localhost:8080

# 2. Build custom image
cat > Dockerfile <<'EOF'
FROM ubuntu:22.04
RUN apt-get update && apt-get install -y curl
CMD ["curl", "--version"]
EOF

docker build -t test-image .
docker run test-image

# 3. Use management scripts
./solutions/docker_manager.sh --list
./solutions/image_cleaner.sh --dry-run
./solutions/container_monitor.sh --once
```

---

## Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker CLI Reference](https://docs.docker.com/engine/reference/commandline/cli/)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Play with Docker](https://labs.play-with-docker.com/)

---

## Next Steps

1. **Exercise 02: Building ML Docker Images** - Optimize for ML workloads
2. Master Docker Compose for multi-container apps
3. Learn advanced Dockerfile techniques
4. Explore container orchestration
5. Study Docker security practices

---

**Master Docker fundamentals for ML infrastructure! 🐳**
