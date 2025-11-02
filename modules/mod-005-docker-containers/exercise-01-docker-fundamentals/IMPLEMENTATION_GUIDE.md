# Implementation Guide: Docker Fundamentals for ML Infrastructure

## Overview

This comprehensive guide provides production-ready Docker container operations with a focus on ML infrastructure requirements. Learn to run, manage, and debug containers for AI/ML workloads including GPU support, resource management, and production best practices.

**Target Audience**: Junior AI Infrastructure Engineers
**Prerequisites**: Basic Linux command-line skills, Docker installed
**Estimated Time**: 4-6 hours
**Difficulty**: Beginner to Intermediate

---

## Table of Contents

1. [Docker Installation and Setup](#1-docker-installation-and-setup)
2. [Basic Container Operations](#2-basic-container-operations)
3. [Image Management](#3-image-management)
4. [Container Lifecycle Management](#4-container-lifecycle-management)
5. [Resource Limits and Constraints](#5-resource-limits-and-constraints)
6. [Container Logs and Debugging](#6-container-logs-and-debugging)
7. [Production Container Management](#7-production-container-management)
8. [ML-Specific Considerations](#8-ml-specific-considerations)

---

## 1. Docker Installation and Setup

### 1.1 Docker Installation (Ubuntu/Debian)

```bash
# Remove old versions
sudo apt-get remove docker docker-engine docker.io containerd runc

# Update package index
sudo apt-get update

# Install prerequisites
sudo apt-get install -y \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

# Add Docker's official GPG key
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
    sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# Set up stable repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker Engine
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io \
    docker-compose-plugin

# Verify installation
docker --version
sudo systemctl status docker
```

### 1.2 Post-Installation Configuration

```bash
# Add current user to docker group (avoid sudo)
sudo usermod -aG docker $USER

# Apply group changes (or logout/login)
newgrp docker

# Verify non-root access
docker run hello-world

# Configure Docker daemon
sudo mkdir -p /etc/docker

# Create daemon configuration
sudo tee /etc/docker/daemon.json > /dev/null <<EOF
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  },
  "default-ulimits": {
    "nofile": {
      "Name": "nofile",
      "Hard": 64000,
      "Soft": 64000
    }
  }
}
EOF

# Restart Docker to apply changes
sudo systemctl restart docker

# Verify Docker info
docker info
```

### 1.3 Docker System Verification

```bash
# Check Docker version
docker version

# View system-wide information
docker info | grep -E 'Server Version|Storage Driver|Logging Driver|Cgroup'

# Test basic functionality
docker run --rm alpine echo "Docker is working!"

# Check available resources
docker system df
```

**Expected Output:**
```
Images          0         0         0B        0B
Containers      0         0         0B        0B
Local Volumes   0         0         0B        0B
Build Cache     0         0         0B        0B
```

---

## 2. Basic Container Operations

### 2.1 Running Containers

#### Interactive Mode

```bash
# Run Ubuntu container with interactive terminal
docker run -it ubuntu:22.04 bash

# Inside container:
cat /etc/os-release
ls -la
exit

# Run with specific command
docker run --rm ubuntu:22.04 cat /etc/os-release

# Run Python container interactively
docker run -it --rm python:3.11-slim python
# >>> import sys
# >>> print(f"Python {sys.version}")
# >>> exit()
```

#### Detached Mode

```bash
# Run nginx in background
docker run -d --name web-server nginx:latest

# Verify container is running
docker ps

# Run with automatic cleanup on stop
docker run -d --rm --name temp-nginx nginx:latest

# Run with custom command
docker run -d --name custom-nginx nginx:latest \
    sh -c "echo 'Starting nginx' && nginx -g 'daemon off;'"
```

#### Container Naming and Identification

```bash
# Run with custom name
docker run -d --name ml-api nginx:latest

# Run without name (Docker assigns random name)
docker run -d nginx:latest

# List containers with custom format
docker ps --format "table {{.ID}}\t{{.Names}}\t{{.Status}}\t{{.Image}}"

# Get container ID by name
CONTAINER_ID=$(docker ps -qf "name=ml-api")
echo "Container ID: $CONTAINER_ID"
```

### 2.2 Port Mapping

```bash
# Map single port (host:container)
docker run -d -p 8080:80 --name web nginx:latest

# Test connection
curl http://localhost:8080

# Map multiple ports
docker run -d \
    -p 8080:80 \
    -p 8443:443 \
    --name web-multi \
    nginx:latest

# Use random host port (-P publishes all exposed ports)
docker run -d -P --name web-random nginx:latest

# Find assigned port
docker port web-random
# Output: 80/tcp -> 0.0.0.0:32768

# Bind to specific interface
docker run -d -p 127.0.0.1:8080:80 --name local-only nginx:latest

# ML API example: Flask app on port 5000
docker run -d \
    -p 5000:5000 \
    --name ml-flask-api \
    python:3.11-slim \
    sh -c "pip install flask && python app.py"
```

### 2.3 Environment Variables

```bash
# Single environment variable
docker run -d \
    -e MODEL_PATH=/models/my-model \
    --name ml-app \
    python:3.11-slim sleep 3600

# Multiple environment variables
docker run -d \
    -e DATABASE_URL=postgresql://user:pass@db:5432/mldb \
    -e MODEL_VERSION=v1.2.0 \
    -e BATCH_SIZE=32 \
    -e CUDA_VISIBLE_DEVICES=0 \
    --name ml-inference \
    tensorflow/tensorflow:latest-gpu

# Using environment file
cat > ml-config.env <<EOF
MODEL_NAME=resnet50
MODEL_VERSION=1.0.0
BATCH_SIZE=32
MAX_WORKERS=4
LOG_LEVEL=INFO
CUDA_VISIBLE_DEVICES=0,1
EOF

docker run -d \
    --env-file ml-config.env \
    --name ml-service \
    tensorflow/tensorflow:latest-gpu

# Verify environment variables
docker exec ml-service env | grep MODEL
```

### 2.4 Volume Mounts

```bash
# Mount host directory (for model files)
docker run -d \
    -v /path/to/models:/models:ro \
    -v /path/to/data:/data \
    --name ml-trainer \
    pytorch/pytorch:latest

# Named volume (persistent storage)
docker volume create ml-models
docker run -d \
    -v ml-models:/models \
    --name model-server \
    tensorflow/serving

# Temporary filesystem (tmpfs) for fast I/O
docker run -d \
    --tmpfs /tmp:rw,size=1g,mode=1777 \
    --name fast-processing \
    python:3.11-slim
```

---

## 3. Image Management

### 3.1 Pulling Images

```bash
# Pull latest version
docker pull nginx:latest

# Pull specific version
docker pull python:3.11-slim

# Pull ML framework images
docker pull tensorflow/tensorflow:2.13.0-gpu
docker pull pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
docker pull nvcr.io/nvidia/pytorch:23.08-py3

# Pull from private registry
docker login myregistry.com
docker pull myregistry.com/ml-models/inference:v1.0

# Pull all tags of an image
docker pull -a ubuntu

# Verify downloaded images
docker images
```

### 3.2 Tagging Images

```bash
# Tag for different environments
docker tag nginx:latest myapp:dev
docker tag nginx:latest myapp:staging
docker tag nginx:latest myapp:prod

# Tag with version
docker tag ml-model:latest ml-model:1.0.0
docker tag ml-model:latest ml-model:1.0
docker tag ml-model:latest ml-model:1

# Tag for registry
docker tag ml-model:latest myregistry.com/ml/model:1.0.0

# Tag with commit hash (CI/CD pattern)
GIT_HASH=$(git rev-parse --short HEAD)
docker tag ml-model:latest ml-model:${GIT_HASH}

# List images with specific name
docker images ml-model
```

### 3.3 Pushing Images

```bash
# Login to Docker Hub
docker login

# Push to Docker Hub
docker tag ml-model:latest username/ml-model:1.0.0
docker push username/ml-model:1.0.0

# Push to private registry
docker tag ml-model:latest registry.company.com/ml/model:1.0.0
docker push registry.company.com/ml/model:1.0.0

# Push all tags
docker push username/ml-model --all-tags

# Push to AWS ECR
aws ecr get-login-password --region us-east-1 | \
    docker login --username AWS --password-stdin \
    123456789012.dkr.ecr.us-east-1.amazonaws.com

docker tag ml-model:latest \
    123456789012.dkr.ecr.us-east-1.amazonaws.com/ml-model:1.0.0
docker push 123456789012.dkr.ecr.us-east-1.amazonaws.com/ml-model:1.0.0
```

### 3.4 Image Inspection and Management

```bash
# Inspect image details
docker image inspect nginx:latest

# Get image digest
docker images --digests nginx

# View image layers
docker history nginx:latest

# Remove image
docker rmi nginx:latest

# Remove unused images
docker image prune

# Remove all unused images (including tagged)
docker image prune -a

# List dangling images
docker images -f "dangling=true"

# Remove dangling images
docker images -f "dangling=true" -q | xargs docker rmi
```

---

## 4. Container Lifecycle Management

### 4.1 Starting and Stopping Containers

```bash
# Start container
docker start ml-api

# Stop container gracefully (SIGTERM, 10s grace period)
docker stop ml-api

# Stop with custom timeout
docker stop -t 30 ml-api

# Forcefully stop (SIGKILL)
docker kill ml-api

# Restart container
docker restart ml-api

# Pause container (freeze processes)
docker pause ml-api

# Unpause container
docker unpause ml-api
```

### 4.2 Container States

```bash
# List running containers
docker ps

# List all containers (including stopped)
docker ps -a

# Filter by status
docker ps -f "status=running"
docker ps -f "status=exited"
docker ps -f "status=paused"

# Filter by name
docker ps -f "name=ml-"

# Custom output format
docker ps --format "{{.ID}}: {{.Names}} - {{.Status}}"

# Get container IDs only
docker ps -q
```

### 4.3 Removing Containers

```bash
# Remove stopped container
docker rm ml-api

# Force remove running container
docker rm -f ml-api

# Remove multiple containers
docker rm container1 container2 container3

# Remove all stopped containers
docker container prune

# Remove containers older than 24h
docker container prune --filter "until=24h"

# Remove all containers (dangerous!)
docker stop $(docker ps -q)
docker rm $(docker ps -a -q)
```

### 4.4 Container Attach and Detach

```bash
# Attach to running container's main process
docker attach ml-api

# Detach without stopping: Ctrl+P, Ctrl+Q

# Attach to container output only (read-only)
docker attach --no-stdin ml-api

# Run container and attach automatically
docker run -it --name interactive-container ubuntu:22.04 bash
```

---

## 5. Resource Limits and Constraints

### 5.1 Memory Limits

```bash
# Set memory limit
docker run -d \
    --memory="2g" \
    --name ml-limited \
    tensorflow/tensorflow:latest-gpu

# Set memory with swap limit
docker run -d \
    --memory="2g" \
    --memory-swap="4g" \
    --name ml-with-swap \
    pytorch/pytorch:latest

# Disable swap
docker run -d \
    --memory="2g" \
    --memory-swap="2g" \
    --name ml-no-swap \
    pytorch/pytorch:latest

# Set memory reservation (soft limit)
docker run -d \
    --memory="2g" \
    --memory-reservation="1g" \
    --name ml-reserved \
    tensorflow/tensorflow:latest

# OOM kill disable (requires setting memory limit)
docker run -d \
    --memory="2g" \
    --oom-kill-disable \
    --name ml-no-oom \
    tensorflow/tensorflow:latest

# Verify memory settings
docker stats ml-limited --no-stream
docker inspect ml-limited | grep -A 10 Memory
```

### 5.2 CPU Limits

```bash
# Limit to 2 CPUs
docker run -d \
    --cpus="2.0" \
    --name ml-2cpu \
    tensorflow/tensorflow:latest

# CPU shares (relative weight, default 1024)
docker run -d \
    --cpu-shares=2048 \
    --name high-priority-ml \
    pytorch/pytorch:latest

docker run -d \
    --cpu-shares=512 \
    --name low-priority-ml \
    pytorch/pytorch:latest

# Pin to specific CPU cores
docker run -d \
    --cpuset-cpus="0,1" \
    --name ml-cores-0-1 \
    tensorflow/tensorflow:latest

# CPU quota (100000 = 1 CPU)
docker run -d \
    --cpu-period=100000 \
    --cpu-quota=50000 \
    --name ml-half-cpu \
    pytorch/pytorch:latest

# Verify CPU settings
docker stats ml-2cpu --no-stream
docker inspect ml-2cpu | grep -A 5 Cpu
```

### 5.3 GPU Access (NVIDIA)

```bash
# Enable all GPUs
docker run -d \
    --gpus all \
    --name ml-all-gpus \
    tensorflow/tensorflow:latest-gpu

# Specify GPU by ID
docker run -d \
    --gpus '"device=0"' \
    --name ml-gpu-0 \
    pytorch/pytorch:latest

# Multiple specific GPUs
docker run -d \
    --gpus '"device=0,1"' \
    --name ml-gpu-0-1 \
    tensorflow/tensorflow:latest-gpu

# Limit GPU count
docker run -d \
    --gpus 2 \
    --name ml-2-gpus \
    nvidia/cuda:11.8.0-runtime-ubuntu22.04

# GPU with resource limits
docker run -d \
    --gpus all \
    --memory="8g" \
    --cpus="4" \
    -e CUDA_VISIBLE_DEVICES=0 \
    --name ml-constrained \
    tensorflow/tensorflow:latest-gpu

# Verify GPU access
docker exec ml-all-gpus nvidia-smi
```

### 5.4 Disk I/O Limits

```bash
# Limit read/write IOPS
docker run -d \
    --device-read-iops=/dev/sda:100 \
    --device-write-iops=/dev/sda:100 \
    --name io-limited \
    ubuntu:22.04

# Limit read/write bandwidth (bytes per second)
docker run -d \
    --device-read-bps=/dev/sda:10mb \
    --device-write-bps=/dev/sda:10mb \
    --name bandwidth-limited \
    ubuntu:22.04

# Storage driver options
docker run -d \
    --storage-opt size=10G \
    --name size-limited \
    ubuntu:22.04
```

### 5.5 Testing Resource Limits

```bash
# Test memory limit with stress tool
docker run -d \
    --memory="100m" \
    --name stress-memory \
    progrium/stress --vm 1 --vm-bytes 150M --vm-hang 0

# Watch container get OOM killed
docker logs stress-memory
docker inspect stress-memory | grep OOMKilled

# Test CPU limit
docker run -d \
    --cpus="1.0" \
    --name stress-cpu \
    progrium/stress --cpu 4

# Monitor resource usage
docker stats stress-cpu

# Clean up stress tests
docker rm -f stress-memory stress-cpu
```

---

## 6. Container Logs and Debugging

### 6.1 Viewing Logs

```bash
# View all logs
docker logs ml-api

# Follow logs in real-time
docker logs -f ml-api

# Show timestamps
docker logs -t ml-api

# Show last N lines
docker logs --tail 100 ml-api

# Show logs since timestamp
docker logs --since 2024-01-01T10:00:00 ml-api

# Show logs for time range
docker logs --since 1h ml-api
docker logs --since 30m ml-api
docker logs --since 2024-01-01 --until 2024-01-02 ml-api

# Combine options
docker logs -f --tail 50 --timestamps ml-api
```

### 6.2 Log Drivers

```bash
# JSON file driver (default)
docker run -d \
    --log-driver json-file \
    --log-opt max-size=10m \
    --log-opt max-file=3 \
    --name ml-json-logs \
    tensorflow/tensorflow:latest

# Syslog driver
docker run -d \
    --log-driver syslog \
    --log-opt syslog-address=tcp://192.168.1.100:514 \
    --name ml-syslog \
    pytorch/pytorch:latest

# Disable logging (for performance)
docker run -d \
    --log-driver none \
    --name no-logs \
    nginx:latest

# Fluentd driver (for centralized logging)
docker run -d \
    --log-driver fluentd \
    --log-opt fluentd-address=localhost:24224 \
    --log-opt tag=ml.inference \
    --name ml-fluentd \
    tensorflow/tensorflow:latest

# View log driver configuration
docker inspect ml-json-logs | grep -A 10 LogConfig
```

### 6.3 Executing Commands for Debugging

```bash
# Execute command in running container
docker exec ml-api ls -la /app

# Interactive shell
docker exec -it ml-api bash

# Execute as specific user
docker exec -u root ml-api apt-get update

# Run command with environment variables
docker exec -e DEBUG=true ml-api python check.py

# Check process list
docker exec ml-api ps aux

# Check network connectivity
docker exec ml-api ping -c 3 google.com

# Check Python environment
docker exec ml-api python --version
docker exec ml-api pip list

# Check CUDA/GPU
docker exec ml-api nvidia-smi

# View logs in real-time
docker exec ml-api tail -f /var/log/app.log
```

### 6.4 Container Inspection

```bash
# Full container details
docker inspect ml-api

# Get specific field
docker inspect ml-api --format='{{.State.Status}}'
docker inspect ml-api --format='{{.NetworkSettings.IPAddress}}'
docker inspect ml-api --format='{{.Config.Image}}'

# Get multiple fields
docker inspect ml-api --format='{{.Name}} {{.State.Status}} {{.NetworkSettings.IPAddress}}'

# Get JSON output
docker inspect ml-api | jq '.[0].State'

# Get environment variables
docker inspect ml-api --format='{{json .Config.Env}}' | jq

# Get mounted volumes
docker inspect ml-api --format='{{json .Mounts}}' | jq

# Get port mappings
docker inspect ml-api --format='{{json .NetworkSettings.Ports}}' | jq
```

### 6.5 Process and Resource Monitoring

```bash
# View processes in container
docker top ml-api

# View with custom format
docker top ml-api aux

# Real-time resource statistics
docker stats ml-api

# One-shot statistics
docker stats --no-stream ml-api

# Monitor multiple containers
docker stats ml-api-1 ml-api-2 ml-api-3

# Custom format
docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"

# View filesystem changes
docker diff ml-api
```

### 6.6 Advanced Debugging

```bash
# View container events
docker events --filter 'container=ml-api'

# Export container filesystem
docker export ml-api -o ml-api-export.tar

# Create image from container
docker commit ml-api ml-api-debug:latest

# Copy files from container
docker cp ml-api:/app/logs ./logs
docker cp ml-api:/var/log/app.log ./app.log

# Copy files to container
docker cp ./config.json ml-api:/app/config.json

# Check container disk usage
docker system df
docker system df -v

# Inspect volumes
docker volume ls
docker volume inspect ml-models
```

---

## 7. Production Container Management

### 7.1 Health Checks

```bash
# Run with built-in health check
docker run -d \
    --name ml-api \
    --health-cmd='curl -f http://localhost:5000/health || exit 1' \
    --health-interval=30s \
    --health-timeout=10s \
    --health-retries=3 \
    --health-start-period=40s \
    ml-inference:latest

# Check health status
docker ps
docker inspect ml-api --format='{{.State.Health.Status}}'

# View health check logs
docker inspect ml-api --format='{{json .State.Health}}' | jq
```

### 7.2 Restart Policies

```bash
# No restart (default)
docker run -d --restart=no --name ml-no-restart nginx

# Always restart
docker run -d --restart=always --name ml-always nginx

# Restart on failure
docker run -d --restart=on-failure --name ml-on-failure nginx

# Restart on failure with max attempts
docker run -d --restart=on-failure:5 --name ml-retry nginx

# Unless stopped manually
docker run -d --restart=unless-stopped --name ml-unless-stopped nginx

# Update restart policy
docker update --restart=always ml-api
```

### 7.3 Signal Handling

```bash
# Stop with SIGTERM (graceful)
docker stop ml-api

# Stop with custom timeout
docker stop -t 60 ml-api

# Send specific signal
docker kill --signal=SIGHUP ml-api
docker kill --signal=SIGUSR1 ml-api

# Send SIGTERM then SIGKILL
docker stop ml-api  # SIGTERM, waits 10s, then SIGKILL
```

### 7.4 Container Updates

```bash
# Update resource limits
docker update \
    --memory="4g" \
    --cpus="2.0" \
    ml-api

# Update restart policy
docker update --restart=always ml-api

# Update multiple containers
docker update --memory="2g" ml-api-1 ml-api-2 ml-api-3

# Verify updates
docker inspect ml-api | grep -A 5 Memory
```

### 7.5 Production Deployment Script

```bash
#!/bin/bash
# deploy-ml-api.sh - Production ML API deployment

set -euo pipefail

# Configuration
IMAGE="ml-inference:1.0.0"
CONTAINER_NAME="ml-api-prod"
PORT=5000
HEALTH_URL="http://localhost:${PORT}/health"

# Color output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1"
    exit 1
}

# Pull latest image
log "Pulling image: ${IMAGE}"
docker pull ${IMAGE} || error "Failed to pull image"

# Stop and remove old container
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    log "Stopping old container"
    docker stop ${CONTAINER_NAME} || true
    docker rm ${CONTAINER_NAME} || true
fi

# Run new container
log "Starting new container"
docker run -d \
    --name ${CONTAINER_NAME} \
    --restart=unless-stopped \
    -p ${PORT}:${PORT} \
    --memory="4g" \
    --cpus="2.0" \
    --gpus all \
    -e MODEL_PATH=/models/production \
    -e LOG_LEVEL=INFO \
    -v /data/models:/models:ro \
    -v /data/logs:/logs \
    --health-cmd="curl -f ${HEALTH_URL} || exit 1" \
    --health-interval=30s \
    --health-timeout=10s \
    --health-retries=3 \
    --health-start-period=60s \
    --log-driver json-file \
    --log-opt max-size=10m \
    --log-opt max-file=3 \
    ${IMAGE} || error "Failed to start container"

# Wait for health check
log "Waiting for container to be healthy"
TIMEOUT=120
ELAPSED=0
while [ $ELAPSED -lt $TIMEOUT ]; do
    HEALTH=$(docker inspect ${CONTAINER_NAME} --format='{{.State.Health.Status}}' 2>/dev/null || echo "starting")

    if [ "$HEALTH" = "healthy" ]; then
        log "Container is healthy"
        break
    fi

    if [ "$HEALTH" = "unhealthy" ]; then
        error "Container is unhealthy"
    fi

    echo "Waiting for health check... ($ELAPSED/$TIMEOUT seconds)"
    sleep 5
    ELAPSED=$((ELAPSED + 5))
done

if [ $ELAPSED -ge $TIMEOUT ]; then
    error "Timeout waiting for container health"
fi

# Verify container is running
if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    error "Container is not running"
fi

log "Deployment successful"
log "Container: ${CONTAINER_NAME}"
log "Status: $(docker inspect ${CONTAINER_NAME} --format='{{.State.Status}}')"
log "Health: $(docker inspect ${CONTAINER_NAME} --format='{{.State.Health.Status}}')"

# Show logs
log "Recent logs:"
docker logs --tail 20 ${CONTAINER_NAME}
```

### 7.6 Monitoring and Alerting Script

```bash
#!/bin/bash
# monitor-containers.sh - Monitor container health and resources

ALERT_MEMORY_PERCENT=80
ALERT_CPU_PERCENT=80

check_container() {
    local container=$1

    # Check if running
    if ! docker ps --format '{{.Names}}' | grep -q "^${container}$"; then
        echo "ALERT: Container ${container} is not running"
        return 1
    fi

    # Check health status
    health=$(docker inspect ${container} --format='{{.State.Health.Status}}' 2>/dev/null || echo "none")
    if [ "$health" = "unhealthy" ]; then
        echo "ALERT: Container ${container} is unhealthy"
        return 1
    fi

    # Check resource usage
    stats=$(docker stats --no-stream --format "{{.MemPerc}},{{.CPUPerc}}" ${container})
    mem=$(echo $stats | cut -d, -f1 | sed 's/%//')
    cpu=$(echo $stats | cut -d, -f2 | sed 's/%//')

    mem_int=$(echo $mem | cut -d. -f1)
    cpu_int=$(echo $cpu | cut -d. -f1)

    if [ "$mem_int" -gt "$ALERT_MEMORY_PERCENT" ]; then
        echo "WARNING: Container ${container} high memory usage: ${mem}%"
    fi

    if [ "$cpu_int" -gt "$ALERT_CPU_PERCENT" ]; then
        echo "WARNING: Container ${container} high CPU usage: ${cpu}%"
    fi

    echo "OK: Container ${container} - Memory: ${mem}%, CPU: ${cpu}%, Health: ${health}"
    return 0
}

# Monitor all containers with prefix 'ml-'
for container in $(docker ps --format '{{.Names}}' | grep '^ml-'); do
    check_container "$container"
done
```

---

## 8. ML-Specific Considerations

### 8.1 Running ML Model Inference

```bash
# TensorFlow Serving
docker run -d \
    --name tf-serving \
    -p 8501:8501 \
    -p 8500:8500 \
    -v /path/to/models:/models \
    -e MODEL_NAME=my_model \
    --memory="4g" \
    --cpus="2.0" \
    tensorflow/serving:latest

# PyTorch TorchServe
docker run -d \
    --name torchserve \
    -p 8080:8080 \
    -p 8081:8081 \
    -v /path/to/models:/models \
    --memory="4g" \
    --gpus all \
    pytorch/torchserve:latest-gpu \
    torchserve --start --model-store=/models

# ONNX Runtime
docker run -d \
    --name onnx-runtime \
    -p 8001:8001 \
    -v /path/to/models:/models \
    --memory="2g" \
    mcr.microsoft.com/onnxruntime/server:latest
```

### 8.2 GPU Container Management

```bash
# Check NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# Run with specific GPU
docker run -d \
    --gpus '"device=0"' \
    --name ml-gpu-0 \
    -e CUDA_VISIBLE_DEVICES=0 \
    tensorflow/tensorflow:latest-gpu

# Monitor GPU usage
docker exec ml-gpu-0 nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv

# Set GPU memory growth (TensorFlow)
docker run -d \
    --gpus all \
    --name tf-gpu \
    -e TF_FORCE_GPU_ALLOW_GROWTH=true \
    tensorflow/tensorflow:latest-gpu

# Mixed precision training
docker run -d \
    --gpus all \
    --name mixed-precision \
    -e TF_ENABLE_AUTO_MIXED_PRECISION=1 \
    tensorflow/tensorflow:latest-gpu
```

### 8.3 Data Pipeline Containers

```bash
# Apache Airflow
docker run -d \
    --name airflow \
    -p 8080:8080 \
    -v /path/to/dags:/opt/airflow/dags \
    -v /path/to/logs:/opt/airflow/logs \
    -e AIRFLOW__CORE__EXECUTOR=LocalExecutor \
    apache/airflow:latest

# Apache Spark
docker run -d \
    --name spark-master \
    -p 8080:8080 \
    -p 7077:7077 \
    --memory="4g" \
    --cpus="2.0" \
    bitnami/spark:latest

# Jupyter Notebook for ML development
docker run -d \
    --name jupyter-ml \
    -p 8888:8888 \
    -v /path/to/notebooks:/home/jovyan/work \
    --gpus all \
    --memory="8g" \
    jupyter/tensorflow-notebook:latest
```

### 8.4 ML Database Containers

```bash
# Vector database (Milvus)
docker run -d \
    --name milvus \
    -p 19530:19530 \
    -p 9091:9091 \
    -v /path/to/milvus:/var/lib/milvus \
    --memory="8g" \
    milvusdb/milvus:latest

# PostgreSQL with pgvector
docker run -d \
    --name postgres-vector \
    -p 5432:5432 \
    -e POSTGRES_PASSWORD=secret \
    -v /path/to/data:/var/lib/postgresql/data \
    --memory="4g" \
    ankane/pgvector:latest

# Redis for caching
docker run -d \
    --name redis-cache \
    -p 6379:6379 \
    -v /path/to/redis:/data \
    --memory="2g" \
    redis:7-alpine
```

### 8.5 Complete ML Stack Example

```bash
#!/bin/bash
# setup-ml-stack.sh - Deploy complete ML infrastructure stack

set -e

echo "Setting up ML infrastructure stack..."

# 1. PostgreSQL database
echo "Starting PostgreSQL..."
docker run -d \
    --name ml-postgres \
    --restart=unless-stopped \
    -p 5432:5432 \
    -e POSTGRES_DB=mldb \
    -e POSTGRES_USER=mluser \
    -e POSTGRES_PASSWORD=mlpass \
    -v ml-postgres-data:/var/lib/postgresql/data \
    --memory="2g" \
    postgres:15

# 2. Redis cache
echo "Starting Redis..."
docker run -d \
    --name ml-redis \
    --restart=unless-stopped \
    -p 6379:6379 \
    -v ml-redis-data:/data \
    --memory="1g" \
    redis:7-alpine

# 3. Model registry (MLflow)
echo "Starting MLflow..."
docker run -d \
    --name mlflow \
    --restart=unless-stopped \
    -p 5000:5000 \
    -v ml-mlflow-data:/mlflow \
    --memory="2g" \
    ghcr.io/mlflow/mlflow:latest \
    mlflow server --host 0.0.0.0 --backend-store-uri /mlflow

# 4. Model serving (TensorFlow Serving)
echo "Starting TensorFlow Serving..."
docker run -d \
    --name tf-serving \
    --restart=unless-stopped \
    -p 8501:8501 \
    -v /data/models:/models \
    --gpus all \
    --memory="4g" \
    tensorflow/serving:latest-gpu

# Wait for services
echo "Waiting for services to start..."
sleep 10

# Verify all running
echo "Verifying containers..."
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# Health checks
echo "Running health checks..."
docker exec ml-postgres pg_isready || echo "PostgreSQL not ready"
docker exec ml-redis redis-cli ping || echo "Redis not ready"

echo "ML stack deployment complete!"
echo "Services:"
echo "  PostgreSQL: localhost:5432"
echo "  Redis: localhost:6379"
echo "  MLflow: http://localhost:5000"
echo "  TF Serving: http://localhost:8501"
```

---

## Summary

This implementation guide covered:

1. **Docker Installation**: Proper setup and configuration for ML workloads
2. **Basic Operations**: Running containers in various modes with ports and environment variables
3. **Image Management**: Pulling, tagging, and pushing images to registries
4. **Lifecycle Management**: Starting, stopping, and managing container states
5. **Resource Limits**: CPU, memory, GPU, and I/O constraints for ML workloads
6. **Logs and Debugging**: Comprehensive logging, monitoring, and troubleshooting
7. **Production Management**: Health checks, restart policies, and deployment automation
8. **ML Infrastructure**: GPU support, model serving, and complete ML stack deployment

## Best Practices for ML Infrastructure

1. **Always set resource limits** to prevent runaway processes
2. **Use health checks** for production deployments
3. **Implement proper logging** with rotation
4. **Tag images with versions** for reproducibility
5. **Use named volumes** for persistent data
6. **Monitor GPU utilization** in ML containers
7. **Implement graceful shutdown** for model servers
8. **Use restart policies** for high availability
9. **Document environment variables** required for containers
10. **Test resource limits** before production deployment

## Next Steps

- **Exercise 02**: Building Custom Docker Images
- **Exercise 03**: Docker Compose for Multi-Container Applications
- **Exercise 04**: Container Networking
- **Exercise 05**: Docker Volumes and Data Persistence

---

**Document Version**: 1.0
**Last Updated**: November 2025
**Target Audience**: Junior AI Infrastructure Engineers
**Estimated Completion Time**: 4-6 hours
