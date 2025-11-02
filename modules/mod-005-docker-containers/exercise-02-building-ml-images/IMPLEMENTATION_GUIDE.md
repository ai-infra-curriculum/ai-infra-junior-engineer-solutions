# Implementation Guide: Building ML Docker Images

## Exercise Information

**Module**: MOD-005 Docker Containers
**Exercise**: 02 - Building Custom ML Images
**Difficulty**: Beginner to Intermediate
**Estimated Time**: 3-4 hours

## Overview

This implementation guide provides comprehensive, step-by-step instructions for mastering the art of building custom Docker images optimized for machine learning workloads. You'll learn Dockerfile best practices, multi-stage builds, layer optimization, GPU support, and production-ready patterns specifically tailored for ML applications.

### Learning Objectives

By completing this guide, you will:

- Write production-ready Dockerfiles from scratch
- Implement multi-stage builds to reduce image size by 80%+
- Apply layer caching strategies to accelerate build times
- Containerize Python and ML applications with proper dependencies
- Optimize ML model serving containers for CPU/GPU deployments
- Implement security best practices (non-root users, minimal base images)
- Work with PyTorch and TensorFlow base images
- Add CUDA and GPU support for ML workloads
- Scan images for security vulnerabilities
- Create production ML image builders

---

## Part 1: Dockerfile Basics and Best Practices

### Understanding Docker Image Layers

Docker images are built in layers. Each instruction in a Dockerfile creates a new layer. Understanding this is crucial for optimization.

#### Layer Architecture

```
┌─────────────────────────────────┐
│  CMD ["python", "app.py"]       │  ← Layer 5
├─────────────────────────────────┤
│  COPY app.py .                  │  ← Layer 4
├─────────────────────────────────┤
│  RUN pip install -r req.txt     │  ← Layer 3
├─────────────────────────────────┤
│  COPY requirements.txt .        │  ← Layer 2
├─────────────────────────────────┤
│  FROM python:3.11-slim          │  ← Layer 1 (Base)
└─────────────────────────────────┘
```

### Best Practice #1: Order Matters for Caching

**Rule**: Place instructions that change frequently at the bottom of your Dockerfile.

#### Bad Example (Cache Invalidation)

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Application code changes frequently
COPY . .

# Requirements change infrequently
COPY requirements.txt .
RUN pip install -r requirements.txt

CMD ["python", "app.py"]
```

**Problem**: Every time `app.py` changes, Docker rebuilds ALL layers, including pip install.

#### Good Example (Cache Optimization)

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# 1. Copy requirements first (rarely changes)
COPY requirements.txt .

# 2. Install dependencies (cached until requirements.txt changes)
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copy application code last (changes frequently)
COPY . .

CMD ["python", "app.py"]
```

**Benefit**: Code changes only rebuild the final layer. Dependencies remain cached.

### Best Practice #2: Minimize Layers

**Rule**: Combine related commands using `&&` to reduce layers and image size.

#### Bad Example (Many Layers)

```dockerfile
FROM python:3.11-slim

RUN apt-get update
RUN apt-get install -y curl
RUN apt-get install -y wget
RUN apt-get install -y vim
RUN apt-get clean
```

**Result**: 5 layers, larger image

#### Good Example (Single Layer)

```dockerfile
FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        wget \
        vim && \
    rm -rf /var/lib/apt/lists/*
```

**Result**: 1 layer, smaller image, faster builds

### Best Practice #3: Use .dockerignore

Create a `.dockerignore` file to exclude unnecessary files from the build context.

```bash
# .dockerignore example
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
*.so
.pytest_cache/
.coverage
htmlcov/

# Virtual environments
venv/
env/
ENV/
virtualenv/

# IDE files
.vscode/
.idea/
*.swp
*.swo

# Version control
.git/
.gitignore
.gitattributes

# Documentation
docs/
*.md
!README.md

# Data files
*.csv
*.h5
*.pkl
data/
datasets/
models/*.pth
models/*.h5

# OS files
.DS_Store
Thumbs.db

# Build artifacts
build/
dist/
*.egg-info/
```

### Best Practice #4: Pin Versions

**Rule**: Always specify exact versions for reproducibility.

#### Bad Example

```dockerfile
FROM python:latest

RUN pip install flask torch numpy
```

**Problems**:
- `latest` changes over time
- Unspecified package versions can break
- Not reproducible

#### Good Example

```dockerfile
FROM python:3.11-slim

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
```

```text
# requirements.txt
flask==3.0.0
torch==2.1.0
numpy==1.24.3
pillow==10.1.0
```

### Best Practice #5: Use Non-Root Users

**Rule**: Run containers as non-root for security.

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Create non-root user
RUN groupadd -r appuser && \
    useradd -r -g appuser -u 1000 appuser

# Install dependencies as root
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application with correct ownership
COPY --chown=appuser:appuser . .

# Switch to non-root user
USER appuser

CMD ["python", "app.py"]
```

### Best Practice #6: Leverage Build Cache

Understanding how Docker caches layers:

```dockerfile
# These layers are cached independently
FROM python:3.11-slim                    # Cached unless base image updates

WORKDIR /app                             # Cached (rarely changes)

COPY requirements.txt .                  # Cached until file content changes

RUN pip install -r requirements.txt      # Cached if requirements.txt unchanged

COPY . .                                 # Rebuilt when any file changes

CMD ["python", "app.py"]                 # Cached (no execution, just metadata)
```

**Tip**: Use `--no-cache` flag when debugging: `docker build --no-cache -t myapp .`

### Best Practice #7: Clean Up in Same Layer

**Rule**: Clean up temporary files in the same RUN command that created them.

#### Bad Example

```dockerfile
RUN apt-get update
RUN apt-get install -y curl
RUN rm -rf /var/lib/apt/lists/*
```

**Problem**: The cleanup happens in a different layer, so the image still contains the apt lists.

#### Good Example

```dockerfile
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*
```

**Benefit**: Cleanup happens in the same layer, reducing final image size.

---

## Part 2: Multi-Stage Builds for ML Images

Multi-stage builds are essential for creating small, efficient ML images. They separate the build environment from the runtime environment.

### Why Multi-Stage Builds?

**Problem**: ML libraries often require build tools (gcc, g++, make) to compile from source. These tools:
- Add 500MB+ to image size
- Are not needed at runtime
- Increase attack surface

**Solution**: Use one stage to build, another stage for runtime.

### Basic Multi-Stage Pattern

```dockerfile
# Stage 1: Builder - includes build tools
FROM python:3.11 AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        make && \
    rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: Runtime - minimal image
FROM python:3.11-slim

WORKDIR /app

# Install only runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Copy only the installed packages (not build tools!)
COPY --from=builder /root/.local /root/.local

# Copy application
COPY . .

# Update PATH
ENV PATH=/root/.local/bin:$PATH

CMD ["python", "app.py"]
```

**Result**: Final image is ~60% smaller because it doesn't include gcc, g++, make, etc.

### Advanced ML Multi-Stage Build

```dockerfile
# syntax=docker/dockerfile:1.4

#############################################
# Stage 1: Base dependencies
#############################################
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

#############################################
# Stage 2: Builder - compile dependencies
#############################################
FROM base AS builder

# Install build tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        make \
        cmake \
        git \
        curl && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

#############################################
# Stage 3: Testing (optional)
#############################################
FROM builder AS testing

COPY requirements-dev.txt .
RUN pip install --user --no-cache-dir -r requirements-dev.txt

COPY tests/ tests/
COPY src/ src/

RUN pytest tests/ --verbose

#############################################
# Stage 4: Runtime - production image
#############################################
FROM base AS runtime

# Install only runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgomp1 \
        libgfortran5 \
        curl && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r mluser && \
    useradd -r -g mluser -u 1000 mluser

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local

# Copy application
COPY --chown=mluser:mluser . .

# Update PATH
ENV PATH=/root/.local/bin:$PATH

USER mluser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "4", "app:app"]
```

### Building Specific Stages

```bash
# Build just the runtime stage (default)
docker build -t ml-app:prod .

# Build and run tests
docker build --target testing -t ml-app:test .

# Build development image with extra tools
docker build --target builder -t ml-app:dev .
```

### Multi-Stage with Model Files

```dockerfile
# Stage 1: Download/prepare model
FROM python:3.11-slim AS model-prep

WORKDIR /models

# Download pre-trained model
RUN pip install --no-cache-dir gdown && \
    gdown https://drive.google.com/uc?id=MODEL_ID -O model.pth

# Stage 2: Build dependencies
FROM python:3.11 AS builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 3: Runtime
FROM python:3.11-slim

WORKDIR /app

# Copy model from first stage
COPY --from=model-prep /models/model.pth /app/models/

# Copy packages from builder
COPY --from=builder /root/.local /root/.local

COPY . .

ENV PATH=/root/.local/bin:$PATH

CMD ["python", "serve.py"]
```

---

## Part 3: Optimizing Layer Caching

Layer caching is critical for fast iterative development. Understanding cache invalidation is key.

### Cache Invalidation Rules

Docker invalidates cache when:
1. **File content changes**: Any modification to copied files
2. **Instruction changes**: Dockerfile instruction text changes
3. **Parent layer changes**: Any previous layer invalidates all subsequent layers
4. **Base image updates**: Using tags like `latest` can cause cache misses

### Optimization Strategy #1: Separate Dependency Installation

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Layer 1: Copy only requirements (changes rarely)
COPY requirements.txt .

# Layer 2: Install dependencies (cached unless requirements.txt changes)
RUN pip install --no-cache-dir -r requirements.txt

# Layer 3: Copy source code (changes frequently)
COPY src/ ./src/

# Layer 4: Copy config (changes occasionally)
COPY config/ ./config/

# Layer 5: Copy main app (changes frequently)
COPY app.py .

CMD ["python", "app.py"]
```

**Result**: Code changes don't rebuild dependencies.

### Optimization Strategy #2: Split Requirements

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install stable dependencies first (rarely change)
COPY requirements-core.txt .
RUN pip install --no-cache-dir -r requirements-core.txt

# Install ML dependencies (change occasionally)
COPY requirements-ml.txt .
RUN pip install --no-cache-dir -r requirements-ml.txt

# Install dev dependencies (for development builds)
ARG INSTALL_DEV=false
COPY requirements-dev.txt .
RUN if [ "$INSTALL_DEV" = "true" ]; then \
        pip install --no-cache-dir -r requirements-dev.txt; \
    fi

COPY . .

CMD ["python", "app.py"]
```

```text
# requirements-core.txt (stable)
flask==3.0.0
gunicorn==21.2.0
requests==2.31.0

# requirements-ml.txt (changes with model updates)
torch==2.1.0
torchvision==0.16.0
transformers==4.35.0

# requirements-dev.txt (development only)
pytest==7.4.3
black==23.11.0
flake8==6.1.0
```

### Optimization Strategy #3: Cache Mount for pip

Use BuildKit cache mounts to cache pip downloads across builds:

```dockerfile
# syntax=docker/dockerfile:1.4

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

# Use cache mount for pip downloads
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

COPY . .

CMD ["python", "app.py"]
```

**Benefit**: pip packages are cached on host, speeding up rebuilds even when requirements.txt changes.

### Optimization Strategy #4: Conditional Layers

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Base requirements (always installed)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# GPU support (conditional)
ARG WITH_GPU=false
RUN if [ "$WITH_GPU" = "true" ]; then \
        pip install --no-cache-dir \
            torch==2.1.0+cu118 \
            --index-url https://download.pytorch.org/whl/cu118; \
    else \
        pip install --no-cache-dir \
            torch==2.1.0+cpu \
            --index-url https://download.pytorch.org/whl/cpu; \
    fi

COPY . .

CMD ["python", "app.py"]
```

```bash
# Build CPU version (smaller, faster)
docker build -t ml-app:cpu .

# Build GPU version
docker build --build-arg WITH_GPU=true -t ml-app:gpu .
```

### Measuring Cache Efficiency

```bash
# First build (no cache)
time docker build --no-cache -t ml-app:test .
# Real: 5m 23s

# Second build (full cache)
time docker build -t ml-app:test .
# Real: 0m 2s

# Build after changing app.py (partial cache)
echo "# comment" >> app.py
time docker build -t ml-app:test .
# Real: 0m 8s

# Build after changing requirements.txt (cache miss on dependencies)
echo "numpy==1.24.3" >> requirements.txt
time docker build -t ml-app:test .
# Real: 2m 45s
```

---

## Part 4: PyTorch and TensorFlow Base Images

Using official ML framework base images can save time and ensure compatibility.

### PyTorch Official Images

#### CPU-Only PyTorch

```dockerfile
# Official PyTorch CPU image
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

# Already includes:
# - Python 3.10
# - PyTorch 2.1.0 (CPU)
# - torchvision
# - torchaudio

# Install additional dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "train.py"]
```

#### GPU-Enabled PyTorch

```dockerfile
# Official PyTorch with CUDA support
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

# Verify GPU support
RUN python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "train_gpu.py"]
```

#### Custom PyTorch Image (Smaller)

```dockerfile
# Start with minimal CUDA base
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Install Python
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.11 \
        python3-pip && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir \
    torch==2.1.0+cu118 \
    torchvision==0.16.0+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python3", "app.py"]
```

### TensorFlow Official Images

#### TensorFlow CPU

```dockerfile
# Official TensorFlow CPU image
FROM tensorflow/tensorflow:2.14.0

WORKDIR /app

# Already includes:
# - Python 3.11
# - TensorFlow 2.14.0 (CPU)
# - Jupyter
# - Common ML libraries

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "train.py"]
```

#### TensorFlow GPU

```dockerfile
# Official TensorFlow GPU image
FROM tensorflow/tensorflow:2.14.0-gpu

WORKDIR /app

# Verify GPU support
RUN python -c "import tensorflow as tf; print(f'GPU devices: {tf.config.list_physical_devices(\"GPU\")}')"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "train_gpu.py"]
```

#### Custom TensorFlow Image

```dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Install Python
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.11 \
        python3-pip && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install TensorFlow with GPU support
RUN pip install --no-cache-dir tensorflow[and-cuda]==2.14.0

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python3", "app.py"]
```

### Comparison: Official vs Custom Images

| Aspect | Official Images | Custom Images |
|--------|----------------|---------------|
| **Size** | Larger (2-5 GB) | Smaller (500MB-2GB) |
| **Setup Time** | Immediate | Manual installation |
| **Compatibility** | Guaranteed | Requires testing |
| **Updates** | Official releases | Manual updates |
| **Flexibility** | Limited | Full control |
| **Best For** | Quick start, prototyping | Production, optimization |

### Multi-Framework Support

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install both PyTorch and TensorFlow (CPU versions)
RUN pip install --no-cache-dir \
    torch==2.1.0+cpu \
    torchvision==0.16.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir tensorflow-cpu==2.14.0

# Install ONNX for model conversion
RUN pip install --no-cache-dir \
    onnx==1.15.0 \
    onnxruntime==1.16.0

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "serve.py"]
```

---

## Part 5: CUDA and GPU Support

GPU support requires NVIDIA Container Toolkit and proper image configuration.

### Prerequisites for GPU Support

```bash
# Check if NVIDIA drivers are installed
nvidia-smi

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### CUDA Base Images

NVIDIA provides official CUDA base images with different variants:

```dockerfile
# Variant 1: Base - minimal CUDA toolkit
FROM nvidia/cuda:11.8.0-base-ubuntu22.04

# Variant 2: Runtime - CUDA runtime libraries
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Variant 3: cuDNN Runtime - includes cuDNN for deep learning
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Variant 4: Devel - includes development tools (larger)
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
```

**Recommendation**: Use `cudnn8-runtime` for production inference, `cudnn8-devel` for training.

### Complete GPU-Enabled ML Image

```dockerfile
# Multi-stage build for GPU inference
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 AS builder

# Install Python and build tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.11 \
        python3-pip \
        python3.11-dev \
        gcc \
        g++ && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir \
    torch==2.1.0+cu118 \
    torchvision==0.16.0+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Runtime stage
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Install Python runtime
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.11 \
        python3-pip \
        curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed packages
COPY --from=builder /root/.local /root/.local

# Copy application
COPY . .

ENV PATH=/root/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

# Verify GPU access
RUN python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'Device count: {torch.cuda.device_count()}')"

EXPOSE 8000

CMD ["python3", "serve.py"]
```

### Running GPU Containers

```bash
# Build GPU image
docker build -t ml-gpu:latest .

# Run with GPU access
docker run --gpus all -p 8000:8000 ml-gpu:latest

# Run with specific GPU
docker run --gpus '"device=0"' -p 8000:8000 ml-gpu:latest

# Run with multiple GPUs
docker run --gpus 2 -p 8000:8000 ml-gpu:latest

# Verify GPU access inside container
docker run --gpus all ml-gpu:latest nvidia-smi
```

### GPU Memory Management

```dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

WORKDIR /app

# Install dependencies...
# (previous steps omitted for brevity)

# Set GPU memory environment variables
ENV CUDA_VISIBLE_DEVICES=0
ENV CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
ENV CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log

# PyTorch GPU memory settings
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# TensorFlow GPU memory settings
ENV TF_FORCE_GPU_ALLOW_GROWTH=true
ENV TF_GPU_ALLOCATOR=cuda_malloc_async

COPY . .

CMD ["python", "serve.py"]
```

### Multi-GPU Setup

```dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

WORKDIR /app

# Install horovod for distributed training
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.11 \
        python3-pip \
        cmake \
        g++ && \
    pip install --no-cache-dir \
        torch==2.1.0+cu118 \
        --index-url https://download.pytorch.org/whl/cu118 && \
    pip install --no-cache-dir horovod[pytorch] && \
    rm -rf /var/lib/apt/lists/*

COPY . .

# Support distributed training
CMD ["horovodrun", "-np", "4", "python", "train_distributed.py"]
```

```bash
# Run with all GPUs for distributed training
docker run --gpus all ml-distributed:latest
```

---

## Part 6: Image Size Optimization

Optimizing image size improves deployment speed, reduces storage costs, and minimizes attack surface.

### Size Optimization Techniques

#### Technique 1: Use Slim/Alpine Base Images

```dockerfile
# Standard Python: ~1.02 GB
FROM python:3.11

# Slim variant: ~182 MB
FROM python:3.11-slim

# Alpine variant: ~52 MB (smallest, but compatibility issues)
FROM python:3.11-alpine
```

**Recommendation**: Use `slim` for ML workloads (Alpine lacks glibc, causing issues with compiled ML libraries).

#### Technique 2: Multi-Stage Builds

```dockerfile
# Builder stage - includes build tools (not in final image)
FROM python:3.11 AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && \
    apt-get install -y gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Runtime stage - minimal
FROM python:3.11-slim

WORKDIR /app

# Copy only installed packages
COPY --from=builder /root/.local /root/.local

COPY . .

ENV PATH=/root/.local/bin:$PATH

CMD ["python", "app.py"]
```

**Result**: ~60% size reduction

#### Technique 3: Install CPU-Only ML Libraries

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# CPU-only PyTorch (much smaller than GPU version)
RUN pip install --no-cache-dir \
    torch==2.1.0+cpu \
    torchvision==0.16.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# CPU-only TensorFlow
RUN pip install --no-cache-dir tensorflow-cpu==2.14.0

COPY . .

CMD ["python", "serve.py"]
```

**Savings**:
- PyTorch: GPU version ~2.5GB → CPU version ~800MB
- TensorFlow: GPU version ~2.8GB → CPU version ~450MB

#### Technique 4: Remove Unnecessary Files

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    # Remove pip cache
    rm -rf ~/.cache/pip && \
    # Remove unnecessary Python files
    find /usr/local/lib/python3.11 -type d -name __pycache__ -exec rm -rf {} + && \
    find /usr/local/lib/python3.11 -type f -name '*.pyc' -delete && \
    find /usr/local/lib/python3.11 -type f -name '*.pyo' -delete

COPY . .

CMD ["python", "app.py"]
```

#### Technique 5: Minimize System Dependencies

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install only required system libraries
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgomp1 \
        libgfortran5 && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "app.py"]
```

#### Technique 6: Use .dockerignore Effectively

```bash
# .dockerignore - exclude large files
*.h5
*.pth
*.ckpt
*.bin
*.pkl
data/
datasets/
models/checkpoints/
logs/
*.log
.git/
tests/
docs/
```

### Size Comparison Example

```dockerfile
# Version 1: Unoptimized (2.8 GB)
FROM python:3.11
COPY . .
RUN pip install torch torchvision flask

# Version 2: Basic optimization (1.2 GB)
FROM python:3.11-slim
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .

# Version 3: Multi-stage (650 MB)
FROM python:3.11 AS builder
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

FROM python:3.11-slim
COPY --from=builder /root/.local /root/.local
COPY app.py .

# Version 4: CPU-only + multi-stage (420 MB)
FROM python:3.11-slim AS builder
RUN pip install --user --no-cache-dir \
    torch==2.1.0+cpu \
    torchvision==0.16.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

FROM python:3.11-slim
COPY --from=builder /root/.local /root/.local
COPY app.py .
```

### Analyzing Image Size

```bash
# Check image size
docker images ml-app

# View layer sizes
docker history ml-app --human

# Find largest layers
docker history ml-app --human --no-trunc | sort -k2 -hr | head -10

# Use dive tool for detailed analysis
docker run --rm -it \
    -v /var/run/docker.sock:/var/run/docker.sock \
    wagoodman/dive:latest ml-app
```

### Size Optimization Checklist

- [ ] Use `slim` or `alpine` base images
- [ ] Implement multi-stage builds
- [ ] Use CPU-only ML libraries when GPU not needed
- [ ] Combine RUN commands to reduce layers
- [ ] Clean up package manager caches
- [ ] Remove development dependencies
- [ ] Use `.dockerignore` to exclude large files
- [ ] Delete `__pycache__` and `.pyc` files
- [ ] Minimize system dependencies
- [ ] Consider distroless images for production

---

## Part 7: Production ML Image Builder

A complete, production-ready ML image template with all best practices.

### Complete Production Dockerfile

```dockerfile
# syntax=docker/dockerfile:1.4

################################################################################
# Production ML Image Template
# - Multi-stage build
# - GPU support (configurable)
# - Security hardened
# - Optimized for size and performance
################################################################################

################################################################################
# Build arguments
################################################################################
ARG PYTHON_VERSION=3.11
ARG CUDA_VERSION=11.8.0
ARG CUDNN_VERSION=8
ARG WITH_GPU=false

################################################################################
# Stage 1: Base configuration
################################################################################
FROM python:${PYTHON_VERSION}-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100

WORKDIR /app

################################################################################
# Stage 2: Dependencies builder
################################################################################
FROM base AS builder

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        make \
        cmake \
        git \
        curl && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt requirements-ml.txt ./

# Install Python dependencies
RUN pip install --user --no-cache-dir -r requirements.txt && \
    pip install --user --no-cache-dir -r requirements-ml.txt

################################################################################
# Stage 3: GPU builder (conditional)
################################################################################
FROM nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-runtime-ubuntu22.04 AS gpu-base

# Install Python
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} \
        python3-pip && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install GPU-specific ML libraries
RUN pip install --no-cache-dir \
    torch==2.1.0+cu118 \
    torchvision==0.16.0+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

################################################################################
# Stage 4: Testing (optional)
################################################################################
FROM builder AS testing

COPY requirements-dev.txt .
RUN pip install --user --no-cache-dir -r requirements-dev.txt

COPY tests/ tests/
COPY src/ src/

# Run tests during build
RUN python -m pytest tests/ -v --tb=short

################################################################################
# Stage 5: Runtime (production)
################################################################################
FROM base AS runtime

# Install runtime system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgomp1 \
        libgfortran5 \
        curl \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

# Create non-root user
RUN groupadd -r mluser && \
    useradd -r -g mluser -u 1000 -m -d /home/mluser mluser && \
    chown -R mluser:mluser /app

# Copy installed packages from builder
COPY --from=builder --chown=mluser:mluser /root/.local /home/mluser/.local

# Copy application code
COPY --chown=mluser:mluser src/ ./src/
COPY --chown=mluser:mluser models/ ./models/
COPY --chown=mluser:mluser config/ ./config/
COPY --chown=mluser:mluser app.py wsgi.py ./

# Update PATH
ENV PATH=/home/mluser/.local/bin:$PATH

# Switch to non-root user
USER mluser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Metadata labels
LABEL maintainer="ml-team@company.com" \
      version="1.0.0" \
      description="Production ML serving container" \
      org.opencontainers.image.source="https://github.com/company/ml-app"

# Default command
CMD ["gunicorn", \
     "--bind", "0.0.0.0:8000", \
     "--workers", "4", \
     "--worker-class", "sync", \
     "--timeout", "120", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "--log-level", "info", \
     "wsgi:application"]
```

### Supporting Files

#### requirements.txt (core dependencies)

```text
flask==3.0.0
gunicorn==21.2.0
requests==2.31.0
pydantic==2.5.0
python-dotenv==1.0.0
numpy==1.24.3
```

#### requirements-ml.txt (ML dependencies)

```text
torch==2.1.0+cpu
torchvision==0.16.0+cpu
--index-url https://download.pytorch.org/whl/cpu

pillow==10.1.0
scikit-learn==1.3.2
pandas==2.1.3
```

#### requirements-dev.txt (development dependencies)

```text
pytest==7.4.3
pytest-cov==4.1.0
black==23.11.0
flake8==6.1.0
mypy==1.7.1
```

#### .dockerignore

```bash
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
ENV/
env/
virtualenv/

# IDE
.vscode/
.idea/
*.swp
*.swo
.DS_Store

# Testing
.pytest_cache/
.coverage
htmlcov/
.tox/

# Data
data/
datasets/
*.csv
*.h5
*.pkl
*.pth
*.ckpt
*.bin
checkpoints/

# Documentation
docs/
*.md
!README.md

# Git
.git/
.gitignore

# Logs
logs/
*.log

# CI/CD
.github/
.gitlab-ci.yml
Jenkinsfile
```

### Build Scripts

#### build.sh

```bash
#!/bin/bash
set -e

# Configuration
IMAGE_NAME="ml-app"
VERSION="${1:-latest}"
REGISTRY="${DOCKER_REGISTRY:-docker.io/username}"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}Building ML Docker Image${NC}"
echo "Image: ${IMAGE_NAME}:${VERSION}"
echo "Registry: ${REGISTRY}"

# Build CPU version
echo -e "${GREEN}Building CPU version...${NC}"
docker build \
    --target runtime \
    --build-arg PYTHON_VERSION=3.11 \
    --build-arg WITH_GPU=false \
    -t ${IMAGE_NAME}:${VERSION}-cpu \
    -t ${IMAGE_NAME}:latest-cpu \
    -t ${REGISTRY}/${IMAGE_NAME}:${VERSION}-cpu \
    .

# Build GPU version
echo -e "${GREEN}Building GPU version...${NC}"
docker build \
    --target runtime \
    --build-arg PYTHON_VERSION=3.11 \
    --build-arg WITH_GPU=true \
    --build-arg CUDA_VERSION=11.8.0 \
    --build-arg CUDNN_VERSION=8 \
    -t ${IMAGE_NAME}:${VERSION}-gpu \
    -t ${IMAGE_NAME}:latest-gpu \
    -t ${REGISTRY}/${IMAGE_NAME}:${VERSION}-gpu \
    .

# Run tests
echo -e "${GREEN}Running tests...${NC}"
docker build \
    --target testing \
    -t ${IMAGE_NAME}:test \
    .

# Show image sizes
echo -e "${BLUE}Image sizes:${NC}"
docker images | grep ${IMAGE_NAME}

echo -e "${GREEN}Build complete!${NC}"
```

#### push.sh

```bash
#!/bin/bash
set -e

IMAGE_NAME="ml-app"
VERSION="${1:-latest}"
REGISTRY="${DOCKER_REGISTRY:-docker.io/username}"

echo "Pushing images to ${REGISTRY}..."

# Push CPU version
docker push ${REGISTRY}/${IMAGE_NAME}:${VERSION}-cpu
docker push ${REGISTRY}/${IMAGE_NAME}:latest-cpu

# Push GPU version
docker push ${REGISTRY}/${IMAGE_NAME}:${VERSION}-gpu
docker push ${REGISTRY}/${IMAGE_NAME}:latest-gpu

echo "Push complete!"
```

#### test.sh

```bash
#!/bin/bash
set -e

IMAGE_NAME="ml-app"
VERSION="${1:-latest}"

echo "Testing ${IMAGE_NAME}:${VERSION}-cpu"

# Start container
CONTAINER_ID=$(docker run -d -p 8000:8000 ${IMAGE_NAME}:${VERSION}-cpu)

# Wait for startup
sleep 5

# Run health check
if curl -f http://localhost:8000/health; then
    echo "✓ Health check passed"
else
    echo "✗ Health check failed"
    docker logs $CONTAINER_ID
    docker rm -f $CONTAINER_ID
    exit 1
fi

# Test prediction endpoint
if curl -f -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"data": [1,2,3,4,5]}'; then
    echo "✓ Prediction test passed"
else
    echo "✗ Prediction test failed"
    docker logs $CONTAINER_ID
    docker rm -f $CONTAINER_ID
    exit 1
fi

# Cleanup
docker rm -f $CONTAINER_ID

echo "All tests passed!"
```

### Security Scanning

```bash
#!/bin/bash
# scan.sh - Security scanning script

IMAGE_NAME="ml-app:latest-cpu"

echo "Scanning ${IMAGE_NAME} for vulnerabilities..."

# Scan with Trivy
docker run --rm \
    -v /var/run/docker.sock:/var/run/docker.sock \
    aquasec/trivy:latest image \
    --severity HIGH,CRITICAL \
    --exit-code 1 \
    ${IMAGE_NAME}

# Scan with Grype
docker run --rm \
    -v /var/run/docker.sock:/var/run/docker.sock \
    anchore/grype:latest \
    ${IMAGE_NAME}

echo "Security scan complete!"
```

### Docker Compose for Development

```yaml
# docker-compose.yml
version: '3.8'

services:
  ml-app:
    build:
      context: .
      target: runtime
      args:
        PYTHON_VERSION: 3.11
        WITH_GPU: false
    image: ml-app:dev
    ports:
      - "8000:8000"
    volumes:
      - ./src:/app/src
      - ./models:/app/models
      - ./config:/app/config
    environment:
      - ENV=development
      - DEBUG=true
      - LOG_LEVEL=debug
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 5s
      retries: 3
      start_period: 60s

  ml-app-gpu:
    build:
      context: .
      target: runtime
      args:
        PYTHON_VERSION: 3.11
        WITH_GPU: true
        CUDA_VERSION: 11.8.0
        CUDNN_VERSION: 8
    image: ml-app:dev-gpu
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - CUDA_VISIBLE_DEVICES=0
    ports:
      - "8001:8000"
    volumes:
      - ./src:/app/src
      - ./models:/app/models
```

---

## Part 8: Advanced Patterns and Best Practices

### Pattern 1: Configuration Management

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Copy dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Support multiple environments
ARG ENV=production
ENV APP_ENV=${ENV}

# Load environment-specific configuration
COPY config/config.${ENV}.yaml /app/config/config.yaml

CMD ["python", "app.py"]
```

```bash
# Build for different environments
docker build --build-arg ENV=development -t ml-app:dev .
docker build --build-arg ENV=staging -t ml-app:staging .
docker build --build-arg ENV=production -t ml-app:prod .
```

### Pattern 2: Secrets Management

```dockerfile
# syntax=docker/dockerfile:1.4

FROM python:3.11-slim

WORKDIR /app

# Use BuildKit secrets (not stored in image)
RUN --mount=type=secret,id=aws_key,target=/run/secrets/aws_key \
    --mount=type=secret,id=aws_secret,target=/run/secrets/aws_secret \
    export AWS_ACCESS_KEY_ID=$(cat /run/secrets/aws_key) && \
    export AWS_SECRET_ACCESS_KEY=$(cat /run/secrets/aws_secret) && \
    # Download model from S3 using credentials
    python download_model.py && \
    # Credentials are not stored in the image
    unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY

COPY . .

CMD ["python", "serve.py"]
```

```bash
# Build with secrets
docker build \
    --secret id=aws_key,src=~/.aws/credentials \
    --secret id=aws_secret,src=~/.aws/credentials \
    -t ml-app:latest .
```

### Pattern 3: Model Versioning

```dockerfile
FROM python:3.11-slim

WORKDIR /app

ARG MODEL_VERSION=v1.0.0
ARG MODEL_URL=https://models.company.com/resnet50/${MODEL_VERSION}/model.pth

# Download specific model version
RUN pip install --no-cache-dir gdown && \
    mkdir -p models && \
    curl -L ${MODEL_URL} -o models/model.pth

# Label with model version
LABEL model.version=${MODEL_VERSION}
LABEL model.url=${MODEL_URL}

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "serve.py"]
```

```bash
# Build with specific model version
docker build \
    --build-arg MODEL_VERSION=v2.1.0 \
    -t ml-app:v2.1.0 .
```

### Pattern 4: Health Checks and Monitoring

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Comprehensive health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import requests; \
                   import sys; \
                   try: \
                       r = requests.get('http://localhost:8000/health', timeout=3); \
                       sys.exit(0 if r.status_code == 200 else 1); \
                   except: \
                       sys.exit(1)"

# Prometheus metrics endpoint
EXPOSE 8000 9090

CMD ["python", "serve.py"]
```

### Pattern 5: Graceful Shutdown

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Install tini for proper signal handling
RUN apt-get update && \
    apt-get install -y --no-install-recommends tini && \
    rm -rf /var/lib/apt/lists/*

ENTRYPOINT ["/usr/bin/tini", "--"]

CMD ["gunicorn", \
     "--bind", "0.0.0.0:8000", \
     "--graceful-timeout", "120", \
     "--timeout", "120", \
     "app:application"]
```

### Pattern 6: Logging Configuration

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Configure logging
ENV PYTHONUNBUFFERED=1 \
    LOG_LEVEL=INFO \
    LOG_FORMAT=json

# Create log directory
RUN mkdir -p /var/log/app && \
    chown -R 1000:1000 /var/log/app

USER 1000

CMD ["python", "app.py"]
```

---

## Summary

### Key Takeaways

1. **Dockerfile Best Practices**
   - Order instructions from least to most frequently changing
   - Combine related RUN commands to minimize layers
   - Use `.dockerignore` to exclude unnecessary files
   - Pin all versions for reproducibility
   - Run containers as non-root users

2. **Multi-Stage Builds**
   - Separate build and runtime environments
   - Copy only necessary artifacts to final stage
   - Can reduce image size by 60-80%
   - Enable different targets for dev/test/prod

3. **Layer Caching**
   - Copy requirements before application code
   - Use BuildKit cache mounts for package managers
   - Split dependencies by change frequency
   - Understand cache invalidation rules

4. **ML Framework Images**
   - Use official PyTorch/TensorFlow images for quick start
   - Build custom images for production optimization
   - Install CPU-only libraries when GPU not needed
   - Use appropriate CUDA base images for GPU support

5. **GPU Support**
   - Requires NVIDIA Container Toolkit
   - Use nvidia/cuda base images
   - Choose runtime vs devel variants appropriately
   - Test GPU access during build

6. **Size Optimization**
   - Start with slim base images
   - Use multi-stage builds
   - Install CPU-only ML libraries
   - Clean up package manager caches
   - Remove unnecessary dependencies

7. **Production Readiness**
   - Implement health checks
   - Use production WSGI servers
   - Configure proper logging
   - Handle signals gracefully
   - Scan for security vulnerabilities

### Image Size Comparison

| Configuration | Size | Use Case |
|--------------|------|----------|
| Unoptimized | 2.8 GB | Quick prototyping |
| Slim base | 1.2 GB | Development |
| Multi-stage | 650 MB | Production CPU |
| CPU-only + multi-stage | 420 MB | Production CPU optimized |
| GPU runtime | 2.1 GB | Production GPU |

### Next Steps

1. **Practice**: Build images for different ML frameworks
2. **Optimize**: Analyze and reduce your image sizes
3. **Secure**: Scan images and fix vulnerabilities
4. **Deploy**: Use images in production with orchestration
5. **Monitor**: Track image build times and sizes

### Additional Resources

- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Multi-Stage Builds](https://docs.docker.com/build/building/multi-stage/)
- [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)
- [PyTorch Docker Images](https://hub.docker.com/r/pytorch/pytorch)
- [TensorFlow Docker Images](https://hub.docker.com/r/tensorflow/tensorflow)
- [Security Scanning with Trivy](https://github.com/aquasecurity/trivy)

---

**Document Version**: 1.0
**Last Updated**: 2025-11-02
**Author**: AI Infrastructure Team
