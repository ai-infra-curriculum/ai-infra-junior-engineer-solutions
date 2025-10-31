# Step-by-Step Implementation Guide: Building ML Docker Images

## Overview

Build optimized Docker images for ML applications including PyTorch, TensorFlow, and production ML services. Learn multi-stage builds, layer caching, GPU support, and image optimization techniques.

**Time**: 3-4 hours | **Difficulty**: Intermediate

---

## Learning Objectives

✅ Build optimized ML Docker images
✅ Configure GPU support in containers
✅ Implement multi-stage builds
✅ Optimize layer caching
✅ Handle large model files
✅ Create production-ready ML images
✅ Implement health checks and monitoring

---

## Phase 1: PyTorch ML Image

### Dockerfile for PyTorch Training

```dockerfile
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

WORKDIR /workspace

# Install additional dependencies
RUN pip install --no-cache-dir \
    tensorboard \
    mlflow \
    wandb \
    transformers \
    datasets \
    scikit-learn \
    pandas \
    matplotlib

# Copy training code
COPY src/ ./src/
COPY configs/ ./configs/
COPY train.py .

# Create directories for outputs
RUN mkdir -p /workspace/checkpoints /workspace/logs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Training command
CMD ["python", "train.py", "--config", "configs/default.yaml"]
```

### Building and Running

```bash
# Build
docker build -t ml-training:pytorch -f Dockerfile.pytorch .

# Run training (GPU)
docker run --gpus all \
    -v $(pwd)/data:/workspace/data \
    -v $(pwd)/checkpoints:/workspace/checkpoints \
    ml-training:pytorch

# Run training (CPU only)
docker run \
    -v $(pwd)/data:/workspace/data \
    ml-training:pytorch
```

---

## Phase 2: TensorFlow ML Image

```dockerfile
FROM tensorflow/tensorflow:2.12.0-gpu

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose TensorBoard port
EXPOSE 6006

ENV TF_CPP_MIN_LOG_LEVEL=2

# Healthcheck
HEALTHCHECK --interval=30s CMD python -c "import tensorflow as tf; print(tf.__version__)"

CMD ["python", "train_tf.py"]
```

---

## Phase 3: Multi-Stage Build for Production

### Optimized Production Image

```dockerfile
# Stage 1: Builder
FROM python:3.10 AS builder

WORKDIR /build

COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM python:3.10-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY app.py .
COPY models/ ./models/

# Create non-root user
RUN useradd -m -u 1000 mluser && \
    chown -R mluser:mluser /app

USER mluser

ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "app.py"]
```

**Size comparison:**
- Single stage: ~2.5 GB
- Multi-stage: ~800 MB
- **Savings: 68%**

---

## Phase 4: GPU-Enabled ML Image

### CUDA-Enabled Dockerfile

```dockerfile
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04

# Install Python
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch with CUDA support
RUN pip3 install --no-cache-dir \
    torch==2.0.0+cu117 \
    torchvision==0.15.0+cu117 \
    --extra-index-url https://download.pytorch.org/whl/cu117

COPY . .

# Verify CUDA
RUN python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"

CMD ["python3", "train_gpu.py"]
```

### Running with GPU

```bash
# Single GPU
docker run --gpus '"device=0"' ml-training:cuda

# All GPUs
docker run --gpus all ml-training:cuda

# Multiple specific GPUs
docker run --gpus '"device=0,1"' ml-training:cuda

# With memory limit
docker run --gpus all -m 16g --cpus 8 ml-training:cuda
```

---

## Phase 5: FastAPI ML Serving Image

### Production API Server

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    torch==2.0.0 \
    transformers==4.35.0 \
    python-multipart

# Copy application
COPY api/ ./api/
COPY models/ ./models/

# Download model at build time
RUN python -c "from transformers import AutoModel; AutoModel.from_pretrained('bert-base-uncased')"

EXPOSE 8000

# Production server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

---

## Phase 6: Handling Large Models

### Option 1: Download at Runtime

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY download_model.py .

# Download model on first run
CMD ["sh", "-c", "python download_model.py && python app.py"]
```

### Option 2: Use Volume Mounts

```bash
# Download models locally first
python download_models.py

# Run with volume mount
docker run \
    -v $(pwd)/models:/app/models:ro \
    ml-api:latest
```

### Option 3: Use Multi-Stage with Caching

```dockerfile
# Download models in builder stage
FROM python:3.10 AS model-downloader

RUN pip install transformers torch

RUN python -c "
from transformers import AutoModel, AutoTokenizer
model_name = 'bert-base-uncased'
AutoModel.from_pretrained(model_name)
AutoTokenizer.from_pretrained(model_name)
"

# Copy to final image
FROM python:3.10-slim

COPY --from=model-downloader /root/.cache /root/.cache

# Rest of Dockerfile...
```

---

## Best Practices

### 1. Layer Caching Optimization

```dockerfile
# ❌ Bad: Changes to code invalidate dependency install
COPY . .
RUN pip install -r requirements.txt

# ✅ Good: Dependencies cached separately
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
```

### 2. Security

```dockerfile
# Create non-root user
RUN useradd -m -u 1000 mluser
USER mluser

# Don't expose secrets
# Use build args or runtime env vars
ARG API_KEY
ENV API_KEY=${API_KEY}
```

### 3. Image Size

```dockerfile
# Use slim variants
FROM python:3.10-slim  # vs python:3.10 (900MB vs 300MB)

# Clean up in same layer
RUN apt-get update && apt-get install -y pkg \
    && rm -rf /var/lib/apt/lists/*

# Use --no-cache-dir for pip
RUN pip install --no-cache-dir package
```

### 4. Health Checks

```dockerfile
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
```

---

## Testing Images

```bash
# Build
docker build -t ml-app:test .

# Run tests
docker run --rm ml-app:test pytest tests/

# Security scan
docker scan ml-app:test

# Check image size
docker images ml-app:test

# Inspect layers
docker history ml-app:test
```

---

## Image Optimization Checklist

✅ Use multi-stage builds
✅ Use appropriate base image (-slim, -alpine)
✅ Minimize layers (combine RUN commands)
✅ Order layers by change frequency
✅ Use .dockerignore
✅ Don't include unnecessary files
✅ Clean up in same RUN command
✅ Use --no-cache-dir for pip
✅ Pin dependency versions
✅ Implement health checks

---

**ML Docker images optimized!** 🚀
