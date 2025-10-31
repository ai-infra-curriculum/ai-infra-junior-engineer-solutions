# Exercise 02: Building ML Docker Images

## Overview

Master building optimized Docker images for ML workloads including GPU support, multi-stage builds, framework-specific optimizations, and production-ready configurations.

## Learning Objectives

- ✅ Build optimized Dockerfiles for ML applications
- ✅ Implement multi-stage builds for smaller images
- ✅ Configure GPU support in containers
- ✅ Optimize layer caching for fast builds
- ✅ Reduce image size for ML workloads
- ✅ Build framework-specific images (PyTorch, TensorFlow)
- ✅ Implement security best practices in images

## Complete Project

This exercise includes a **complete, production-ready solution** in the `solutions/` directory with:
- Optimized Dockerfiles for PyTorch, TensorFlow, and scikit-learn
- Multi-stage build configurations
- GPU-enabled images
- Automated build and optimization scripts
- Image analysis and size reduction tools
- Comprehensive test suite (20+ tests)

See `solutions/` for working examples you can use immediately.

---

## Quick Start

### Build All ML Images

```bash
cd solutions/
./build_images.py --framework all --verbose
```

### Build Specific Framework

```bash
# TensorFlow
./build_images.py --framework tensorflow

# PyTorch
./build_images.py --framework pytorch

# Scikit-learn
./build_images.py --framework scikit-learn

# GPU-enabled PyTorch
./build_images.py --framework pytorch-gpu
```

### Analyze Image Optimization

```bash
# Analyze an image
./image_optimizer.py ml-tensorflow:latest

# Compare two images
./image_optimizer.py ml-tensorflow:latest --compare ml-pytorch:latest

# Export SBOM
./image_optimizer.py ml-tensorflow:latest --sbom tensorflow-sbom.json
```

### Run Tests

```bash
python test_ml_images.py
```

---

## Solutions Overview

### Dockerfiles

1. **Dockerfile.tensorflow** - TensorFlow 2.15 with CPU support
   - Multi-stage build
   - Optimized for size (~500MB)
   - Non-root user
   - Health checks

2. **Dockerfile.pytorch** - PyTorch 2.1 with CPU support
   - Multi-stage build
   - Optimized layer caching
   - Production-ready configuration

3. **Dockerfile.scikit-learn** - Scikit-learn 1.3.2
   - Minimal image size (~300MB)
   - Fast builds with caching
   - Includes common ML libraries

4. **Dockerfile.multistage-optimized** - Advanced optimization demo
   - Shows all optimization techniques
   - Minimal runtime dependencies
   - Efficient layer caching

5. **Dockerfile.gpu** - GPU-enabled PyTorch
   - CUDA 12.1 + cuDNN 8
   - NVIDIA runtime support
   - GPU health checks

### Tools & Scripts

1. **build_images.py** - Automated build tool
   - Build all or specific frameworks
   - Push to registry
   - Build caching
   - Security scanning integration
   - Build reports

2. **image_optimizer.py** - Image analysis tool
   - Size analysis
   - Layer inspection
   - Optimization recommendations
   - Image comparison
   - SBOM export

3. **test_ml_images.py** - Test suite
   - 20+ comprehensive tests
   - Dockerfile validation
   - Best practices checks
   - Security checks

### Best Practices Implemented

1. **Multi-stage Builds**
   - Separate build and runtime stages
   - Minimal runtime image
   - 50-70% size reduction

2. **Layer Optimization**
   - Combine RUN commands
   - Copy requirements separately
   - Clean up in same layer

3. **Security**
   - Non-root users
   - No secrets in images
   - Minimal base images
   - Security scanning ready

4. **Caching**
   - Smart layer ordering
   - Separate dependency installation
   - BuildKit cache mounts

5. **Production Ready**
   - Health checks
   - Proper labels
   - Environment variables
   - Signal handling

---

## Usage Examples

### Example 1: Build TensorFlow Image

```bash
# Build with default settings
./build_images.py --framework tensorflow

# Build without cache
./build_images.py --framework tensorflow --no-cache

# Build and push to registry
./build_images.py --framework tensorflow --registry myregistry.io --push

# Build with scanning
./build_images.py --framework tensorflow --scan

# Build with optimization checks
./build_images.py --framework tensorflow --optimize
```

### Example 2: Analyze Image

```bash
# Basic analysis
./image_optimizer.py ml-tensorflow:latest

# JSON output
./image_optimizer.py ml-tensorflow:latest --json

# Compare images
./image_optimizer.py ml-tensorflow:latest --compare ml-pytorch:latest

# Full analysis with SBOM
./image_optimizer.py ml-tensorflow:latest --sbom tf-sbom.json --verbose
```

### Example 3: Run ML Container

```bash
# Run TensorFlow container
docker run -p 8000:8000 ml-tensorflow:latest

# Run PyTorch with GPU
docker run --gpus all -p 8000:8000 ml-pytorch-gpu:latest

# Run with volume for models
docker run -v $(pwd)/models:/models -p 8000:8000 ml-sklearn:latest

# Run with custom environment
docker run -e MODEL_PATH=/models/my-model.pkl \
  -p 8000:8000 ml-sklearn:latest
```

### Example 4: Test Images

```bash
# Run all tests
python test_ml_images.py

# Run specific test
python test_ml_images.py TestMLDockerImages.test_01_dockerfile_tensorflow_exists

# Verbose output
python test_ml_images.py -v
```

---

## Key Concepts

### Multi-Stage Builds

Multi-stage builds separate build-time and runtime dependencies:

```dockerfile
# Stage 1: Builder
FROM python:3.11-slim as builder
RUN apt-get update && apt-get install -y gcc g++
RUN pip install tensorflow

# Stage 2: Runtime
FROM python:3.11-slim
COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
# Much smaller final image!
```

Benefits:
- 50-70% smaller images
- No build tools in production
- Faster deployments
- Better security

### Layer Caching

Optimize layer order for better caching:

```dockerfile
# Good: Copy requirements first
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .

# Bad: Copy everything first
COPY . .
RUN pip install -r requirements.txt
```

### Image Size Optimization

Techniques used:
1. Use slim/alpine base images
2. Combine RUN commands
3. Clean up in same layer
4. Use .dockerignore
5. Multi-stage builds
6. Remove build dependencies

### GPU Support

For GPU-enabled containers:
1. Use NVIDIA base images
2. Install CUDA-compatible packages
3. Set CUDA environment variables
4. Test GPU availability

---

## Performance Metrics

### Image Sizes

| Framework | Size | Layers | Build Time |
|-----------|------|--------|------------|
| TensorFlow | ~500MB | 12 | 3-5 min |
| PyTorch | ~600MB | 13 | 3-5 min |
| Scikit-learn | ~300MB | 10 | 2-3 min |
| Optimized | ~450MB | 8 | 4-6 min |
| GPU | ~4GB | 15 | 5-8 min |

### Optimization Results

- Multi-stage build: 50-70% size reduction
- Layer optimization: 30-40% faster builds
- BuildKit cache: 80-90% faster rebuilds

---

## Resources

- [Dockerfile Best Practices](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/)
- [PyTorch Docker Images](https://hub.docker.com/r/pytorch/pytorch)
- [TensorFlow Docker Images](https://hub.docker.com/r/tensorflow/tensorflow)
- [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)
- [Docker BuildKit](https://docs.docker.com/build/buildkit/)

---

## Next Steps

1. **Exercise 03: Docker Compose** - Multi-container ML applications
2. Practice optimizing existing ML Docker images
3. Build custom base images for your ML stack
4. Implement automated image builds in CI/CD
5. Explore advanced topics: BuildKit, BuildX, multi-arch builds

---

**Build optimized Docker images for ML workloads! 🐳🤖**
