# Step-by-Step Guide: Docker in CI/CD Pipeline

## Overview
Implement automated Docker builds using GitHub Actions, multi-stage Dockerfiles, and container registry integration for efficient ML application deployment.

## Phase 1: Multi-Stage Dockerfile (15 minutes)

### Create Application Structure
```bash
mkdir -p docker-cicd/app
cd docker-cicd

# Create simple Flask ML API
cat > app/main.py << 'EOF'
from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

@app.route('/health')
def health():
    return jsonify({"status": "healthy"})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    # Simple prediction logic
    result = sum(data.get('features', []))
    return jsonify({"prediction": result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
EOF
```

### Create Multi-Stage Dockerfile
Create `Dockerfile`:
```dockerfile
# Stage 1: Builder
FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim

WORKDIR /app

# Copy only necessary files from builder
COPY --from=builder /root/.local /root/.local
COPY app/ ./app/

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5000/health')"

CMD ["python", "app/main.py"]
```

### Create Requirements
```bash
cat > requirements.txt << 'EOF'
flask==3.0.0
numpy==1.26.2
gunicorn==21.2.0
EOF
```

**Validation**: Build locally: `docker build -t ml-app:local .`

## Phase 2: Optimize Docker Build (10 minutes)

### Add .dockerignore
Create `.dockerignore`:
```
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
*.so
*.egg
*.egg-info/
dist/
build/
.git/
.gitignore
.env
.venv/
venv/
*.md
.github/
tests/
.pytest_cache/
htmlcov/
.coverage
```

### Test Build Efficiency
```bash
# Build and check image size
docker build -t ml-app:optimized .
docker images ml-app:optimized

# Inspect layers
docker history ml-app:optimized

# Expected: Image size ~150-200MB (vs 800MB+ without multi-stage)
```

**Validation**: Image size should be significantly smaller than single-stage build.

## Phase 3: Local Docker Testing (15 minutes)

### Create Docker Compose for Testing
Create `docker-compose.yml`:
```yaml
version: '3.8'

services:
  app:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "5000:5000"
    environment:
      - FLASK_ENV=development
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 10s
      timeout: 5s
      retries: 3
      start_period: 10s

  test:
    build:
      context: .
      dockerfile: Dockerfile.test
    depends_on:
      - app
    command: pytest tests/ -v
```

### Create Test Dockerfile
Create `Dockerfile.test`:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt requirements-test.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-test.txt

COPY . .

CMD ["pytest", "tests/", "-v", "--cov=app"]
```

### Run Tests in Container
```bash
# Build and run
docker-compose up --build

# Run tests only
docker-compose run test

# Clean up
docker-compose down
```

**Validation**: Tests run successfully in containerized environment.

## Phase 4: GitHub Actions Docker Build (15 minutes)

### Create Docker Build Workflow
Create `.github/workflows/docker-build.yml`:
```yaml
name: Docker Build and Push

on:
  push:
    branches: [ main, develop ]
    tags: [ 'v*' ]
  pull_request:
    branches: [ main ]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  build-and-push:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write

    steps:
    - name: Checkout code
      uses: actions/checkout@v3

    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v2

    - name: Log in to Container Registry
      if: github.event_name != 'pull_request'
      uses: docker/login-action@v2
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}

    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v4
      with:
        images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=semver,pattern={{version}}
          type=semver,pattern={{major}}.{{minor}}
          type=sha,prefix={{branch}}-

    - name: Build and push Docker image
      uses: docker/build-push-action@v4
      with:
        context: .
        push: ${{ github.event_name != 'pull_request' }}
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
        cache-from: type=gha
        cache-to: type=gha,mode=max
```

**Validation**: Push to GitHub and verify workflow builds image successfully.

## Phase 5: Registry Integration (15 minutes)

### Configure Docker Hub (Alternative)
Create `.github/workflows/docker-hub.yml`:
```yaml
name: Docker Hub Build

on:
  push:
    tags: [ 'v*' ]

jobs:
  docker-hub:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3

    - name: Docker Hub Login
      uses: docker/login-action@v2
      with:
        username: ${{ secrets.DOCKERHUB_USERNAME }}
        password: ${{ secrets.DOCKERHUB_TOKEN }}

    - name: Extract version
      id: version
      run: echo "VERSION=${GITHUB_REF#refs/tags/v}" >> $GITHUB_OUTPUT

    - name: Build and Push
      uses: docker/build-push-action@v4
      with:
        context: .
        push: true
        tags: |
          ${{ secrets.DOCKERHUB_USERNAME }}/ml-app:latest
          ${{ secrets.DOCKERHUB_USERNAME }}/ml-app:${{ steps.version.outputs.VERSION }}
        cache-from: type=registry,ref=${{ secrets.DOCKERHUB_USERNAME }}/ml-app:buildcache
        cache-to: type=registry,ref=${{ secrets.DOCKERHUB_USERNAME }}/ml-app:buildcache,mode=max
```

### Test Registry Pull
```bash
# Pull from GitHub Container Registry
docker pull ghcr.io/username/repo:main

# Pull from Docker Hub
docker pull username/ml-app:latest

# Run pulled image
docker run -p 5000:5000 ghcr.io/username/repo:main
```

**Validation**: Successfully pull and run image from registry.

## Phase 6: Advanced CI/CD Patterns (10 minutes)

### Add Security Scanning
Update `.github/workflows/docker-build.yml`:
```yaml
    - name: Run Trivy vulnerability scanner
      uses: aquasecurity/trivy-action@master
      with:
        image-ref: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}
        format: 'sarif'
        output: 'trivy-results.sarif'

    - name: Upload Trivy results to GitHub Security
      uses: github/codeql-action/upload-sarif@v2
      with:
        sarif_file: 'trivy-results.sarif'
```

### Add Build Matrix
```yaml
  build-matrix:
    strategy:
      matrix:
        platform:
          - linux/amd64
          - linux/arm64
    steps:
    - name: Build multi-platform
      uses: docker/build-push-action@v4
      with:
        platforms: ${{ matrix.platform }}
        push: true
        tags: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:latest
```

### Create Release Pipeline
```bash
# Tag and push for release
git tag -a v1.0.0 -m "Release version 1.0.0"
git push origin v1.0.0

# GitHub Actions automatically builds and pushes:
# - ghcr.io/user/repo:v1.0.0
# - ghcr.io/user/repo:1.0
# - ghcr.io/user/repo:latest
```

**Validation**: Verify multi-platform images in registry.

## Summary

You've implemented a production-grade Docker CI/CD pipeline featuring:
- **Multi-stage builds** reducing image size by 60-70%
- **Automated builds** on every push via GitHub Actions
- **Container registry integration** with GHCR and Docker Hub support
- **Build caching** using GitHub Actions cache for faster builds
- **Security scanning** with Trivy for vulnerability detection
- **Multi-platform builds** supporting AMD64 and ARM64 architectures
- **Semantic versioning** with automatic tag generation

This pipeline ensures consistent, secure, and efficient container builds for ML applications with minimal manual intervention.
