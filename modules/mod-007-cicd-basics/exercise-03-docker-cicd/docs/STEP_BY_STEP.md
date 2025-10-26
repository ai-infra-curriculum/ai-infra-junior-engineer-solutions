# Step-by-Step Implementation Guide: Docker in CI/CD

## Overview

Automate Docker builds and deployments! Learn multi-stage builds in CI, image optimization, registry management, vulnerability scanning, and container deployment automation.

**Time**: 2 hours | **Difficulty**: Intermediate

---

## Learning Objectives

✅ Build Docker images in CI/CD
✅ Push images to container registries
✅ Implement image tagging strategies
✅ Scan images for vulnerabilities
✅ Use multi-stage builds
✅ Cache Docker layers
✅ Deploy containers automatically

---

## Docker Build Workflow

```.github/workflows/docker.yml
name: Docker Build & Push

on:
  push:
    branches: [main]
    tags: ['v*']

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  build:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write

    steps:
      - uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Log in to registry
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
          tags: |
            type=ref,event=branch
            type=semver,pattern={{version}}
            type=sha,prefix={{branch}}-

      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

      - name: Scan image
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:latest
          format: 'sarif'
          output: 'trivy-results.sarif'

      - name: Upload scan results
        uses: github/codeql-action/upload-sarif@v2
        with:
          sarif_file: 'trivy-results.sarif'
```

---

## Multi-Stage Dockerfile

```dockerfile
# Build stage
FROM python:3.11-slim AS builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim
WORKDIR /app

# Copy installed packages
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy application
COPY src/ ./src/
COPY models/ ./models/

# Non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

CMD ["python", "-m", "src.main"]
```

---

## Best Practices

✅ Use multi-stage builds
✅ Tag images with version and commit SHA
✅ Scan for vulnerabilities
✅ Use layer caching (BuildKit)
✅ Run as non-root user
✅ Keep images small
✅ Use official base images
✅ Implement image signing

---

**Docker CI/CD mastered!** 🐳
