#!/bin/bash
# build_optimize.sh - Build and optimize Docker images for ML

set -euo pipefail

IMAGE_NAME="${1:-ml-app}"
TAG="${2:-latest}"
DOCKERFILE="${3:-Dockerfile}"

echo "Building: $IMAGE_NAME:$TAG"

# Build with BuildKit for better caching
DOCKER_BUILDKIT=1 docker build \
    --file "$DOCKERFILE" \
    --tag "$IMAGE_NAME:$TAG" \
    --progress=plain \
    .

# Show image size
echo ""
echo "Image size:"
docker images "$IMAGE_NAME:$TAG" --format "{{.Repository}}:{{.Tag}} - {{.Size}}"

# Analyze layers with dive (if available)
if command -v dive &> /dev/null; then
    echo ""
    echo "Run 'dive $IMAGE_NAME:$TAG' to analyze layers"
fi

# Scan for vulnerabilities with trivy (if available)
if command -v trivy &> /dev/null; then
    echo ""
    echo "Scanning for vulnerabilities..."
    trivy image "$IMAGE_NAME:$TAG"
fi

echo ""
echo "Build complete: $IMAGE_NAME:$TAG"
