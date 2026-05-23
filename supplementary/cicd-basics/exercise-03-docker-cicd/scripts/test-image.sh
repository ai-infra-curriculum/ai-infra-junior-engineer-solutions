#!/bin/bash

# test-image.sh
# Test Docker image functionality

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

IMAGE_NAME="${1:-ml-api:latest}"
CONTAINER_NAME="ml-api-test-$$"
PORT="8000"

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

cleanup() {
    print_info "Cleaning up..."
    docker stop "$CONTAINER_NAME" >/dev/null 2>&1 || true
    docker rm "$CONTAINER_NAME" >/dev/null 2>&1 || true
}

trap cleanup EXIT

print_info "Testing image: $IMAGE_NAME"

# Start container
print_info "Starting container..."
if ! docker run -d \
    --name "$CONTAINER_NAME" \
    -p "$PORT:8000" \
    "$IMAGE_NAME"; then
    print_error "Failed to start container"
    exit 1
fi

# Wait for container to be healthy
print_info "Waiting for container to be ready..."
RETRIES=30
while [ $RETRIES -gt 0 ]; do
    if docker inspect "$CONTAINER_NAME" | grep -q '"Status": "healthy"'; then
        break
    fi

    if docker inspect "$CONTAINER_NAME" | grep -q '"Status": "running"'; then
        # If no healthcheck, just check if running
        sleep 2
        break
    fi

    sleep 1
    RETRIES=$((RETRIES - 1))
done

if [ $RETRIES -eq 0 ]; then
    print_error "Container failed to become healthy"
    docker logs "$CONTAINER_NAME"
    exit 1
fi

print_success "Container is running"

# Test health endpoint
print_info "Testing /health endpoint..."
if curl -f -s "http://localhost:$PORT/health" > /dev/null; then
    print_success "Health check passed"
else
    print_error "Health check failed"
    docker logs "$CONTAINER_NAME"
    exit 1
fi

# Test root endpoint
print_info "Testing / endpoint..."
if curl -f -s "http://localhost:$PORT/" > /dev/null; then
    print_success "Root endpoint passed"
else
    print_error "Root endpoint failed"
    exit 1
fi

# Test prediction endpoint
print_info "Testing /predict endpoint..."
PREDICTION=$(curl -s -X POST \
    -H "Content-Type: application/json" \
    -d '{"features": [1.0, 2.0, 3.0, 4.0]}' \
    "http://localhost:$PORT/predict")

if echo "$PREDICTION" | grep -q "prediction"; then
    print_success "Prediction endpoint passed"
    print_info "Response: $PREDICTION"
else
    print_error "Prediction endpoint failed"
    exit 1
fi

# Check container logs
print_info "Checking container logs..."
if docker logs "$CONTAINER_NAME" | grep -q "Starting ML API"; then
    print_success "Application started correctly"
else
    print_error "Application startup issue"
    docker logs "$CONTAINER_NAME"
    exit 1
fi

print_success "All tests passed!"
echo ""
echo "Image $IMAGE_NAME is working correctly"
