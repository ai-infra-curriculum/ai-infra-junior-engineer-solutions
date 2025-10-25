#!/bin/bash

# build.sh
# Build Docker images with various configurations

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
IMAGE_NAME="${IMAGE_NAME:-ml-api}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
REGISTRY="${REGISTRY:-}"
DOCKERFILE="${DOCKERFILE:-dockerfiles/Dockerfile.optimized}"

print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --name)
            IMAGE_NAME="$2"
            shift 2
            ;;
        --tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        --dockerfile)
            DOCKERFILE="$2"
            shift 2
            ;;
        --registry)
            REGISTRY="$2"
            shift 2
            ;;
        --push)
            PUSH=true
            shift
            ;;
        --no-cache)
            NO_CACHE=true
            shift
            ;;
        --platform)
            PLATFORM="$2"
            shift 2
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Build full image name
if [ -n "$REGISTRY" ]; then
    FULL_IMAGE_NAME="${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
else
    FULL_IMAGE_NAME="${IMAGE_NAME}:${IMAGE_TAG}"
fi

print_header "Building Docker Image"

print_info "Image: $FULL_IMAGE_NAME"
print_info "Dockerfile: $DOCKERFILE"

# Build arguments
BUILD_ARGS=(
    "--file" "$DOCKERFILE"
    "--tag" "$FULL_IMAGE_NAME"
)

if [ "$NO_CACHE" = true ]; then
    BUILD_ARGS+=("--no-cache")
fi

if [ -n "$PLATFORM" ]; then
    BUILD_ARGS+=("--platform" "$PLATFORM")
fi

# Enable BuildKit
export DOCKER_BUILDKIT=1

# Build
print_info "Starting build..."
if docker build "${BUILD_ARGS[@]}" .; then
    print_success "Image built successfully: $FULL_IMAGE_NAME"
else
    print_error "Build failed"
    exit 1
fi

# Get image size
IMAGE_SIZE=$(docker images "$FULL_IMAGE_NAME" --format "{{.Size}}")
print_info "Image size: $IMAGE_SIZE"

# Tag with latest if not already
if [ "$IMAGE_TAG" != "latest" ]; then
    docker tag "$FULL_IMAGE_NAME" "${IMAGE_NAME}:latest"
    print_info "Tagged as: ${IMAGE_NAME}:latest"
fi

# Push if requested
if [ "$PUSH" = true ]; then
    print_header "Pushing Image"

    if [ -z "$REGISTRY" ]; then
        print_error "Registry not specified for push"
        exit 1
    fi

    print_info "Pushing to: $REGISTRY"

    if docker push "$FULL_IMAGE_NAME"; then
        print_success "Image pushed successfully"
    else
        print_error "Push failed"
        exit 1
    fi
fi

print_header "Build Complete"
echo ""
echo "To run the image:"
echo "  docker run -p 8000:8000 $FULL_IMAGE_NAME"
echo ""
echo "To test the image:"
echo "  ./scripts/test-image.sh --image $FULL_IMAGE_NAME"
echo ""
