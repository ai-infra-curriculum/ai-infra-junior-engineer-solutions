#!/bin/bash
#
# deploy_model.sh - Automated model deployment script
#
# Description:
#   Automate ML model deployment to various environments including
#   Docker, Kubernetes, and cloud platforms with validation and rollback.
#
# Usage:
#   ./deploy_model.sh [OPTIONS]
#
# Options:
#   -e, --environment ENV    Target environment (dev, staging, prod)
#   -t, --target TARGET      Deployment target (docker, k8s, cloud)
#   -m, --model PATH         Model file path
#   -v, --version VERSION    Model version
#   -i, --image NAME         Docker image name
#   -n, --namespace NS       Kubernetes namespace
#   --replicas N             Number of replicas (default: 3)
#   --validate               Validate deployment
#   --rollback               Rollback to previous version
#   --dry-run                Dry run mode
#   -h, --help               Display this help message
#

set -euo pipefail

# ===========================
# Configuration
# ===========================

readonly SCRIPT_NAME="$(basename "$0")"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Defaults
ENVIRONMENT="dev"
TARGET="docker"
MODEL_PATH=""
MODEL_VERSION=""
IMAGE_NAME="ml-model-api"
NAMESPACE="default"
REPLICAS=3
VALIDATE=false
ROLLBACK=false
DRY_RUN=false

# ===========================
# Colors
# ===========================

readonly GREEN='\033[0;32m'
readonly RED='\033[0;31m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly CYAN='\033[0;36m'
readonly RESET='\033[0m'
readonly BOLD='\033[1m'

# ===========================
# Logging
# ===========================

log_info() {
    echo -e "${BLUE}[INFO]${RESET} $*"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${RESET} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${RESET} $*" >&2
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${RESET} $*"
}

# ===========================
# Docker Deployment
# ===========================

deploy_docker() {
    log_info "Deploying to Docker..."

    # Build image
    log_info "Building Docker image: $IMAGE_NAME:$MODEL_VERSION"

    if [[ "$DRY_RUN" == false ]]; then
        docker build -t "$IMAGE_NAME:$MODEL_VERSION" .
        docker tag "$IMAGE_NAME:$MODEL_VERSION" "$IMAGE_NAME:latest"
    fi

    log_success "Docker image built: $IMAGE_NAME:$MODEL_VERSION"

    # Run container
    log_info "Starting container..."

    if [[ "$DRY_RUN" == false ]]; then
        # Stop existing container if running
        docker stop ml-model-api 2>/dev/null || true
        docker rm ml-model-api 2>/dev/null || true

        # Run new container
        docker run -d \
            --name ml-model-api \
            -p 8000:8000 \
            -v "$(pwd)/models:/app/models" \
            --restart unless-stopped \
            "$IMAGE_NAME:$MODEL_VERSION"
    fi

    log_success "Container started: ml-model-api"

    # Wait for health check
    if [[ "$VALIDATE" == true ]]; then
        validate_deployment_docker
    fi
}

validate_deployment_docker() {
    log_info "Validating Docker deployment..."

    local max_attempts=30
    local attempt=0

    while [[ $attempt -lt $max_attempts ]]; do
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            log_success "Health check passed"
            return 0
        fi

        attempt=$((attempt + 1))
        echo -n "."
        sleep 2
    done

    log_error "Health check failed after $max_attempts attempts"
    return 1
}

# ===========================
# Kubernetes Deployment
# ===========================

deploy_kubernetes() {
    log_info "Deploying to Kubernetes..."

    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl not found"
        return 1
    fi

    # Create deployment YAML
    cat > /tmp/deployment.yaml <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-model-api
  namespace: $NAMESPACE
  labels:
    app: ml-model-api
    version: $MODEL_VERSION
spec:
  replicas: $REPLICAS
  selector:
    matchLabels:
      app: ml-model-api
  template:
    metadata:
      labels:
        app: ml-model-api
        version: $MODEL_VERSION
    spec:
      containers:
      - name: api
        image: $IMAGE_NAME:$MODEL_VERSION
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: ml-model-api
  namespace: $NAMESPACE
spec:
  selector:
    app: ml-model-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
EOF

    log_info "Created deployment manifest"

    # Apply deployment
    if [[ "$DRY_RUN" == false ]]; then
        kubectl apply -f /tmp/deployment.yaml
    else
        kubectl apply -f /tmp/deployment.yaml --dry-run=client
    fi

    log_success "Deployment applied to Kubernetes"

    # Wait for rollout
    if [[ "$DRY_RUN" == false && "$VALIDATE" == true ]]; then
        validate_deployment_kubernetes
    fi
}

validate_deployment_kubernetes() {
    log_info "Validating Kubernetes deployment..."

    kubectl rollout status deployment/ml-model-api -n "$NAMESPACE" --timeout=5m

    if [[ $? -eq 0 ]]; then
        log_success "Deployment rolled out successfully"

        # Check pod status
        kubectl get pods -n "$NAMESPACE" -l app=ml-model-api

        return 0
    else
        log_error "Deployment rollout failed"
        return 1
    fi
}

# ===========================
# Rollback
# ===========================

rollback_deployment() {
    log_warning "Rolling back deployment..."

    if [[ "$TARGET" == "docker" ]]; then
        rollback_docker
    elif [[ "$TARGET" == "k8s" ]]; then
        rollback_kubernetes
    fi
}

rollback_docker() {
    log_info "Rolling back Docker deployment..."

    # Get previous version
    local previous_version=$(docker images "$IMAGE_NAME" --format "{{.Tag}}" | grep -v "latest" | head -2 | tail -1)

    if [[ -z "$previous_version" ]]; then
        log_error "No previous version found"
        return 1
    fi

    log_info "Rolling back to version: $previous_version"

    if [[ "$DRY_RUN" == false ]]; then
        docker stop ml-model-api
        docker rm ml-model-api

        docker run -d \
            --name ml-model-api \
            -p 8000:8000 \
            -v "$(pwd)/models:/app/models" \
            --restart unless-stopped \
            "$IMAGE_NAME:$previous_version"
    fi

    log_success "Rolled back to $previous_version"
}

rollback_kubernetes() {
    log_info "Rolling back Kubernetes deployment..."

    if [[ "$DRY_RUN" == false ]]; then
        kubectl rollout undo deployment/ml-model-api -n "$NAMESPACE"
        kubectl rollout status deployment/ml-model-api -n "$NAMESPACE"
    fi

    log_success "Rollback complete"
}

# ===========================
# Validation
# ===========================

validate_model() {
    log_info "Validating model file..."

    if [[ ! -f "$MODEL_PATH" ]]; then
        log_error "Model file not found: $MODEL_PATH"
        return 1
    fi

    local file_size=$(stat -f%z "$MODEL_PATH" 2>/dev/null || stat -c%s "$MODEL_PATH" 2>/dev/null)
    log_info "Model size: $((file_size / 1024 / 1024)) MB"

    log_success "Model validation passed"
}

# ===========================
# Main Deployment
# ===========================

main_deploy() {
    echo -e "${BOLD}${CYAN}Model Deployment${RESET}"
    echo "========================================"
    echo "Environment: $ENVIRONMENT"
    echo "Target: $TARGET"
    echo "Model Version: $MODEL_VERSION"
    echo "Image: $IMAGE_NAME:$MODEL_VERSION"

    if [[ "$TARGET" == "k8s" ]]; then
        echo "Namespace: $NAMESPACE"
        echo "Replicas: $REPLICAS"
    fi

    echo "========================================"
    echo

    # Validate model if path provided
    if [[ -n "$MODEL_PATH" ]]; then
        validate_model
    fi

    # Execute deployment
    if [[ "$ROLLBACK" == true ]]; then
        rollback_deployment
    else
        case "$TARGET" in
            docker)
                deploy_docker
                ;;
            k8s|kubernetes)
                deploy_kubernetes
                ;;
            *)
                log_error "Unknown target: $TARGET"
                return 1
                ;;
        esac
    fi

    log_success "Deployment complete!"
}

# ===========================
# Usage
# ===========================

usage() {
    cat <<EOF
Usage: $SCRIPT_NAME [OPTIONS]

Automated model deployment script.

OPTIONS:
    -e, --environment ENV    Target environment (dev, staging, prod)
    -t, --target TARGET      Deployment target (docker, k8s, cloud)
    -m, --model PATH         Model file path
    -v, --version VERSION    Model version
    -i, --image NAME         Docker image name (default: ml-model-api)
    -n, --namespace NS       Kubernetes namespace (default: default)
    --replicas N             Number of replicas (default: 3)
    --validate               Validate deployment
    --rollback               Rollback to previous version
    --dry-run                Dry run mode
    -h, --help               Display this help message

EXAMPLES:
    # Deploy to Docker
    $SCRIPT_NAME -t docker -v v1.0 -m models/model.pt --validate

    # Deploy to Kubernetes
    $SCRIPT_NAME -t k8s -v v1.0 -n production --replicas 5

    # Rollback deployment
    $SCRIPT_NAME -t k8s --rollback

EOF
}

# ===========================
# Argument Parsing
# ===========================

parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -t|--target)
                TARGET="$2"
                shift 2
                ;;
            -m|--model)
                MODEL_PATH="$2"
                shift 2
                ;;
            -v|--version)
                MODEL_VERSION="$2"
                shift 2
                ;;
            -i|--image)
                IMAGE_NAME="$2"
                shift 2
                ;;
            -n|--namespace)
                NAMESPACE="$2"
                shift 2
                ;;
            --replicas)
                REPLICAS="$2"
                shift 2
                ;;
            --validate)
                VALIDATE=true
                shift
                ;;
            --rollback)
                ROLLBACK=true
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done
}

# ===========================
# Main
# ===========================

main() {
    parse_arguments "$@"

    # Validate required parameters
    if [[ -z "$MODEL_VERSION" && "$ROLLBACK" == false ]]; then
        log_error "Model version is required"
        usage
        exit 1
    fi

    main_deploy
}

main "$@"
