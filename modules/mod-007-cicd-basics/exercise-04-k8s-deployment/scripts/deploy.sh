#!/bin/bash

# deploy.sh - Deploy ML API to Kubernetes using Helm
# Usage: ./deploy.sh [OPTIONS]
#
# Options:
#   --environment, -e    Environment to deploy to (dev, staging, prod) [default: dev]
#   --image-tag, -t      Docker image tag to deploy [default: latest]
#   --namespace, -n      Kubernetes namespace [default: ml-api-{environment}]
#   --dry-run           Perform dry-run without actual deployment
#   --upgrade           Upgrade existing release (default behavior)
#   --install           Install new release
#   --wait              Wait for deployment to complete
#   --timeout           Timeout for deployment [default: 5m]
#   --help, -h          Show this help message

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
ENVIRONMENT="dev"
IMAGE_TAG="latest"
NAMESPACE=""
DRY_RUN=false
WAIT=true
TIMEOUT="5m"
HELM_ACTION="upgrade --install"

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHART_DIR="${SCRIPT_DIR}/../helm-chart/ml-api"

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Help function
show_help() {
    grep '^#' "$0" | grep -v '#!/bin/bash' | sed 's/^# //g'
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -t|--image-tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        -n|--namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --no-wait)
            WAIT=false
            shift
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            ;;
    esac
done

# Validate environment
if [[ ! "$ENVIRONMENT" =~ ^(dev|staging|prod)$ ]]; then
    log_error "Invalid environment: $ENVIRONMENT. Must be dev, staging, or prod."
    exit 1
fi

# Set namespace if not provided
if [[ -z "$NAMESPACE" ]]; then
    if [[ "$ENVIRONMENT" == "prod" ]]; then
        NAMESPACE="ml-api"
    else
        NAMESPACE="ml-api-${ENVIRONMENT}"
    fi
fi

# Check prerequisites
log_info "Checking prerequisites..."

if ! command -v kubectl &> /dev/null; then
    log_error "kubectl is not installed"
    exit 1
fi

if ! command -v helm &> /dev/null; then
    log_error "helm is not installed"
    exit 1
fi

# Check kubectl connection
if ! kubectl cluster-info &> /dev/null; then
    log_error "Cannot connect to Kubernetes cluster"
    exit 1
fi

log_info "Prerequisites check passed"

# Check if chart directory exists
if [[ ! -d "$CHART_DIR" ]]; then
    log_error "Helm chart directory not found: $CHART_DIR"
    exit 1
fi

# Determine values file
VALUES_FILE="${CHART_DIR}/values-${ENVIRONMENT}.yaml"
if [[ ! -f "$VALUES_FILE" ]]; then
    log_warn "Environment-specific values file not found: $VALUES_FILE"
    log_info "Using default values.yaml"
    VALUES_FILE="${CHART_DIR}/values.yaml"
fi

# Build helm command
HELM_CMD=(
    helm
    upgrade
    --install
    ml-api
    "$CHART_DIR"
    --namespace "$NAMESPACE"
    --create-namespace
    --values "$VALUES_FILE"
    --set "image.tag=${IMAGE_TAG}"
)

if [[ "$DRY_RUN" == true ]]; then
    HELM_CMD+=(--dry-run --debug)
    log_info "Performing dry-run deployment..."
else
    log_info "Deploying ML API..."
fi

if [[ "$WAIT" == true ]]; then
    HELM_CMD+=(--wait --timeout "$TIMEOUT")
fi

# Display deployment information
log_info "Deployment details:"
echo "  Environment: $ENVIRONMENT"
echo "  Namespace: $NAMESPACE"
echo "  Image tag: $IMAGE_TAG"
echo "  Values file: $VALUES_FILE"
echo "  Dry run: $DRY_RUN"
echo ""

# Execute helm command
log_info "Executing: ${HELM_CMD[*]}"
if "${HELM_CMD[@]}"; then
    log_info "Deployment completed successfully"
else
    log_error "Deployment failed"
    exit 1
fi

# Skip post-deployment checks for dry-run
if [[ "$DRY_RUN" == true ]]; then
    exit 0
fi

# Post-deployment checks
log_info "Performing post-deployment checks..."

# Wait a bit for resources to be created
sleep 5

# Check deployment status
log_info "Checking deployment status..."
if kubectl get deployment ml-api -n "$NAMESPACE" &> /dev/null; then
    kubectl rollout status deployment/ml-api -n "$NAMESPACE" --timeout="${TIMEOUT}"
else
    log_warn "Deployment not found in namespace $NAMESPACE"
fi

# Show pods
log_info "Pods in namespace $NAMESPACE:"
kubectl get pods -n "$NAMESPACE" -l "app.kubernetes.io/name=ml-api"

# Show service
log_info "Services in namespace $NAMESPACE:"
kubectl get svc -n "$NAMESPACE" -l "app.kubernetes.io/name=ml-api"

# Show ingress
if kubectl get ingress -n "$NAMESPACE" &> /dev/null; then
    log_info "Ingress in namespace $NAMESPACE:"
    kubectl get ingress -n "$NAMESPACE" -l "app.kubernetes.io/name=ml-api"
fi

# Show Helm release info
log_info "Helm release information:"
helm list -n "$NAMESPACE" | grep ml-api || true

log_info "Deployment complete!"
log_info ""
log_info "To check logs:"
echo "  kubectl logs -f deployment/ml-api -n $NAMESPACE"
log_info ""
log_info "To access the application:"
echo "  kubectl port-forward svc/ml-api 8000:80 -n $NAMESPACE"
