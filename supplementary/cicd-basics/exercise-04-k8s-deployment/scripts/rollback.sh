#!/bin/bash

# rollback.sh - Rollback ML API deployment to a previous revision
# Usage: ./rollback.sh [OPTIONS]
#
# Options:
#   --environment, -e    Environment (dev, staging, prod) [default: dev]
#   --namespace, -n      Kubernetes namespace [default: ml-api-{environment}]
#   --revision, -r       Revision number to rollback to (0 = previous) [default: 0]
#   --dry-run           Perform dry-run without actual rollback
#   --wait              Wait for rollback to complete [default: true]
#   --timeout           Timeout for rollback [default: 5m]
#   --history           Show release history and exit
#   --help, -h          Show this help message

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
ENVIRONMENT="dev"
NAMESPACE=""
REVISION=0
DRY_RUN=false
WAIT=true
TIMEOUT="5m"
SHOW_HISTORY=false

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

log_blue() {
    echo -e "${BLUE}$1${NC}"
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
        -n|--namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        -r|--revision)
            REVISION="$2"
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
        --history)
            SHOW_HISTORY=true
            shift
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

# Check if release exists
if ! helm list -n "$NAMESPACE" | grep -q ml-api; then
    log_error "Helm release 'ml-api' not found in namespace $NAMESPACE"
    exit 1
fi

# Show history if requested
if [[ "$SHOW_HISTORY" == true ]]; then
    log_info "Helm release history for ml-api in namespace $NAMESPACE:"
    helm history ml-api -n "$NAMESPACE"
    exit 0
fi

# Show current release info
log_info "Current release information:"
helm list -n "$NAMESPACE" | grep ml-api

echo ""
log_info "Release history:"
helm history ml-api -n "$NAMESPACE"

# Confirm rollback (unless dry-run)
if [[ "$DRY_RUN" == false ]]; then
    echo ""
    log_warn "You are about to rollback ml-api in $ENVIRONMENT environment"
    if [[ "$REVISION" == "0" ]]; then
        log_warn "This will rollback to the previous revision"
    else
        log_warn "This will rollback to revision $REVISION"
    fi

    if [[ "$ENVIRONMENT" == "prod" ]]; then
        log_error "PRODUCTION ROLLBACK - Are you sure? (yes/no)"
        read -r confirmation
        if [[ "$confirmation" != "yes" ]]; then
            log_info "Rollback cancelled"
            exit 0
        fi
    else
        echo "Continue? (yes/no)"
        read -r confirmation
        if [[ "$confirmation" != "yes" ]]; then
            log_info "Rollback cancelled"
            exit 0
        fi
    fi
fi

# Build helm rollback command
HELM_CMD=(
    helm
    rollback
    ml-api
)

if [[ "$REVISION" != "0" ]]; then
    HELM_CMD+=("$REVISION")
fi

HELM_CMD+=(
    --namespace "$NAMESPACE"
)

if [[ "$DRY_RUN" == true ]]; then
    HELM_CMD+=(--dry-run)
    log_info "Performing dry-run rollback..."
else
    log_info "Rolling back ML API..."
fi

if [[ "$WAIT" == true ]]; then
    HELM_CMD+=(--wait --timeout "$TIMEOUT")
fi

# Display rollback information
log_info "Rollback details:"
echo "  Environment: $ENVIRONMENT"
echo "  Namespace: $NAMESPACE"
echo "  Revision: $REVISION (0 = previous)"
echo "  Dry run: $DRY_RUN"
echo ""

# Execute helm rollback
log_info "Executing: ${HELM_CMD[*]}"
if "${HELM_CMD[@]}"; then
    log_info "Rollback completed successfully"
else
    log_error "Rollback failed"
    exit 1
fi

# Skip post-rollback checks for dry-run
if [[ "$DRY_RUN" == true ]]; then
    exit 0
fi

# Post-rollback checks
log_info "Performing post-rollback checks..."

# Wait a bit for rollback to propagate
sleep 5

# Check deployment status
log_info "Checking deployment status..."
kubectl rollout status deployment/ml-api -n "$NAMESPACE" --timeout="${TIMEOUT}"

# Show current revision
log_info "Current deployment revision:"
kubectl get deployment ml-api -n "$NAMESPACE" -o jsonpath='{.metadata.annotations.deployment\.kubernetes\.io/revision}'
echo ""

# Show pods
log_info "Current pods:"
kubectl get pods -n "$NAMESPACE" -l "app.kubernetes.io/name=ml-api"

# Show updated release info
echo ""
log_info "Updated release information:"
helm list -n "$NAMESPACE" | grep ml-api

log_info "Rollback complete!"
log_info ""
log_info "To check logs:"
echo "  kubectl logs -f deployment/ml-api -n $NAMESPACE"
