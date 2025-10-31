#!/bin/bash

# test-deployment.sh - Test ML API deployment
# Usage: ./test-deployment.sh [OPTIONS]
#
# Options:
#   --environment, -e    Environment to test (dev, staging, prod) [default: dev]
#   --namespace, -n      Kubernetes namespace [default: ml-api-{environment}]
#   --external-url      External URL to test (overrides port-forward)
#   --skip-port-forward Skip port-forward setup
#   --port              Local port for port-forward [default: 8080]
#   --help, -h          Show this help message

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
ENVIRONMENT="dev"
NAMESPACE=""
EXTERNAL_URL=""
SKIP_PORT_FORWARD=false
LOCAL_PORT=8080
PORT_FORWARD_PID=""

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

log_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

log_fail() {
    echo -e "${RED}[✗]${NC} $1"
}

# Cleanup function
cleanup() {
    if [[ -n "$PORT_FORWARD_PID" ]]; then
        log_info "Cleaning up port-forward (PID: $PORT_FORWARD_PID)..."
        kill "$PORT_FORWARD_PID" 2>/dev/null || true
    fi
}

trap cleanup EXIT

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
        --external-url)
            EXTERNAL_URL="$2"
            SKIP_PORT_FORWARD=true
            shift 2
            ;;
        --skip-port-forward)
            SKIP_PORT_FORWARD=true
            shift
            ;;
        --port)
            LOCAL_PORT="$2"
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

if ! command -v curl &> /dev/null; then
    log_error "curl is not installed"
    exit 1
fi

# Check kubectl connection
if ! kubectl cluster-info &> /dev/null; then
    log_error "Cannot connect to Kubernetes cluster"
    exit 1
fi

log_info "Testing ML API deployment in $ENVIRONMENT environment"
echo ""

# Test 1: Check deployment exists
log_info "Test 1: Checking if deployment exists..."
if kubectl get deployment ml-api -n "$NAMESPACE" &> /dev/null; then
    log_success "Deployment exists"
else
    log_fail "Deployment not found"
    exit 1
fi

# Test 2: Check deployment status
log_info "Test 2: Checking deployment status..."
READY_REPLICAS=$(kubectl get deployment ml-api -n "$NAMESPACE" -o jsonpath='{.status.readyReplicas}')
DESIRED_REPLICAS=$(kubectl get deployment ml-api -n "$NAMESPACE" -o jsonpath='{.spec.replicas}')

if [[ "$READY_REPLICAS" == "$DESIRED_REPLICAS" ]] && [[ "$READY_REPLICAS" -gt 0 ]]; then
    log_success "Deployment is ready ($READY_REPLICAS/$DESIRED_REPLICAS replicas)"
else
    log_fail "Deployment not ready ($READY_REPLICAS/$DESIRED_REPLICAS replicas)"
    exit 1
fi

# Test 3: Check pods are running
log_info "Test 3: Checking pod status..."
POD_COUNT=$(kubectl get pods -n "$NAMESPACE" -l "app.kubernetes.io/name=ml-api" --field-selector=status.phase=Running --no-headers | wc -l)

if [[ "$POD_COUNT" -gt 0 ]]; then
    log_success "$POD_COUNT pod(s) running"
    kubectl get pods -n "$NAMESPACE" -l "app.kubernetes.io/name=ml-api"
else
    log_fail "No pods running"
    exit 1
fi

# Test 4: Check service exists
log_info "Test 4: Checking service..."
if kubectl get svc ml-api -n "$NAMESPACE" &> /dev/null; then
    log_success "Service exists"
else
    log_fail "Service not found"
    exit 1
fi

# Set up URL for testing
if [[ -n "$EXTERNAL_URL" ]]; then
    BASE_URL="$EXTERNAL_URL"
    log_info "Using external URL: $BASE_URL"
elif [[ "$SKIP_PORT_FORWARD" == false ]]; then
    log_info "Setting up port-forward to localhost:$LOCAL_PORT..."
    kubectl port-forward svc/ml-api "$LOCAL_PORT:80" -n "$NAMESPACE" > /dev/null 2>&1 &
    PORT_FORWARD_PID=$!

    # Wait for port-forward to be ready
    sleep 3

    BASE_URL="http://localhost:$LOCAL_PORT"
    log_info "Port-forward ready at $BASE_URL"
else
    log_warn "Skipping endpoint tests (no URL provided)"
    exit 0
fi

# Test 5: Health endpoint
log_info "Test 5: Testing /health endpoint..."
if curl -sf "$BASE_URL/health" > /dev/null; then
    HEALTH_RESPONSE=$(curl -s "$BASE_URL/health")
    log_success "Health endpoint responding"
    echo "  Response: $HEALTH_RESPONSE"
else
    log_fail "Health endpoint not responding"
    exit 1
fi

# Test 6: Ready endpoint
log_info "Test 6: Testing /ready endpoint..."
if curl -sf "$BASE_URL/ready" > /dev/null; then
    READY_RESPONSE=$(curl -s "$BASE_URL/ready")
    log_success "Ready endpoint responding"
    echo "  Response: $READY_RESPONSE"
else
    log_fail "Ready endpoint not responding"
fi

# Test 7: Predict endpoint (if exists)
log_info "Test 7: Testing /predict endpoint..."
PREDICT_PAYLOAD='{"features": [[1.0, 2.0, 3.0, 4.0, 5.0]]}'

if curl -sf -X POST "$BASE_URL/predict" \
    -H "Content-Type: application/json" \
    -d "$PREDICT_PAYLOAD" > /dev/null; then
    PREDICT_RESPONSE=$(curl -s -X POST "$BASE_URL/predict" \
        -H "Content-Type: application/json" \
        -d "$PREDICT_PAYLOAD")
    log_success "Predict endpoint responding"
    echo "  Response: $PREDICT_RESPONSE"
else
    log_warn "Predict endpoint test failed (may not be implemented)"
fi

# Test 8: Check resource usage
log_info "Test 8: Checking resource usage..."
echo ""
kubectl top pods -n "$NAMESPACE" -l "app.kubernetes.io/name=ml-api" 2>/dev/null || log_warn "Metrics not available (requires metrics-server)"

# Test 9: Check logs for errors
log_info "Test 9: Checking recent logs for errors..."
ERROR_COUNT=$(kubectl logs -n "$NAMESPACE" -l "app.kubernetes.io/name=ml-api" --tail=100 | grep -ci "error" || true)

if [[ "$ERROR_COUNT" -eq 0 ]]; then
    log_success "No errors found in recent logs"
else
    log_warn "Found $ERROR_COUNT error message(s) in recent logs"
    echo "  Recent errors:"
    kubectl logs -n "$NAMESPACE" -l "app.kubernetes.io/name=ml-api" --tail=100 | grep -i "error" | head -5
fi

# Summary
echo ""
log_info "========================================="
log_info "Test Summary"
log_info "========================================="
log_info "Environment: $ENVIRONMENT"
log_info "Namespace: $NAMESPACE"
log_info "Replicas: $READY_REPLICAS/$DESIRED_REPLICAS"
log_info "Running pods: $POD_COUNT"
log_success "All critical tests passed!"

exit 0
