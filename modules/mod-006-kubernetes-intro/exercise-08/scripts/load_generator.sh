#!/bin/bash
# Load Generator Script for Testing Autoscaling
# Generates HTTP load to trigger HPA scaling

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Configuration
SERVICE_NAME="${1:-fraud-detector-api}"
NUM_REQUESTS="${2:-1000}"
CONCURRENCY="${3:-10}"
NAMESPACE="${4:-ml-platform}"

usage() {
    cat <<EOF
Usage: $0 [SERVICE_NAME] [NUM_REQUESTS] [CONCURRENCY] [NAMESPACE]

Arguments:
  SERVICE_NAME    Name of the service to load test (default: fraud-detector-api)
  NUM_REQUESTS    Total number of requests to send (default: 1000)
  CONCURRENCY     Number of concurrent clients (default: 10)
  NAMESPACE       Kubernetes namespace (default: ml-platform)

Examples:
  $0 fraud-detector-api 1000 10 ml-platform
  $0 recommendation-api 5000 20
  $0 fraud-detector-api

EOF
    exit 1
}

# Check if service exists
check_service() {
    if ! kubectl get svc "$SERVICE_NAME" -n "$NAMESPACE" &>/dev/null; then
        log_error "Service '$SERVICE_NAME' not found in namespace '$NAMESPACE'"
        exit 1
    fi
    log_success "Service '$SERVICE_NAME' found in namespace '$NAMESPACE'"
}

# Get initial state
get_initial_state() {
    log_info "Capturing initial state..."

    # Get deployment name from service selector
    local deployment=$(kubectl get svc "$SERVICE_NAME" -n "$NAMESPACE" -o jsonpath='{.metadata.labels.app}')

    if [ -z "$deployment" ]; then
        log_warning "Could not determine deployment from service"
        return
    fi

    echo ""
    log_info "Initial HPA status:"
    kubectl get hpa -n "$NAMESPACE" -l app="$deployment" 2>/dev/null || echo "No HPA found"

    echo ""
    log_info "Initial pod count:"
    kubectl get pods -n "$NAMESPACE" -l app="$deployment" --no-headers | wc -l

    echo ""
    log_info "Initial resource utilization:"
    kubectl top pods -n "$NAMESPACE" -l app="$deployment" 2>/dev/null || echo "Metrics not yet available"

    echo ""
}

# Generate load using busybox
generate_load_busybox() {
    log_info "Generating load using busybox (simple method)..."
    log_info "Service: $SERVICE_NAME"
    log_info "Requests: $NUM_REQUESTS"
    log_info "Concurrency: $CONCURRENCY"
    log_info "Duration: approximately $((NUM_REQUESTS / CONCURRENCY / 10)) seconds"

    local service_url="http://${SERVICE_NAME}.${NAMESPACE}.svc.cluster.local"

    # Create multiple load generator pods
    for i in $(seq 1 "$CONCURRENCY"); do
        log_info "Starting load generator $i/$CONCURRENCY..."

        kubectl run "load-gen-${i}-${RANDOM}" \
            --image=busybox:latest \
            --restart=Never \
            -n "$NAMESPACE" \
            --labels="load-test=true" \
            --rm \
            --attach=false \
            -- /bin/sh -c "
                echo 'Load generator $i started'
                count=0
                max=$((NUM_REQUESTS / CONCURRENCY))
                while [ \$count -lt \$max ]; do
                    wget -q -O- $service_url &>/dev/null
                    count=\$((count + 1))
                    if [ \$((count % 100)) -eq 0 ]; then
                        echo 'Completed \$count requests'
                    fi
                done
                echo 'Load generator $i completed \$count requests'
            " &
    done

    log_success "Started $CONCURRENCY load generators"
}

# Generate load using hey (if available)
generate_load_hey() {
    log_info "Generating load using 'hey' tool..."
    log_info "Service: $SERVICE_NAME"
    log_info "Requests: $NUM_REQUESTS"
    log_info "Concurrency: $CONCURRENCY"

    local service_url="http://${SERVICE_NAME}.${NAMESPACE}.svc.cluster.local"

    kubectl run "load-hey-${RANDOM}" \
        --image=williamyeh/hey:latest \
        --restart=Never \
        -n "$NAMESPACE" \
        --labels="load-test=true" \
        --rm \
        --attach=true \
        -- -n "$NUM_REQUESTS" -c "$CONCURRENCY" "$service_url"
}

# Monitor scaling
monitor_scaling() {
    log_info "Monitoring autoscaling..."

    local deployment=$(kubectl get svc "$SERVICE_NAME" -n "$NAMESPACE" -o jsonpath='{.metadata.labels.app}')

    if [ -z "$deployment" ]; then
        log_warning "Cannot monitor - deployment not found"
        return
    fi

    # Monitor for 2 minutes
    for i in {1..24}; do
        echo ""
        echo "==================== Check $i/24 ===================="

        echo ""
        echo "HPA Status:"
        kubectl get hpa -n "$NAMESPACE" -l app="$deployment" 2>/dev/null || echo "No HPA"

        echo ""
        echo "Pod Count:"
        kubectl get pods -n "$NAMESPACE" -l app="$deployment" --no-headers | wc -l

        echo ""
        echo "Resource Utilization:"
        kubectl top pods -n "$NAMESPACE" -l app="$deployment" 2>/dev/null | head -n 6 || echo "Metrics not available"

        sleep 5
    done
}

# Get final state
get_final_state() {
    log_info "Capturing final state..."

    local deployment=$(kubectl get svc "$SERVICE_NAME" -n "$NAMESPACE" -o jsonpath='{.metadata.labels.app}')

    if [ -z "$deployment" ]; then
        log_warning "Could not determine deployment from service"
        return
    fi

    echo ""
    log_info "Final HPA status:"
    kubectl get hpa -n "$NAMESPACE" -l app="$deployment" 2>/dev/null || echo "No HPA found"

    echo ""
    log_info "Final pod count:"
    kubectl get pods -n "$NAMESPACE" -l app="$deployment" --no-headers | wc -l

    echo ""
    log_info "Final resource utilization:"
    kubectl top pods -n "$NAMESPACE" -l app="$deployment" 2>/dev/null || echo "Metrics not yet available"

    echo ""
    log_info "Recent HPA events:"
    kubectl describe hpa -n "$NAMESPACE" -l app="$deployment" 2>/dev/null | grep -A 10 "Events:" || echo "No events"

    echo ""
}

# Cleanup
cleanup() {
    log_info "Cleaning up load generator pods..."
    kubectl delete pods -n "$NAMESPACE" -l load-test=true --force --grace-period=0 2>/dev/null || true
    log_success "Cleanup complete"
}

# Main function
main() {
    echo "=========================================="
    echo "Load Generator for Autoscaling Testing"
    echo "=========================================="
    echo ""

    # Parse arguments
    if [ "$#" -eq 0 ] || [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
        usage
    fi

    # Setup trap for cleanup
    trap cleanup EXIT

    # Execute load test
    check_service
    get_initial_state
    generate_load_busybox

    log_info "Load generation in progress..."
    log_info "Waiting for scaling to occur (90 seconds)..."
    sleep 90

    monitor_scaling
    get_final_state

    echo ""
    echo "=========================================="
    log_success "Load test completed!"
    echo "=========================================="
    echo ""

    echo "Observations:"
    echo "1. Check if HPA scaled up the deployment"
    echo "2. Verify CPU/memory utilization increased during load"
    echo "3. Wait 5-10 minutes and check if HPA scales back down"
    echo ""
    echo "To monitor scale-down:"
    echo "  watch kubectl get hpa,pods -n $NAMESPACE"
    echo ""
}

main "$@"
