#!/bin/bash
# Comprehensive Autoscaling Test Suite for ML Platform
# Tests HPA, VPA, KEDA, and PDB configurations

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
NAMESPACE="${NAMESPACE:-ml-platform}"
MONITORING_NS="${MONITORING_NS:-monitoring}"
LOAD_DURATION="${LOAD_DURATION:-120}"  # seconds
SCALE_WAIT_TIME="${SCALE_WAIT_TIME:-90}"  # seconds

# Helper functions
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

print_header() {
    echo ""
    echo "=========================================="
    echo "$1"
    echo "=========================================="
    echo ""
}

check_resource() {
    local resource=$1
    local name=$2
    local namespace=$3

    if kubectl get "$resource" "$name" -n "$namespace" &>/dev/null; then
        log_success "$resource '$name' exists in namespace '$namespace'"
        return 0
    else
        log_error "$resource '$name' not found in namespace '$namespace'"
        return 1
    fi
}

wait_for_condition() {
    local resource=$1
    local name=$2
    local condition=$3
    local namespace=$4
    local timeout=$5

    log_info "Waiting for $resource/$name to be $condition (timeout: ${timeout}s)..."

    if kubectl wait --for=condition="$condition" "$resource/$name" -n "$namespace" --timeout="${timeout}s" &>/dev/null; then
        log_success "$resource/$name is $condition"
        return 0
    else
        log_warning "$resource/$name did not become $condition within ${timeout}s"
        return 1
    fi
}

# Main test suite
main() {
    print_header "ML Platform Autoscaling Test Suite"

    log_info "Test Configuration:"
    echo "  Namespace: $NAMESPACE"
    echo "  Monitoring Namespace: $MONITORING_NS"
    echo "  Load Duration: ${LOAD_DURATION}s"
    echo "  Scale Wait Time: ${SCALE_WAIT_TIME}s"
    echo ""

    # Test 1: Prerequisites
    test_prerequisites

    # Test 2: HPA CPU-based scaling
    test_hpa_cpu_scaling

    # Test 3: HPA custom metrics (if Prometheus available)
    if kubectl get svc prometheus-server -n "$MONITORING_NS" &>/dev/null; then
        test_hpa_custom_metrics
    else
        log_warning "Prometheus not available, skipping custom metrics test"
    fi

    # Test 4: VPA recommendations
    test_vpa_recommendations

    # Test 5: KEDA scaling (if KEDA available)
    if kubectl get crd scaledobjects.keda.sh &>/dev/null; then
        test_keda_scaling
    else
        log_warning "KEDA not installed, skipping KEDA tests"
    fi

    # Test 6: Pod Disruption Budgets
    test_pod_disruption_budgets

    # Test 7: Resource utilization
    test_resource_utilization

    # Summary
    print_test_summary
}

test_prerequisites() {
    print_header "Test 1: Prerequisites Check"

    local errors=0

    # Check metrics-server
    if kubectl top nodes &>/dev/null; then
        log_success "Metrics server is working"
        kubectl top nodes
    else
        log_error "Metrics server not working"
        ((errors++))
    fi

    echo ""

    # Check namespace
    if kubectl get namespace "$NAMESPACE" &>/dev/null; then
        log_success "Namespace '$NAMESPACE' exists"
    else
        log_error "Namespace '$NAMESPACE' not found"
        ((errors++))
    fi

    echo ""

    # Check HPA
    log_info "Checking HPA resources..."
    kubectl get hpa -n "$NAMESPACE" 2>/dev/null || log_warning "No HPAs found"

    echo ""

    # Check VPA
    log_info "Checking VPA resources..."
    if kubectl get crd verticalpodautoscalers.autoscaling.k8s.io &>/dev/null; then
        kubectl get vpa -n "$NAMESPACE" 2>/dev/null || log_warning "No VPAs found"
    else
        log_warning "VPA CRD not installed"
    fi

    echo ""

    # Check KEDA
    log_info "Checking KEDA resources..."
    if kubectl get crd scaledobjects.keda.sh &>/dev/null; then
        kubectl get scaledobject -n "$NAMESPACE" 2>/dev/null || log_warning "No KEDA ScaledObjects found"
    else
        log_warning "KEDA not installed"
    fi

    if [ $errors -gt 0 ]; then
        log_error "Prerequisites check failed with $errors errors"
        return 1
    else
        log_success "All prerequisites passed"
        return 0
    fi
}

test_hpa_cpu_scaling() {
    print_header "Test 2: HPA CPU-based Scaling"

    local hpa_name="fraud-detector-hpa"
    local deployment_name="fraud-detector-api"

    # Check if HPA exists
    if ! check_resource "hpa" "$hpa_name" "$NAMESPACE"; then
        log_warning "Skipping test - HPA not found"
        return 1
    fi

    # Check current state
    log_info "Current HPA status:"
    kubectl get hpa "$hpa_name" -n "$NAMESPACE"
    echo ""

    log_info "Current pods:"
    kubectl get pods -n "$NAMESPACE" -l app=fraud-detector
    local initial_replicas=$(kubectl get deployment "$deployment_name" -n "$NAMESPACE" -o jsonpath='{.status.replicas}')
    log_info "Initial replicas: $initial_replicas"
    echo ""

    # Generate load
    log_info "Generating CPU load for ${LOAD_DURATION}s..."
    kubectl run load-generator-${RANDOM} \
        --image=busybox:latest \
        --restart=Never \
        -n "$NAMESPACE" \
        --rm \
        --attach=false \
        -- /bin/sh -c "while true; do wget -q -O- http://${deployment_name}.${NAMESPACE}.svc.cluster.local; done" &

    local load_pid=$!

    # Wait for scaling
    sleep "$SCALE_WAIT_TIME"

    # Check if scaled
    log_info "HPA status after load:"
    kubectl get hpa "$hpa_name" -n "$NAMESPACE"
    echo ""

    log_info "Pods after scaling:"
    kubectl get pods -n "$NAMESPACE" -l app=fraud-detector
    local scaled_replicas=$(kubectl get deployment "$deployment_name" -n "$NAMESPACE" -o jsonpath='{.status.replicas}')
    log_info "Scaled replicas: $scaled_replicas"
    echo ""

    # Stop load
    log_info "Stopping load generator..."
    kubectl delete pod -n "$NAMESPACE" -l run=load-generator --force --grace-period=0 2>/dev/null || true
    kill $load_pid 2>/dev/null || true

    # Verify scaling occurred
    if [ "$scaled_replicas" -gt "$initial_replicas" ]; then
        log_success "HPA scaled up from $initial_replicas to $scaled_replicas replicas"
    else
        log_warning "HPA did not scale up (still at $initial_replicas replicas)"
    fi

    # Check events
    log_info "Recent HPA events:"
    kubectl describe hpa "$hpa_name" -n "$NAMESPACE" | tail -n 20
}

test_hpa_custom_metrics() {
    print_header "Test 3: HPA Custom Metrics Scaling"

    local hpa_name="recommendation-hpa"

    if ! check_resource "hpa" "$hpa_name" "$NAMESPACE"; then
        log_warning "Skipping test - HPA not found"
        return 1
    fi

    log_info "Checking custom metrics availability..."
    if kubectl get --raw "/apis/custom.metrics.k8s.io/v1beta1" &>/dev/null; then
        log_success "Custom metrics API is available"

        # List available metrics
        log_info "Available custom metrics:"
        kubectl get --raw "/apis/custom.metrics.k8s.io/v1beta1" | jq -r '.resources[].name' | head -n 10
    else
        log_error "Custom metrics API not available"
        return 1
    fi

    echo ""
    log_info "Current HPA status with custom metrics:"
    kubectl describe hpa "$hpa_name" -n "$NAMESPACE" | grep -A 20 "Metrics:"
}

test_vpa_recommendations() {
    print_header "Test 4: VPA Recommendations"

    local vpa_name="model-training-vpa"

    # Check if VPA CRD exists
    if ! kubectl get crd verticalpodautoscalers.autoscaling.k8s.io &>/dev/null; then
        log_warning "Skipping test - VPA not installed"
        return 1
    fi

    if ! check_resource "vpa" "$vpa_name" "$NAMESPACE"; then
        log_warning "Skipping test - VPA not found"
        return 1
    fi

    log_info "VPA status for $vpa_name:"
    kubectl get vpa "$vpa_name" -n "$NAMESPACE"
    echo ""

    log_info "VPA recommendations:"
    kubectl describe vpa "$vpa_name" -n "$NAMESPACE" | grep -A 30 "Recommendation:"
    echo ""

    # Check if recommendations are available
    local has_recommendation=$(kubectl get vpa "$vpa_name" -n "$NAMESPACE" -o jsonpath='{.status.recommendation}')
    if [ -n "$has_recommendation" ]; then
        log_success "VPA has generated recommendations"

        # Extract target recommendations
        local target_cpu=$(kubectl get vpa "$vpa_name" -n "$NAMESPACE" -o jsonpath='{.status.recommendation.containerRecommendations[0].target.cpu}')
        local target_memory=$(kubectl get vpa "$vpa_name" -n "$NAMESPACE" -o jsonpath='{.status.recommendation.containerRecommendations[0].target.memory}')

        log_info "Target recommendations:"
        echo "  CPU: $target_cpu"
        echo "  Memory: $target_memory"
    else
        log_warning "VPA has not generated recommendations yet (may need more time)"
    fi
}

test_keda_scaling() {
    print_header "Test 5: KEDA Scaling"

    log_info "KEDA ScaledObjects:"
    kubectl get scaledobject -n "$NAMESPACE"
    echo ""

    # Check specific ScaledObject
    local scaler_name="batch-inference-scaler"
    if check_resource "scaledobject" "$scaler_name" "$NAMESPACE"; then
        log_info "ScaledObject details:"
        kubectl describe scaledobject "$scaler_name" -n "$NAMESPACE" | head -n 40
    fi
}

test_pod_disruption_budgets() {
    print_header "Test 6: Pod Disruption Budgets"

    log_info "Pod Disruption Budgets:"
    kubectl get pdb -n "$NAMESPACE"
    echo ""

    # Check each PDB
    for pdb in $(kubectl get pdb -n "$NAMESPACE" -o name); do
        pdb_name=$(basename "$pdb")
        log_info "PDB: $pdb_name"
        kubectl get pdb "$pdb_name" -n "$NAMESPACE" -o json | jq -r '.status | "  Current Healthy: \(.currentHealthy), Desired Healthy: \(.desiredHealthy), Disruptions Allowed: \(.disruptionsAllowed)"'
        echo ""
    done
}

test_resource_utilization() {
    print_header "Test 7: Resource Utilization"

    log_info "Node resource utilization:"
    kubectl top nodes
    echo ""

    log_info "Pod resource utilization (CPU):"
    kubectl top pods -n "$NAMESPACE" --sort-by=cpu | head -n 15
    echo ""

    log_info "Pod resource utilization (Memory):"
    kubectl top pods -n "$NAMESPACE" --sort-by=memory | head -n 15
    echo ""

    # Resource requests vs limits
    log_info "Resource requests and limits:"
    kubectl get pods -n "$NAMESPACE" -o custom-columns=\
NAME:.metadata.name,\
CPU_REQ:.spec.containers[0].resources.requests.cpu,\
CPU_LIM:.spec.containers[0].resources.limits.cpu,\
MEM_REQ:.spec.containers[0].resources.requests.memory,\
MEM_LIM:.spec.containers[0].resources.limits.memory \
    | head -n 15
}

print_test_summary() {
    print_header "Test Summary"

    log_info "HPA Status:"
    kubectl get hpa -n "$NAMESPACE" 2>/dev/null || echo "No HPAs found"
    echo ""

    if kubectl get crd verticalpodautoscalers.autoscaling.k8s.io &>/dev/null; then
        log_info "VPA Status:"
        kubectl get vpa -n "$NAMESPACE" 2>/dev/null || echo "No VPAs found"
        echo ""
    fi

    if kubectl get crd scaledobjects.keda.sh &>/dev/null; then
        log_info "KEDA ScaledObjects:"
        kubectl get scaledobject -n "$NAMESPACE" 2>/dev/null || echo "No ScaledObjects found"
        echo ""
    fi

    log_info "Pod Disruption Budgets:"
    kubectl get pdb -n "$NAMESPACE" 2>/dev/null || echo "No PDBs found"
    echo ""

    log_success "Test suite completed!"
    echo ""
    echo "Next steps:"
    echo "1. Review HPA scaling events: kubectl describe hpa -n $NAMESPACE"
    echo "2. Check VPA recommendations: kubectl describe vpa -n $NAMESPACE"
    echo "3. Monitor KEDA scaling: kubectl get scaledobject -n $NAMESPACE --watch"
    echo "4. View autoscaling events: kubectl get events -n $NAMESPACE --sort-by='.lastTimestamp'"
}

# Run main function
main "$@"
