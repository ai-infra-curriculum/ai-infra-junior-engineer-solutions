#!/bin/bash
# Environment Setup Script for ML Platform Autoscaling
# Installs metrics-server, Prometheus, Prometheus Adapter, VPA, and KEDA

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

print_header() {
    echo ""
    echo "=========================================="
    echo "$1"
    echo "=========================================="
    echo ""
}

# Check prerequisites
check_prerequisites() {
    print_header "Checking Prerequisites"

    # Check kubectl
    if command -v kubectl &>/dev/null; then
        log_success "kubectl is installed: $(kubectl version --client --short 2>/dev/null | head -n1)"
    else
        log_error "kubectl is not installed"
        exit 1
    fi

    # Check Helm
    if command -v helm &>/dev/null; then
        log_success "Helm is installed: $(helm version --short)"
    else
        log_error "Helm is not installed"
        exit 1
    fi

    # Check cluster connection
    if kubectl cluster-info &>/dev/null; then
        log_success "Connected to Kubernetes cluster"
        kubectl cluster-info | head -n 1
    else
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi

    echo ""
}

# Install metrics-server
install_metrics_server() {
    print_header "Installing Metrics Server"

    if kubectl get deployment metrics-server -n kube-system &>/dev/null; then
        log_warning "Metrics server already installed"
        return 0
    fi

    log_info "Applying metrics-server manifest..."
    kubectl apply -f ../manifests/01-metrics-server.yaml

    log_info "Waiting for metrics-server to be ready..."
    kubectl wait --for=condition=ready pod -l k8s-app=metrics-server -n kube-system --timeout=120s

    # Verify metrics are available
    log_info "Verifying metrics..."
    sleep 10  # Give metrics-server time to collect data

    if kubectl top nodes &>/dev/null; then
        log_success "Metrics server is working correctly"
        kubectl top nodes
    else
        log_warning "Metrics server installed but metrics not yet available"
        log_info "This is normal for new installations. Wait 1-2 minutes and try: kubectl top nodes"
    fi

    echo ""
}

# Create namespace
create_namespace() {
    print_header "Creating ML Platform Namespace"

    if kubectl get namespace ml-platform &>/dev/null; then
        log_warning "Namespace 'ml-platform' already exists"
        return 0
    fi

    log_info "Creating namespace with resource quotas..."
    kubectl apply -f ../manifests/00-namespace.yaml

    log_success "Namespace 'ml-platform' created"
    kubectl describe namespace ml-platform
    echo ""
}

# Install Prometheus
install_prometheus() {
    print_header "Installing Prometheus"

    # Add Helm repo
    log_info "Adding Prometheus Helm repository..."
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    helm repo update

    # Check if already installed
    if helm list -n monitoring | grep -q prometheus; then
        log_warning "Prometheus already installed"
        return 0
    fi

    log_info "Installing Prometheus..."
    helm install prometheus prometheus-community/prometheus \
        --namespace monitoring \
        --create-namespace \
        --set server.persistentVolume.enabled=false \
        --set alertmanager.enabled=false \
        --set pushgateway.enabled=false \
        --set nodeExporter.enabled=true \
        --set kubeStateMetrics.enabled=true \
        --wait \
        --timeout 5m

    log_success "Prometheus installed successfully"

    log_info "Verifying Prometheus installation..."
    kubectl get pods -n monitoring -l app=prometheus

    echo ""
}

# Install Prometheus Adapter
install_prometheus_adapter() {
    print_header "Installing Prometheus Adapter"

    # Check if already installed
    if helm list -n monitoring | grep -q prometheus-adapter; then
        log_warning "Prometheus Adapter already installed"
        return 0
    fi

    log_info "Installing Prometheus Adapter..."
    helm install prometheus-adapter prometheus-community/prometheus-adapter \
        --namespace monitoring \
        --set prometheus.url=http://prometheus-server.monitoring.svc.cluster.local \
        --set prometheus.port=80 \
        --wait \
        --timeout 5m

    log_success "Prometheus Adapter installed successfully"

    # Verify custom metrics API
    log_info "Verifying custom metrics API..."
    sleep 10
    if kubectl get apiservices | grep -q custom.metrics; then
        log_success "Custom metrics API is available"
        kubectl get apiservices | grep custom.metrics
    else
        log_warning "Custom metrics API not yet available"
    fi

    echo ""
}

# Install VPA
install_vpa() {
    print_header "Installing Vertical Pod Autoscaler"

    # Check if VPA CRD exists
    if kubectl get crd verticalpodautoscalers.autoscaling.k8s.io &>/dev/null; then
        log_warning "VPA already installed"
        return 0
    fi

    log_info "Cloning VPA repository..."
    local tmp_dir=$(mktemp -d)
    cd "$tmp_dir"

    git clone --depth 1 https://github.com/kubernetes/autoscaler.git
    cd autoscaler/vertical-pod-autoscaler

    log_info "Installing VPA..."
    ./hack/vpa-up.sh

    cd - > /dev/null
    rm -rf "$tmp_dir"

    log_success "VPA installed successfully"

    log_info "Verifying VPA components..."
    kubectl get pods -n kube-system | grep vpa

    echo ""
}

# Install KEDA
install_keda() {
    print_header "Installing KEDA"

    # Add Helm repo
    log_info "Adding KEDA Helm repository..."
    helm repo add kedacore https://kedacore.github.io/charts
    helm repo update

    # Check if already installed
    if helm list -n keda | grep -q keda; then
        log_warning "KEDA already installed"
        return 0
    fi

    log_info "Installing KEDA..."
    helm install keda kedacore/keda \
        --namespace keda \
        --create-namespace \
        --wait \
        --timeout 5m

    log_success "KEDA installed successfully"

    log_info "Verifying KEDA installation..."
    kubectl get pods -n keda

    echo ""
}

# Deploy sample workloads
deploy_sample_workloads() {
    print_header "Deploying Sample Workloads"

    log_info "Deploying HPA-based workloads..."
    kubectl apply -f ../manifests/10-model-serving-hpa.yaml
    kubectl apply -f ../manifests/11-model-serving-custom-metrics.yaml

    log_info "Deploying VPA-based workloads..."
    kubectl apply -f ../manifests/20-training-job-vpa.yaml

    log_info "Deploying KEDA-based workloads..."
    kubectl apply -f ../manifests/30-keda-scalers.yaml

    log_info "Deploying safety and governance..."
    kubectl apply -f ../manifests/40-safety-governance.yaml

    log_success "Sample workloads deployed"

    log_info "Waiting for workloads to be ready..."
    sleep 10

    echo ""
    log_info "Current deployments:"
    kubectl get deployments -n ml-platform

    echo ""
    log_info "Current HPAs:"
    kubectl get hpa -n ml-platform

    echo ""
    log_info "Current VPAs:"
    kubectl get vpa -n ml-platform 2>/dev/null || log_warning "No VPAs found (VPA may not be installed)"

    echo ""
    log_info "Current KEDA ScaledObjects:"
    kubectl get scaledobject -n ml-platform 2>/dev/null || log_warning "No ScaledObjects found (KEDA may not be installed)"

    echo ""
}

# Print summary
print_summary() {
    print_header "Installation Summary"

    log_info "Installed Components:"

    # Metrics Server
    if kubectl get deployment metrics-server -n kube-system &>/dev/null; then
        echo "  ✅ Metrics Server"
    else
        echo "  ❌ Metrics Server"
    fi

    # Prometheus
    if kubectl get deployment prometheus-server -n monitoring &>/dev/null; then
        echo "  ✅ Prometheus"
    else
        echo "  ❌ Prometheus"
    fi

    # Prometheus Adapter
    if kubectl get deployment prometheus-adapter -n monitoring &>/dev/null; then
        echo "  ✅ Prometheus Adapter"
    else
        echo "  ❌ Prometheus Adapter"
    fi

    # VPA
    if kubectl get crd verticalpodautoscalers.autoscaling.k8s.io &>/dev/null; then
        echo "  ✅ Vertical Pod Autoscaler"
    else
        echo "  ❌ Vertical Pod Autoscaler"
    fi

    # KEDA
    if kubectl get crd scaledobjects.keda.sh &>/dev/null; then
        echo "  ✅ KEDA"
    else
        echo "  ❌ KEDA"
    fi

    echo ""
    log_info "Namespaces:"
    echo "  - ml-platform: Application workloads"
    echo "  - monitoring: Prometheus and adapters"
    echo "  - keda: KEDA operator"

    echo ""
    log_success "Environment setup complete!"

    echo ""
    echo "Next Steps:"
    echo "1. Verify metrics are available:"
    echo "   kubectl top nodes"
    echo "   kubectl top pods -n ml-platform"
    echo ""
    echo "2. Check autoscaling resources:"
    echo "   kubectl get hpa,vpa,scaledobject -n ml-platform"
    echo ""
    echo "3. Run test suite:"
    echo "   ./scripts/test_autoscaling.sh"
    echo ""
    echo "4. Monitor autoscaling:"
    echo "   watch kubectl get hpa,pods -n ml-platform"
    echo ""
    echo "5. Generate load and observe scaling:"
    echo "   ./scripts/load_generator.sh fraud-detector-api 1000"
    echo ""
}

# Main installation flow
main() {
    print_header "ML Platform Autoscaling Environment Setup"

    check_prerequisites
    install_metrics_server
    create_namespace
    install_prometheus
    install_prometheus_adapter
    install_vpa
    install_keda
    deploy_sample_workloads
    print_summary
}

# Run main function
main "$@"
