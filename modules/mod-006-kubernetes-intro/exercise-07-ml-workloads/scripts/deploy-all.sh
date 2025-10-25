#!/bin/bash

# deploy-all.sh
# Automated deployment script for ML Workloads exercise

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MANIFESTS_DIR="$PROJECT_ROOT/manifests"

# Configuration
NAMESPACE="ml-workloads"

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo ""
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo ""
}

# Function to check if kubectl is available
check_kubectl() {
    if ! command -v kubectl &> /dev/null; then
        print_error "kubectl is not installed or not in PATH"
        exit 1
    fi
    print_success "kubectl is available"
}

# Function to check cluster connection
check_cluster() {
    if ! kubectl cluster-info &> /dev/null; then
        print_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    print_success "Connected to Kubernetes cluster"
}

# Function to check for GPU support
check_gpu_support() {
    print_info "Checking for GPU support..."

    local gpu_nodes=$(kubectl get nodes -o json | \
        jq -r '.items[] | select(.status.capacity["nvidia.com/gpu"]) | .metadata.name' 2>/dev/null)

    if [ -n "$gpu_nodes" ]; then
        local gpu_count=$(echo "$gpu_nodes" | wc -l)
        print_success "Found $gpu_count node(s) with GPU support"
        echo "$gpu_nodes" | while read node; do
            echo "  - $node"
        done
    else
        print_warning "No GPU nodes detected"
        print_info "GPU training jobs will remain pending unless GPUs are available"
    fi
}

# Function to deploy manifests
deploy_manifests() {
    local pattern=$1
    local description=$2

    print_header "$description"

    local files=$(find "$MANIFESTS_DIR" -name "$pattern" | sort)

    if [ -z "$files" ]; then
        print_warning "No files matching pattern '$pattern' found"
        return
    fi

    for file in $files; do
        local filename=$(basename "$file")
        print_info "Deploying $filename..."

        if kubectl apply -f "$file"; then
            print_success "Deployed $filename"
        else
            print_error "Failed to deploy $filename"
            return 1
        fi
    done
}

# Function to wait for storage initialization
wait_for_storage_init() {
    print_info "Waiting for storage initialization job..."

    if kubectl wait --for=condition=complete job/init-model-storage \
        -n "$NAMESPACE" --timeout=120s 2>/dev/null; then
        print_success "Storage initialization complete"
    else
        print_warning "Storage initialization may still be running"
        print_info "Check status with: kubectl get job init-model-storage -n $NAMESPACE"
    fi
}

# Function to wait for deployments
wait_for_deployments() {
    print_info "Waiting for model serving deployments to be ready..."

    if kubectl wait --for=condition=available deployment \
        -l component=model-server -n "$NAMESPACE" --timeout=120s 2>/dev/null; then
        print_success "Model serving deployments are ready"
    else
        print_warning "Some deployments may not be ready yet"
    fi
}

# Function to show deployment summary
show_summary() {
    print_header "Deployment Summary"

    print_info "Namespace: $NAMESPACE"
    kubectl get namespace "$NAMESPACE" 2>/dev/null || true
    echo ""

    print_info "Storage (PVCs):"
    kubectl get pvc -n "$NAMESPACE" 2>/dev/null || print_warning "No PVCs found"
    echo ""

    print_info "Model Serving Deployments:"
    kubectl get deployments -n "$NAMESPACE" -l component=model-server 2>/dev/null || print_warning "No serving deployments found"
    echo ""

    print_info "Training Jobs:"
    kubectl get jobs -n "$NAMESPACE" -l job-type=training 2>/dev/null || print_warning "No training jobs found"
    echo ""

    print_info "Services:"
    kubectl get services -n "$NAMESPACE" 2>/dev/null || print_warning "No services found"
    echo ""

    print_info "Ingress:"
    kubectl get ingress -n "$NAMESPACE" 2>/dev/null || print_warning "No ingress resources found"
    echo ""
}

# Function to test model endpoints
test_endpoints() {
    print_header "Testing Model Endpoints"

    # Test model API server
    print_info "Testing model API server..."
    local api_pod=$(kubectl get pod -n "$NAMESPACE" -l app=model-api-server \
        -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)

    if [ -n "$api_pod" ]; then
        if kubectl exec -n "$NAMESPACE" "$api_pod" -- \
            wget -q -O- http://localhost:8080/health &> /dev/null; then
            print_success "Model API server is responding"
        else
            print_warning "Model API server may not be ready"
        fi
    fi

    # Test TensorFlow Serving
    print_info "Testing TensorFlow Serving..."
    local tf_pod=$(kubectl get pod -n "$NAMESPACE" -l app=tensorflow-serving \
        -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)

    if [ -n "$tf_pod" ]; then
        if kubectl exec -n "$NAMESPACE" "$tf_pod" -- \
            wget -q -O- http://localhost:8501/v1/models/sentiment-classifier &> /dev/null; then
            print_success "TensorFlow Serving is responding"
        else
            print_warning "TensorFlow Serving may not be ready"
        fi
    fi
}

# Function to show next steps
show_next_steps() {
    print_header "Next Steps"

    echo "Deployment complete! Here's how to use the ML workloads:"
    echo ""
    echo "1. Check model storage:"
    echo "   kubectl get pvc -n $NAMESPACE"
    echo ""
    echo "2. View model serving pods:"
    echo "   kubectl get pods -n $NAMESPACE -l component=model-server"
    echo ""
    echo "3. Test model inference:"
    echo "   kubectl run test-client --rm -it --image=curlimages/curl -n $NAMESPACE \\"
    echo "     -- curl -X POST http://model-api-server/predict \\"
    echo "     -H 'Content-Type: application/json' \\"
    echo "     -d '{\"text\": \"This is a test\"}'"
    echo ""
    echo "4. Check training jobs:"
    echo "   kubectl get jobs -n $NAMESPACE"
    echo "   kubectl logs job/model-training-simple -n $NAMESPACE"
    echo ""
    echo "5. Run a training job:"
    echo "   kubectl create job manual-training --from=job/model-training-simple -n $NAMESPACE"
    echo ""
    echo "6. View model versions (A/B testing):"
    echo "   kubectl get deployments -n $NAMESPACE -l app=ml-model"
    echo "   kubectl get ingress -n $NAMESPACE"
    echo ""
    echo "7. Test canary deployment:"
    echo "   for i in {1..10}; do"
    echo "     kubectl run test-\$i --rm --image=curlimages/curl -n $NAMESPACE \\"
    echo "       -- curl http://ml-model-service/predict -d '{\"text\":\"test\"}'"
    echo "   done"
    echo ""
    echo "8. Monitor with Prometheus metrics:"
    echo "   kubectl port-forward -n $NAMESPACE deployment/model-v1 8080:8080"
    echo "   curl http://localhost:8080/metrics"
    echo ""
    echo "9. Run model comparison:"
    echo "   kubectl create job comparison --from=job/model-comparison -n $NAMESPACE"
    echo "   kubectl logs job/comparison -n $NAMESPACE"
    echo ""
    echo "10. Scale model serving:"
    echo "    kubectl scale deployment model-api-server -n $NAMESPACE --replicas=5"
    echo ""
    echo "11. View HPA status:"
    echo "    kubectl get hpa -n $NAMESPACE"
    echo ""
    echo "12. Cleanup:"
    echo "    ./scripts/cleanup.sh"
    echo ""
}

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help              Show this help message"
    echo "  -n, --namespace NAME    Set namespace (default: $NAMESPACE)"
    echo "  -s, --skip-tests        Skip endpoint tests"
    echo "  -q, --quiet             Quiet mode (less output)"
    echo "  --storage-only          Deploy only storage resources"
    echo "  --serving-only          Deploy only model serving"
    echo "  --training-only         Deploy only training jobs"
    echo "  --no-gpu                Skip GPU-related resources"
    echo ""
}

# Parse command line arguments
SKIP_TESTS=false
QUIET=false
STORAGE_ONLY=false
SERVING_ONLY=false
TRAINING_ONLY=false
NO_GPU=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            usage
            exit 0
            ;;
        -n|--namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        -s|--skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        -q|--quiet)
            QUIET=true
            shift
            ;;
        --storage-only)
            STORAGE_ONLY=true
            shift
            ;;
        --serving-only)
            SERVING_ONLY=true
            shift
            ;;
        --training-only)
            TRAINING_ONLY=true
            shift
            ;;
        --no-gpu)
            NO_GPU=true
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Main execution
main() {
    print_header "ML Workloads Deployment"

    # Pre-flight checks
    print_info "Running pre-flight checks..."
    check_kubectl
    check_cluster
    check_gpu_support

    print_info "Target namespace: $NAMESPACE"
    echo ""

    # Deploy namespace and resource quotas
    deploy_manifests "01-namespace.yaml" "Deploying Namespace and Resource Quotas"

    # Deploy storage
    if [ "$SERVING_ONLY" = false ] && [ "$TRAINING_ONLY" = false ]; then
        deploy_manifests "02-model-storage.yaml" "Deploying Model Storage"
        wait_for_storage_init
    fi

    # Deploy model serving
    if [ "$STORAGE_ONLY" = false ] && [ "$TRAINING_ONLY" = false ]; then
        deploy_manifests "03-model-serving.yaml" "Deploying Model Serving"
        wait_for_deployments
    fi

    # Deploy training jobs
    if [ "$STORAGE_ONLY" = false ] && [ "$SERVING_ONLY" = false ]; then
        deploy_manifests "04-training-jobs.yaml" "Deploying Training Jobs"
    fi

    # Deploy A/B testing and canary
    if [ "$STORAGE_ONLY" = false ] && [ "$TRAINING_ONLY" = false ]; then
        deploy_manifests "05-ab-testing-canary.yaml" "Deploying A/B Testing and Canary"
    fi

    # Show summary
    if [ "$QUIET" = false ]; then
        show_summary
    fi

    # Run tests
    if [ "$SKIP_TESTS" = false ] && [ "$QUIET" = false ]; then
        test_endpoints
    fi

    # Show next steps
    if [ "$QUIET" = false ]; then
        show_next_steps
    fi

    print_success "Deployment completed successfully!"
}

# Run main function
main "$@"
