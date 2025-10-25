#!/bin/bash

# deploy-all.sh
# Automated deployment script for ConfigMaps and Secrets exercise
# Deploys all manifests in the correct order

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MANIFESTS_DIR="$PROJECT_ROOT/manifests"
EXAMPLES_DIR="$PROJECT_ROOT/examples"

# Configuration
NAMESPACE="config-demo"

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
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
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
        print_info "Please check your kubeconfig and cluster connection"
        exit 1
    fi
    print_success "Connected to Kubernetes cluster"
}

# Function to wait for resource to be ready
wait_for_pods() {
    local namespace=$1
    local label=$2
    local timeout=${3:-60}

    print_info "Waiting for pods with label '$label' to be ready (timeout: ${timeout}s)..."

    if kubectl wait --for=condition=Ready pod \
        -l "$label" \
        -n "$namespace" \
        --timeout="${timeout}s" 2>/dev/null; then
        print_success "Pods are ready"
        return 0
    else
        print_warning "Some pods may not be ready yet"
        return 1
    fi
}

# Function to deploy manifests
deploy_manifests() {
    local dir=$1
    local description=$2

    print_header "$description"

    if [ ! -d "$dir" ]; then
        print_error "Directory not found: $dir"
        return 1
    fi

    # Find all YAML files and sort them
    for file in $(find "$dir" -name "*.yaml" -o -name "*.yml" | sort); do
        filename=$(basename "$file")
        print_info "Deploying $filename..."

        if kubectl apply -f "$file"; then
            print_success "Deployed $filename"
        else
            print_error "Failed to deploy $filename"
            return 1
        fi
    done
}

# Function to display resource summary
show_summary() {
    print_header "Deployment Summary"

    print_info "Namespace: $NAMESPACE"
    kubectl get namespace "$NAMESPACE" -o wide 2>/dev/null || true
    echo ""

    print_info "ConfigMaps:"
    kubectl get configmaps -n "$NAMESPACE" --show-labels 2>/dev/null || print_warning "No ConfigMaps found"
    echo ""

    print_info "Secrets:"
    kubectl get secrets -n "$NAMESPACE" --show-labels 2>/dev/null || print_warning "No Secrets found"
    echo ""

    print_info "Pods:"
    kubectl get pods -n "$NAMESPACE" -o wide 2>/dev/null || print_warning "No Pods found"
    echo ""

    print_info "Deployments:"
    kubectl get deployments -n "$NAMESPACE" -o wide 2>/dev/null || print_warning "No Deployments found"
    echo ""
}

# Function to test basic functionality
test_deployment() {
    print_header "Testing Deployment"

    # Test ConfigMap access
    print_info "Testing ConfigMap access..."
    if kubectl get configmap app-config-literals -n "$NAMESPACE" &> /dev/null; then
        print_success "ConfigMaps are accessible"
        echo "Sample ConfigMap data:"
        kubectl get configmap app-config-literals -n "$NAMESPACE" -o jsonpath='{.data.APP_NAME}' 2>/dev/null
        echo ""
    else
        print_warning "ConfigMaps not found or not accessible"
    fi

    # Test Secret access
    print_info "Testing Secret access..."
    if kubectl get secret app-secrets -n "$NAMESPACE" &> /dev/null; then
        print_success "Secrets are accessible"
        echo "Secret keys:"
        kubectl get secret app-secrets -n "$NAMESPACE" -o jsonpath='{.data}' 2>/dev/null | grep -o '"[^"]*":' | tr -d '":' | head -3
        echo ""
    else
        print_warning "Secrets not found or not accessible"
    fi

    # Check pod logs for one example
    print_info "Checking sample pod logs..."
    if kubectl get pod pod-configmap-env -n "$NAMESPACE" &> /dev/null; then
        echo "Sample output from pod-configmap-env:"
        kubectl logs pod-configmap-env -n "$NAMESPACE" --tail=10 2>/dev/null || print_warning "Pod not ready yet"
    fi
    echo ""
}

# Function to show next steps
show_next_steps() {
    print_header "Next Steps"

    echo "Deployment complete! Here are some things you can try:"
    echo ""
    echo "1. View ConfigMaps:"
    echo "   kubectl get configmaps -n $NAMESPACE"
    echo "   kubectl describe configmap app-config-literals -n $NAMESPACE"
    echo ""
    echo "2. View Secrets:"
    echo "   kubectl get secrets -n $NAMESPACE"
    echo "   kubectl describe secret app-secrets -n $NAMESPACE"
    echo ""
    echo "3. View pod logs:"
    echo "   kubectl logs -n $NAMESPACE pod-configmap-env"
    echo "   kubectl logs -n $NAMESPACE pod-secret-env"
    echo ""
    echo "4. Execute commands in pods:"
    echo "   kubectl exec -it -n $NAMESPACE pod-configmap-volume -- /bin/sh"
    echo "   kubectl exec -n $NAMESPACE pod-configmap-volume -- cat /etc/config/nginx.conf"
    echo ""
    echo "5. Test configuration changes:"
    echo "   kubectl edit configmap app-config-literals -n $NAMESPACE"
    echo "   kubectl logs -n $NAMESPACE pod-configmap-hot-reload -f"
    echo ""
    echo "6. Run test script:"
    echo "   ./scripts/test-configs.sh"
    echo ""
    echo "7. Clean up:"
    echo "   ./scripts/cleanup.sh"
    echo ""
}

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help              Show this help message"
    echo "  -n, --namespace NAME    Set namespace (default: $NAMESPACE)"
    echo "  -m, --manifests-only    Deploy only manifests (skip examples)"
    echo "  -e, --examples-only     Deploy only examples (skip manifests)"
    echo "  -s, --skip-tests        Skip deployment tests"
    echo "  -q, --quiet             Quiet mode (less output)"
    echo ""
    echo "Examples:"
    echo "  $0                      # Deploy everything"
    echo "  $0 -m                   # Deploy only manifests"
    echo "  $0 -e                   # Deploy only examples"
    echo "  $0 -n my-namespace      # Use custom namespace"
    echo ""
}

# Parse command line arguments
MANIFESTS_ONLY=false
EXAMPLES_ONLY=false
SKIP_TESTS=false
QUIET=false

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
        -m|--manifests-only)
            MANIFESTS_ONLY=true
            shift
            ;;
        -e|--examples-only)
            EXAMPLES_ONLY=true
            shift
            ;;
        -s|--skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        -q|--quiet)
            QUIET=true
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
    print_header "ConfigMaps and Secrets Deployment"

    # Pre-flight checks
    print_info "Running pre-flight checks..."
    check_kubectl
    check_cluster

    print_info "Target namespace: $NAMESPACE"
    echo ""

    # Deploy manifests
    if [ "$EXAMPLES_ONLY" = false ]; then
        deploy_manifests "$MANIFESTS_DIR" "Deploying Manifests"

        # Wait for some pods to be ready
        if kubectl get pods -n "$NAMESPACE" -l example=configmap-env &> /dev/null; then
            wait_for_pods "$NAMESPACE" "example=configmap-env" 30 || true
        fi
    fi

    # Deploy examples
    if [ "$MANIFESTS_ONLY" = false ]; then
        deploy_manifests "$EXAMPLES_DIR" "Deploying Examples"

        # Wait for some deployments to be ready
        if kubectl get deployment -n "$NAMESPACE" &> /dev/null; then
            print_info "Waiting for deployments to be ready..."
            kubectl wait --for=condition=available deployment \
                --all -n "$NAMESPACE" --timeout=60s 2>/dev/null || print_warning "Some deployments may not be ready"
        fi
    fi

    # Show summary
    if [ "$QUIET" = false ]; then
        show_summary
    fi

    # Run tests
    if [ "$SKIP_TESTS" = false ] && [ "$QUIET" = false ]; then
        test_deployment
    fi

    # Show next steps
    if [ "$QUIET" = false ]; then
        show_next_steps
    fi

    print_success "Deployment completed successfully!"
}

# Run main function
main "$@"
