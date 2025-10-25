#!/bin/bash

# deploy-all.sh
# Automated deployment script for Ingress and Load Balancing exercise

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
NAMESPACE="ingress-demo"
INGRESS_CLASS="nginx"

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
        print_info "Please check your kubeconfig and cluster connection"
        exit 1
    fi
    print_success "Connected to Kubernetes cluster"
}

# Function to check if Ingress controller is installed
check_ingress_controller() {
    print_info "Checking for Ingress controller..."

    if kubectl get ingressclass "$INGRESS_CLASS" &> /dev/null; then
        print_success "Ingress controller '$INGRESS_CLASS' is available"
        return 0
    fi

    print_warning "Ingress controller '$INGRESS_CLASS' not found"
    print_info "You need an Ingress controller to use Ingress resources"
    echo ""
    echo "To install NGINX Ingress Controller:"
    echo "  kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.9.4/deploy/static/provider/cloud/deploy.yaml"
    echo ""
    echo "For other providers (minikube, kind, etc.), see:"
    echo "  https://kubernetes.github.io/ingress-nginx/deploy/"
    echo ""
    read -p "Do you want to continue anyway? (y/N): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
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

# Function to wait for deployments
wait_for_deployments() {
    print_info "Waiting for deployments to be ready..."

    if kubectl wait --for=condition=available deployment \
        --all -n "$NAMESPACE" --timeout=120s 2>/dev/null; then
        print_success "All deployments are ready"
    else
        print_warning "Some deployments may not be ready yet"
        print_info "Check status with: kubectl get deployments -n $NAMESPACE"
    fi
}

# Function to get ingress controller IP/hostname
get_ingress_address() {
    print_info "Getting Ingress controller address..."

    # Try to get LoadBalancer IP
    local lb_ip=$(kubectl get svc -n ingress-nginx ingress-nginx-controller \
        -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null)

    if [ -n "$lb_ip" ]; then
        echo "$lb_ip"
        return
    fi

    # Try to get LoadBalancer hostname (AWS ELB)
    local lb_host=$(kubectl get svc -n ingress-nginx ingress-nginx-controller \
        -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null)

    if [ -n "$lb_host" ]; then
        echo "$lb_host"
        return
    fi

    # Fallback to NodePort
    local node_ip=$(kubectl get nodes -o jsonpath='{.items[0].status.addresses[?(@.type=="ExternalIP")].address}' 2>/dev/null)
    if [ -z "$node_ip" ]; then
        node_ip=$(kubectl get nodes -o jsonpath='{.items[0].status.addresses[?(@.type=="InternalIP")].address}' 2>/dev/null)
    fi

    if [ -n "$node_ip" ]; then
        echo "$node_ip"
        return
    fi

    # No address found
    echo "localhost"
}

# Function to show deployment summary
show_summary() {
    print_header "Deployment Summary"

    print_info "Namespace: $NAMESPACE"
    kubectl get namespace "$NAMESPACE" -o wide 2>/dev/null || true
    echo ""

    print_info "Deployments:"
    kubectl get deployments -n "$NAMESPACE" 2>/dev/null || print_warning "No Deployments found"
    echo ""

    print_info "Services:"
    kubectl get services -n "$NAMESPACE" 2>/dev/null || print_warning "No Services found"
    echo ""

    print_info "Ingress Resources:"
    kubectl get ingress -n "$NAMESPACE" 2>/dev/null || print_warning "No Ingress resources found"
    echo ""

    print_info "Network Policies:"
    kubectl get networkpolicies -n "$NAMESPACE" 2>/dev/null || print_warning "No Network Policies found"
    echo ""
}

# Function to test basic connectivity
test_deployment() {
    print_header "Testing Deployment"

    # Get a backend pod
    local pod=$(kubectl get pods -n "$NAMESPACE" -l app=backend,version=v1 \
        -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)

    if [ -n "$pod" ]; then
        print_info "Testing backend pod: $pod"
        if kubectl exec -n "$NAMESPACE" "$pod" -- wget -O- http://localhost:8080 &> /dev/null; then
            print_success "Backend pod is responding"
        else
            print_warning "Backend pod is not responding"
        fi
    fi

    # Test service
    print_info "Testing service connectivity..."
    if kubectl run test-client --rm -it --restart=Never --image=busybox:1.36 -n "$NAMESPACE" \
        -- wget -O- http://backend-v1 --timeout=5 &> /dev/null; then
        print_success "Service connectivity works"
    else
        print_warning "Service connectivity test inconclusive"
    fi
}

# Function to show next steps
show_next_steps() {
    print_header "Next Steps"

    local ingress_ip=$(get_ingress_address)

    echo "Deployment complete! Here's how to use the Ingress resources:"
    echo ""
    echo "1. Get Ingress controller address:"
    echo "   Ingress IP: $ingress_ip"
    echo ""
    echo "2. Add entries to /etc/hosts (for local testing):"
    echo "   echo \"$ingress_ip api.example.com\" | sudo tee -a /etc/hosts"
    echo "   echo \"$ingress_ip web.example.com\" | sudo tee -a /etc/hosts"
    echo "   echo \"$ingress_ip admin.example.com\" | sudo tee -a /etc/hosts"
    echo ""
    echo "3. Test Ingress endpoints:"
    echo "   curl http://$ingress_ip/"
    echo "   curl http://api.example.com/"
    echo "   curl http://web.example.com/"
    echo ""
    echo "4. View Ingress resources:"
    echo "   kubectl get ingress -n $NAMESPACE"
    echo "   kubectl describe ingress basic-ingress -n $NAMESPACE"
    echo ""
    echo "5. Test canary deployment:"
    echo "   for i in {1..10}; do curl http://canary.example.com/; echo; done"
    echo ""
    echo "6. Test path-based routing:"
    echo "   curl http://$ingress_ip/api"
    echo "   curl http://$ingress_ip/web"
    echo "   curl http://$ingress_ip/admin"
    echo ""
    echo "7. View Network Policies:"
    echo "   kubectl get networkpolicies -n $NAMESPACE"
    echo ""
    echo "8. Run comprehensive tests:"
    echo "   ./scripts/test-ingress.sh"
    echo ""
    echo "9. Clean up:"
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
    echo "  -s, --skip-tests        Skip deployment tests"
    echo "  -q, --quiet             Quiet mode (less output)"
    echo "  --apps-only             Deploy only applications (skip Ingress)"
    echo "  --ingress-only          Deploy only Ingress resources (skip apps)"
    echo "  --no-network-policies   Skip Network Policy deployment"
    echo ""
}

# Parse command line arguments
SKIP_TESTS=false
QUIET=false
APPS_ONLY=false
INGRESS_ONLY=false
NO_NETWORK_POLICIES=false

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
        --apps-only)
            APPS_ONLY=true
            shift
            ;;
        --ingress-only)
            INGRESS_ONLY=true
            shift
            ;;
        --no-network-policies)
            NO_NETWORK_POLICIES=true
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
    print_header "Ingress and Load Balancing Deployment"

    # Pre-flight checks
    print_info "Running pre-flight checks..."
    check_kubectl
    check_cluster
    check_ingress_controller

    print_info "Target namespace: $NAMESPACE"
    echo ""

    # Deploy namespace
    deploy_manifests "01-namespace.yaml" "Deploying Namespace"

    # Deploy applications
    if [ "$INGRESS_ONLY" = false ]; then
        deploy_manifests "02-backend-apps.yaml" "Deploying Backend Applications"
        deploy_manifests "03-services.yaml" "Deploying Services"

        # Wait for deployments
        wait_for_deployments
    fi

    # Deploy Ingress resources
    if [ "$APPS_ONLY" = false ]; then
        deploy_manifests "04-ingress-basic.yaml" "Deploying Basic Ingress Resources"
        deploy_manifests "05-ingress-tls.yaml" "Deploying TLS Ingress Resources"
        deploy_manifests "06-ingress-advanced.yaml" "Deploying Advanced Ingress Resources"
    fi

    # Deploy Network Policies
    if [ "$NO_NETWORK_POLICIES" = false ] && [ "$APPS_ONLY" = false ]; then
        deploy_manifests "07-network-policies.yaml" "Deploying Network Policies"
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
