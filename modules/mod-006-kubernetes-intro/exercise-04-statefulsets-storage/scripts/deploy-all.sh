#!/bin/bash

################################################################################
# Deploy StatefulSets and Storage Examples
#
# This script deploys all examples for Exercise 04
################################################################################

set -e

# Color definitions
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFESTS_DIR="$SCRIPT_DIR/../manifests"
EXAMPLES_DIR="$SCRIPT_DIR/../examples"

print_header() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# Check prerequisites
check_prerequisites() {
    print_header "Checking Prerequisites"

    if ! command -v kubectl &> /dev/null; then
        print_error "kubectl is not installed"
        exit 1
    fi
    print_success "kubectl is installed"

    if ! kubectl cluster-info &> /dev/null; then
        print_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    print_success "Connected to Kubernetes cluster"
}

# Deploy manifests
deploy_manifests() {
    print_header "Deploying Manifests"

    local files=(
        "01-namespace.yaml"
        "02-storageclass.yaml"
        "03-postgresql-statefulset.yaml"
        "04-volume-types.yaml"
        "05-redis-statefulset.yaml"
    )

    for file in "${files[@]}"; do
        if [ -f "$MANIFESTS_DIR/$file" ]; then
            print_info "Applying $file..."
            if kubectl apply -f "$MANIFESTS_DIR/$file"; then
                print_success "$file applied"
            else
                print_error "Failed to apply $file"
                return 1
            fi
        else
            print_warning "$file not found, skipping"
        fi
    done
}

# Deploy examples
deploy_examples() {
    print_header "Deploying Examples"

    if [ -f "$EXAMPLES_DIR/statefulset-vs-deployment.yaml" ]; then
        print_info "Applying StatefulSet vs Deployment comparison..."
        if kubectl apply -f "$EXAMPLES_DIR/statefulset-vs-deployment.yaml"; then
            print_success "Examples deployed"
        else
            print_error "Failed to deploy examples"
            return 1
        fi
    fi
}

# Wait for resources
wait_for_resources() {
    print_header "Waiting for Resources"

    print_info "Waiting for PostgreSQL StatefulSet..."
    kubectl rollout status statefulset/postgres -n statefulset-demo --timeout=180s || true

    print_info "Waiting for Redis StatefulSet..."
    kubectl rollout status statefulset/redis -n statefulset-demo --timeout=180s || true

    print_info "Checking pod status..."
    kubectl get pods -n statefulset-demo
}

# Show deployment status
show_status() {
    print_header "Deployment Status"

    echo ""
    print_info "Namespaces:"
    kubectl get namespace statefulset-demo

    echo ""
    print_info "StatefulSets:"
    kubectl get statefulsets -n statefulset-demo

    echo ""
    print_info "Pods:"
    kubectl get pods -n statefulset-demo -o wide

    echo ""
    print_info "Services:"
    kubectl get svc -n statefulset-demo

    echo ""
    print_info "PersistentVolumeClaims:"
    kubectl get pvc -n statefulset-demo

    echo ""
    print_info "StorageClasses:"
    kubectl get storageclass
}

# Show next steps
show_next_steps() {
    print_header "Next Steps"

    cat << EOF

${GREEN}Successfully deployed all StatefulSet examples!${NC}

You can now explore the examples using these commands:

${BLUE}1. Check PostgreSQL StatefulSet:${NC}
   kubectl get pods -n statefulset-demo -l app=postgresql
   kubectl logs postgres-0 -n statefulset-demo

${BLUE}2. Connect to PostgreSQL:${NC}
   kubectl exec -it postgres-0 -n statefulset-demo -- psql -U myuser -d mydatabase

${BLUE}3. Check Redis StatefulSet:${NC}
   kubectl get pods -n statefulset-demo -l app=redis

${BLUE}4. Connect to Redis:${NC}
   kubectl exec -it redis-0 -n statefulset-demo -- redis-cli

${BLUE}5. Test StatefulSet vs Deployment:${NC}
   kubectl get pods -n statefulset-demo -l type=statefulset-example
   kubectl get pods -n statefulset-demo -l type=deployment-example

${BLUE}6. Check persistent storage:${NC}
   kubectl get pvc -n statefulset-demo

${BLUE}7. Scale StatefulSets:${NC}
   kubectl scale statefulset redis --replicas=5 -n statefulset-demo
   kubectl get pods -n statefulset-demo -l app=redis -w

${BLUE}8. Test data persistence:${NC}
   $SCRIPT_DIR/test-persistence.sh

${BLUE}9. Clean up:${NC}
   $SCRIPT_DIR/cleanup.sh

For more details, see README.md

EOF
}

# Main execution
main() {
    check_prerequisites
    deploy_manifests
    deploy_examples
    wait_for_resources
    show_status
    show_next_steps
}

main "$@"
