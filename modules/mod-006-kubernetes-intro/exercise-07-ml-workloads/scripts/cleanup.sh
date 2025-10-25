#!/bin/bash

# cleanup.sh
# Cleanup script for ML Workloads exercise

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
NAMESPACE="ml-workloads"
FORCE=false
DELETE_NAMESPACE=false
DELETE_STORAGE=false

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

# Function to confirm action
confirm() {
    if [ "$FORCE" = true ]; then
        return 0
    fi

    local message=$1
    read -p "$message (y/N): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        return 0
    else
        return 1
    fi
}

# Function to delete resources
delete_resources() {
    local resource_type=$1
    local count=$(kubectl get "$resource_type" -n "$NAMESPACE" --no-headers 2>/dev/null | wc -l)

    if [ "$count" -eq 0 ]; then
        print_info "No $resource_type found"
        return 0
    fi

    print_info "Found $count $resource_type"

    if kubectl delete "$resource_type" --all -n "$NAMESPACE" --wait=true --timeout=60s 2>/dev/null; then
        print_success "Deleted $count $resource_type"
    else
        print_warning "Some $resource_type may not have been deleted"
    fi
}

# Function to list resources
list_resources() {
    print_header "Resources in namespace: $NAMESPACE"

    echo "Deployments:"
    kubectl get deployments -n "$NAMESPACE" 2>/dev/null || echo "  (none)"
    echo ""

    echo "Jobs:"
    kubectl get jobs -n "$NAMESPACE" 2>/dev/null || echo "  (none)"
    echo ""

    echo "CronJobs:"
    kubectl get cronjobs -n "$NAMESPACE" 2>/dev/null || echo "  (none)"
    echo ""

    echo "Services:"
    kubectl get services -n "$NAMESPACE" 2>/dev/null || echo "  (none)"
    echo ""

    echo "Ingress:"
    kubectl get ingress -n "$NAMESPACE" 2>/dev/null || echo "  (none)"
    echo ""

    echo "PersistentVolumeClaims:"
    kubectl get pvc -n "$NAMESPACE" 2>/dev/null || echo "  (none)"
    echo ""

    echo "Pods:"
    kubectl get pods -n "$NAMESPACE" 2>/dev/null || echo "  (none)"
    echo ""
}

# Function to delete namespace
delete_namespace_func() {
    print_header "Deleting Namespace"

    print_warning "This will delete the entire namespace and all resources in it!"

    if confirm "Are you sure you want to delete namespace '$NAMESPACE'?"; then
        print_info "Deleting namespace '$NAMESPACE'..."

        if kubectl delete namespace "$NAMESPACE" --wait=true --timeout=180s; then
            print_success "Namespace deleted successfully"
        else
            print_error "Failed to delete namespace"
            exit 1
        fi
    else
        print_info "Namespace deletion cancelled"
    fi
}

# Function to delete all resources without deleting namespace
delete_all_resources() {
    print_header "Deleting All Resources"

    # Delete in order
    delete_resources "ingress"
    delete_resources "horizontalpodautoscalers"
    delete_resources "poddisruptionbudgets"
    delete_resources "cronjobs"
    delete_resources "jobs"
    delete_resources "deployments"
    delete_resources "services"

    if [ "$DELETE_STORAGE" = true ]; then
        print_warning "Deleting storage (PVCs)..."
        delete_resources "persistentvolumeclaims"
    else
        print_info "Keeping PVCs (use --delete-storage to remove)"
    fi

    delete_resources "pods"
    delete_resources "configmaps"
    delete_resources "secrets"

    print_success "Resources deleted (namespace preserved)"
}

# Function to delete only training jobs
delete_training_jobs() {
    print_header "Deleting Training Jobs"

    print_info "Deleting CronJobs..."
    delete_resources "cronjobs"

    print_info "Deleting Jobs..."
    kubectl delete jobs -n "$NAMESPACE" -l job-type=training --wait=true 2>/dev/null || true

    print_success "Training jobs deleted"
}

# Function to delete only serving deployments
delete_serving() {
    print_header "Deleting Model Serving"

    print_info "Deleting serving deployments..."
    kubectl delete deployments -n "$NAMESPACE" -l component=model-server --wait=true 2>/dev/null || true

    print_info "Deleting services..."
    delete_resources "services"

    print_info "Deleting ingress..."
    delete_resources "ingress"

    print_info "Deleting HPAs..."
    delete_resources "horizontalpodautoscalers"

    print_success "Model serving deleted"
}

# Function to show usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help              Show this help message"
    echo "  -n, --namespace NAME    Set namespace (default: $NAMESPACE)"
    echo "  -f, --force             Skip confirmation prompts"
    echo "  -d, --delete-namespace  Delete the entire namespace"
    echo "  --delete-storage        Also delete PVCs (model storage)"
    echo "  --training-only         Delete only training jobs"
    echo "  --serving-only          Delete only model serving"
    echo ""
}

# Function to show menu
show_menu() {
    print_header "Cleanup Menu"

    echo "What would you like to clean up?"
    echo ""
    echo "1) List all resources (no deletion)"
    echo "2) Delete training jobs only"
    echo "3) Delete model serving only"
    echo "4) Delete all resources (keep PVCs and namespace)"
    echo "5) Delete all resources including storage"
    echo "6) Delete entire namespace"
    echo "7) Exit"
    echo ""
    read -p "Enter your choice (1-7): " choice

    case $choice in
        1)
            list_resources
            ;;
        2)
            if confirm "Delete all training jobs?"; then
                delete_training_jobs
            fi
            ;;
        3)
            if confirm "Delete all model serving resources?"; then
                delete_serving
            fi
            ;;
        4)
            if confirm "Delete all resources (keep PVCs)?"; then
                DELETE_STORAGE=false
                delete_all_resources
            fi
            ;;
        5)
            print_warning "This will delete all model data!"
            if confirm "Delete all resources including storage?"; then
                DELETE_STORAGE=true
                delete_all_resources
            fi
            ;;
        6)
            delete_namespace_func
            return
            ;;
        7)
            print_info "Exiting without changes"
            exit 0
            ;;
        *)
            print_error "Invalid choice"
            ;;
    esac

    echo ""
    read -p "Press Enter to continue..."
    show_menu
}

# Parse command line arguments
TRAINING_ONLY=false
SERVING_ONLY=false

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
        -f|--force)
            FORCE=true
            shift
            ;;
        -d|--delete-namespace)
            DELETE_NAMESPACE=true
            shift
            ;;
        --delete-storage)
            DELETE_STORAGE=true
            shift
            ;;
        --training-only)
            TRAINING_ONLY=true
            shift
            ;;
        --serving-only)
            SERVING_ONLY=true
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
    print_header "ML Workloads Cleanup"

    # Check if namespace exists
    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        print_error "Namespace '$NAMESPACE' not found"
        print_info "Nothing to clean up"
        exit 0
    fi

    # Execute based on flags
    if [ "$DELETE_NAMESPACE" = true ]; then
        delete_namespace_func
    elif [ "$TRAINING_ONLY" = true ]; then
        if [ "$FORCE" = false ]; then
            if ! confirm "Delete training jobs in namespace '$NAMESPACE'?"; then
                print_info "Cleanup cancelled"
                exit 0
            fi
        fi
        delete_training_jobs
    elif [ "$SERVING_ONLY" = true ]; then
        if [ "$FORCE" = false ]; then
            if ! confirm "Delete model serving in namespace '$NAMESPACE'?"; then
                print_info "Cleanup cancelled"
                exit 0
            fi
        fi
        delete_serving
    elif [ "$FORCE" = true ]; then
        delete_all_resources
    else
        list_resources
        show_menu
    fi

    print_success "Cleanup complete!"
}

# Run main function
main "$@"
