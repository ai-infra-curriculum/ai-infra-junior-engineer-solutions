#!/bin/bash

# cleanup.sh
# Cleanup script for Ingress and Load Balancing exercise

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
NAMESPACE="ingress-demo"
FORCE=false
DELETE_NAMESPACE=false

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

    echo "Services:"
    kubectl get services -n "$NAMESPACE" 2>/dev/null || echo "  (none)"
    echo ""

    echo "Ingress:"
    kubectl get ingress -n "$NAMESPACE" 2>/dev/null || echo "  (none)"
    echo ""

    echo "Network Policies:"
    kubectl get networkpolicies -n "$NAMESPACE" 2>/dev/null || echo "  (none)"
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

        if kubectl delete namespace "$NAMESPACE" --wait=true --timeout=120s; then
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

    delete_resources "ingress"
    delete_resources "networkpolicies"
    delete_resources "deployments"
    delete_resources "services"
    delete_resources "pods"
    delete_resources "secrets"

    print_success "All resources deleted"
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
    echo ""
}

# Function to show menu
show_menu() {
    print_header "Cleanup Menu"

    echo "What would you like to clean up?"
    echo ""
    echo "1) List all resources (no deletion)"
    echo "2) Delete Ingress resources only"
    echo "3) Delete Network Policies only"
    echo "4) Delete all resources (keep namespace)"
    echo "5) Delete entire namespace"
    echo "6) Exit"
    echo ""
    read -p "Enter your choice (1-6): " choice

    case $choice in
        1)
            list_resources
            ;;
        2)
            if confirm "Delete all Ingress resources?"; then
                delete_resources "ingress"
            fi
            ;;
        3)
            if confirm "Delete all Network Policies?"; then
                delete_resources "networkpolicies"
            fi
            ;;
        4)
            if confirm "Delete all resources in namespace '$NAMESPACE'?"; then
                delete_all_resources
            fi
            ;;
        5)
            delete_namespace_func
            return
            ;;
        6)
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
        *)
            print_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Main execution
main() {
    print_header "Ingress and Load Balancing Cleanup"

    # Check if namespace exists
    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        print_error "Namespace '$NAMESPACE' not found"
        print_info "Nothing to clean up"
        exit 0
    fi

    # Execute based on flags
    if [ "$DELETE_NAMESPACE" = true ]; then
        delete_namespace_func
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
