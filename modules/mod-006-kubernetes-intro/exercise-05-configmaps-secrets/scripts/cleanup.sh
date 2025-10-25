#!/bin/bash

# cleanup.sh
# Cleanup script for ConfigMaps and Secrets exercise
# Removes all resources created during the exercise

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
NAMESPACE="config-demo"
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

# Function to show usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help              Show this help message"
    echo "  -n, --namespace NAME    Set namespace (default: $NAMESPACE)"
    echo "  -f, --force             Skip confirmation prompts"
    echo "  -d, --delete-namespace  Delete the entire namespace"
    echo "  -r, --resources-only    Delete only specific resources, keep ConfigMaps/Secrets"
    echo ""
    echo "Examples:"
    echo "  $0                      # Interactive cleanup"
    echo "  $0 -f                   # Force cleanup without confirmation"
    echo "  $0 -d                   # Delete entire namespace"
    echo "  $0 -r                   # Delete only pods and deployments"
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

# Function to delete resources by type
delete_resources() {
    local resource_type=$1
    local label=${2:-""}

    print_info "Deleting $resource_type..."

    local selector=""
    if [ -n "$label" ]; then
        selector="-l $label"
    fi

    local count=$(kubectl get "$resource_type" -n "$NAMESPACE" $selector --no-headers 2>/dev/null | wc -l)

    if [ "$count" -eq 0 ]; then
        print_info "No $resource_type found"
        return 0
    fi

    print_info "Found $count $resource_type"

    if kubectl delete "$resource_type" --all -n "$NAMESPACE" $selector --wait=true --timeout=60s 2>/dev/null; then
        print_success "Deleted $count $resource_type"
    else
        print_warning "Some $resource_type may not have been deleted"
    fi
}

# Function to list resources before deletion
list_resources() {
    print_header "Resources in namespace: $NAMESPACE"

    echo "Deployments:"
    kubectl get deployments -n "$NAMESPACE" 2>/dev/null || echo "  (none)"
    echo ""

    echo "Pods:"
    kubectl get pods -n "$NAMESPACE" 2>/dev/null || echo "  (none)"
    echo ""

    echo "Services:"
    kubectl get services -n "$NAMESPACE" 2>/dev/null || echo "  (none)"
    echo ""

    echo "ConfigMaps:"
    kubectl get configmaps -n "$NAMESPACE" 2>/dev/null || echo "  (none)"
    echo ""

    echo "Secrets:"
    kubectl get secrets -n "$NAMESPACE" 2>/dev/null || echo "  (none)"
    echo ""
}

# Function to delete specific resources (keeping ConfigMaps and Secrets)
delete_resources_only() {
    print_header "Deleting Workload Resources"

    delete_resources "deployments"
    delete_resources "pods"
    delete_resources "services"

    print_success "Workload resources deleted. ConfigMaps and Secrets preserved."
}

# Function to delete all resources
delete_all_resources() {
    print_header "Deleting All Resources"

    # Delete in order to minimize orphaned resources
    delete_resources "deployments"
    delete_resources "pods"
    delete_resources "services"
    delete_resources "configmaps"
    delete_resources "secrets"

    print_success "All resources deleted"
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
            print_info "You may need to check for stuck resources:"
            print_info "  kubectl get namespace $NAMESPACE -o json"
            exit 1
        fi
    else
        print_info "Namespace deletion cancelled"
    fi
}

# Function to handle stuck resources
force_cleanup_stuck_resources() {
    print_header "Force Cleanup Stuck Resources"

    print_info "Checking for resources with finalizers..."

    # Check pods with finalizers
    local stuck_pods=$(kubectl get pods -n "$NAMESPACE" -o json 2>/dev/null | \
        jq -r '.items[] | select(.metadata.finalizers != null) | .metadata.name' 2>/dev/null)

    if [ -n "$stuck_pods" ]; then
        print_warning "Found pods with finalizers:"
        echo "$stuck_pods"

        if confirm "Remove finalizers from stuck pods?"; then
            echo "$stuck_pods" | while read -r pod; do
                print_info "Removing finalizers from pod: $pod"
                kubectl patch pod "$pod" -n "$NAMESPACE" \
                    -p '{"metadata":{"finalizers":[]}}' --type=merge 2>/dev/null || true
            done
        fi
    fi

    # Check namespace finalizers
    local ns_finalizers=$(kubectl get namespace "$NAMESPACE" -o json 2>/dev/null | \
        jq -r '.spec.finalizers[]' 2>/dev/null)

    if [ -n "$ns_finalizers" ]; then
        print_warning "Namespace has finalizers:"
        echo "$ns_finalizers"

        if confirm "Remove finalizers from namespace?"; then
            print_info "Removing finalizers from namespace"
            kubectl patch namespace "$NAMESPACE" \
                -p '{"spec":{"finalizers":[]}}' --type=merge 2>/dev/null || true
        fi
    fi
}

# Function to export resources before deletion (backup)
backup_resources() {
    print_header "Backing Up Resources"

    local backup_dir="./backup-$(date +%Y%m%d-%H%M%S)"
    mkdir -p "$backup_dir"

    print_info "Backing up resources to: $backup_dir"

    # Backup ConfigMaps
    print_info "Backing up ConfigMaps..."
    kubectl get configmaps -n "$NAMESPACE" -o yaml > "$backup_dir/configmaps.yaml" 2>/dev/null || true

    # Backup Secrets
    print_info "Backing up Secrets..."
    kubectl get secrets -n "$NAMESPACE" -o yaml > "$backup_dir/secrets.yaml" 2>/dev/null || true

    # Backup Deployments
    print_info "Backing up Deployments..."
    kubectl get deployments -n "$NAMESPACE" -o yaml > "$backup_dir/deployments.yaml" 2>/dev/null || true

    # Backup Pods (for reference)
    print_info "Backing up Pod specs..."
    kubectl get pods -n "$NAMESPACE" -o yaml > "$backup_dir/pods.yaml" 2>/dev/null || true

    print_success "Backup complete: $backup_dir"
    echo "To restore, run: kubectl apply -f $backup_dir/"
}

# Function to show cleanup statistics
show_cleanup_stats() {
    print_header "Cleanup Statistics"

    local before_pods=${1:-0}
    local before_configs=${2:-0}
    local before_secrets=${3:-0}
    local before_deploys=${4:-0}

    local after_pods=$(kubectl get pods -n "$NAMESPACE" --no-headers 2>/dev/null | wc -l)
    local after_configs=$(kubectl get configmaps -n "$NAMESPACE" --no-headers 2>/dev/null | wc -l)
    local after_secrets=$(kubectl get secrets -n "$NAMESPACE" --no-headers 2>/dev/null | wc -l)
    local after_deploys=$(kubectl get deployments -n "$NAMESPACE" --no-headers 2>/dev/null | wc -l)

    echo "Resource counts before -> after cleanup:"
    echo "  Pods:        $before_pods -> $after_pods"
    echo "  ConfigMaps:  $before_configs -> $after_configs"
    echo "  Secrets:     $before_secrets -> $after_secrets"
    echo "  Deployments: $before_deploys -> $after_deploys"
    echo ""

    local total_deleted=$((
        (before_pods - after_pods) +
        (before_configs - after_configs) +
        (before_secrets - after_secrets) +
        (before_deploys - after_deploys)
    ))

    print_success "Total resources deleted: $total_deleted"
}

# Interactive menu
show_menu() {
    print_header "Cleanup Menu"

    echo "What would you like to clean up?"
    echo ""
    echo "1) List all resources (no deletion)"
    echo "2) Delete workload resources only (keep ConfigMaps & Secrets)"
    echo "3) Delete all resources (keep namespace)"
    echo "4) Delete entire namespace"
    echo "5) Backup resources before cleanup"
    echo "6) Force cleanup stuck resources"
    echo "7) Exit"
    echo ""
    read -p "Enter your choice (1-7): " choice

    case $choice in
        1)
            list_resources
            ;;
        2)
            if confirm "Delete workload resources (pods, deployments, services)?"; then
                delete_resources_only
            fi
            ;;
        3)
            if confirm "Delete all resources in namespace '$NAMESPACE'?"; then
                delete_all_resources
            fi
            ;;
        4)
            delete_namespace_func
            return  # Exit after namespace deletion
            ;;
        5)
            backup_resources
            ;;
        6)
            force_cleanup_stuck_resources
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
RESOURCES_ONLY=false

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
        -r|--resources-only)
            RESOURCES_ONLY=true
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
    print_header "ConfigMaps and Secrets Cleanup"

    # Check if namespace exists
    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        print_error "Namespace '$NAMESPACE' not found"
        print_info "Nothing to clean up"
        exit 0
    fi

    # Get counts before cleanup
    before_pods=$(kubectl get pods -n "$NAMESPACE" --no-headers 2>/dev/null | wc -l)
    before_configs=$(kubectl get configmaps -n "$NAMESPACE" --no-headers 2>/dev/null | wc -l)
    before_secrets=$(kubectl get secrets -n "$NAMESPACE" --no-headers 2>/dev/null | wc -l)
    before_deploys=$(kubectl get deployments -n "$NAMESPACE" --no-headers 2>/dev/null | wc -l)

    # Execute based on flags
    if [ "$DELETE_NAMESPACE" = true ]; then
        delete_namespace_func
        exit 0
    elif [ "$RESOURCES_ONLY" = true ]; then
        if [ "$FORCE" = false ]; then
            if ! confirm "Delete workload resources in namespace '$NAMESPACE'?"; then
                print_info "Cleanup cancelled"
                exit 0
            fi
        fi
        delete_resources_only
        show_cleanup_stats "$before_pods" "$before_configs" "$before_secrets" "$before_deploys"
    elif [ "$FORCE" = true ]; then
        delete_all_resources
        show_cleanup_stats "$before_pods" "$before_configs" "$before_secrets" "$before_deploys"
    else
        # Interactive mode
        list_resources
        show_menu
    fi

    print_success "Cleanup complete!"
}

# Run main function
main "$@"
