#!/bin/bash

################################################################################
# Cleanup StatefulSets and Storage Examples
#
# This script removes all resources created for Exercise 04
################################################################################

set -e

# Color definitions
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Check if namespace exists
check_namespace() {
    if ! kubectl get namespace statefulset-demo &> /dev/null; then
        print_warning "Namespace 'statefulset-demo' does not exist"
        return 1
    fi
    return 0
}

# Show current resources
show_resources() {
    print_header "Current Resources"

    echo ""
    print_info "StatefulSets:"
    kubectl get statefulsets -n statefulset-demo 2>/dev/null || echo "None"

    echo ""
    print_info "Deployments:"
    kubectl get deployments -n statefulset-demo 2>/dev/null || echo "None"

    echo ""
    print_info "Pods:"
    kubectl get pods -n statefulset-demo 2>/dev/null || echo "None"

    echo ""
    print_info "Services:"
    kubectl get svc -n statefulset-demo 2>/dev/null || echo "None"

    echo ""
    print_info "PersistentVolumeClaims:"
    kubectl get pvc -n statefulset-demo 2>/dev/null || echo "None"
}

# Delete StatefulSets
delete_statefulsets() {
    print_header "Deleting StatefulSets"

    local statefulsets=$(kubectl get statefulsets -n statefulset-demo -o jsonpath='{.items[*].metadata.name}' 2>/dev/null)

    if [ -z "$statefulsets" ]; then
        print_info "No StatefulSets to delete"
        return 0
    fi

    for sts in $statefulsets; do
        print_info "Deleting StatefulSet: $sts"
        kubectl delete statefulset $sts -n statefulset-demo --timeout=60s
        print_success "$sts deleted"
    done
}

# Delete standalone pods
delete_pods() {
    print_header "Deleting Standalone Pods"

    local pods=$(kubectl get pods -n statefulset-demo --field-selector metadata.ownerReferences==null -o jsonpath='{.items[*].metadata.name}' 2>/dev/null)

    if [ -z "$pods" ]; then
        print_info "No standalone pods to delete"
        return 0
    fi

    for pod in $pods; do
        print_info "Deleting pod: $pod"
        kubectl delete pod $pod -n statefulset-demo --timeout=30s --force --grace-period=0 2>/dev/null || true
    done
}

# Delete PVCs
delete_pvcs() {
    print_header "Deleting PersistentVolumeClaims"

    local pvcs=$(kubectl get pvc -n statefulset-demo -o jsonpath='{.items[*].metadata.name}' 2>/dev/null)

    if [ -z "$pvcs" ]; then
        print_info "No PVCs to delete"
        return 0
    fi

    read -p "$(echo -e ${YELLOW}Warning: This will delete all persistent data! Continue? [y/N]:${NC} )" -n 1 -r
    echo

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        for pvc in $pvcs; do
            print_info "Deleting PVC: $pvc"
            kubectl delete pvc $pvc -n statefulset-demo --timeout=60s
            print_success "$pvc deleted"
        done
    else
        print_info "Skipping PVC deletion"
    fi
}

# Delete namespace
delete_namespace() {
    print_header "Deleting Namespace"

    read -p "$(echo -e ${YELLOW}Delete entire namespace 'statefulset-demo'? [y/N]:${NC} )" -n 1 -r
    echo

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Deleting namespace statefulset-demo..."
        kubectl delete namespace statefulset-demo --timeout=120s

        # Wait for namespace to be fully deleted
        print_info "Waiting for namespace deletion to complete..."
        while kubectl get namespace statefulset-demo &> /dev/null; do
            echo -n "."
            sleep 2
        done
        echo ""
        print_success "Namespace deleted"
    else
        print_info "Skipping namespace deletion"
    fi
}

# Cleanup orphaned PVs
cleanup_orphaned_pvs() {
    print_header "Checking for Orphaned PersistentVolumes"

    # Find PVs that were bound to deleted PVCs
    local orphaned_pvs=$(kubectl get pv -o json | jq -r '.items[] | select(.spec.claimRef.namespace=="statefulset-demo") | .metadata.name' 2>/dev/null)

    if [ -z "$orphaned_pvs" ]; then
        print_info "No orphaned PVs found"
        return 0
    fi

    print_warning "Found orphaned PersistentVolumes:"
    echo "$orphaned_pvs"
    echo ""

    read -p "$(echo -e ${YELLOW}Delete orphaned PVs? [y/N]:${NC} )" -n 1 -r
    echo

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        for pv in $orphaned_pvs; do
            print_info "Deleting PV: $pv"
            kubectl delete pv $pv --timeout=30s
            print_success "$pv deleted"
        done
    else
        print_info "Skipping orphaned PV deletion"
        print_warning "You may need to manually delete these PVs later"
    fi
}

# Quick cleanup (no prompts)
quick_cleanup() {
    print_header "Quick Cleanup (No Prompts)"

    if ! check_namespace; then
        print_info "Nothing to clean up"
        return 0
    fi

    print_info "Deleting namespace statefulset-demo and all resources..."
    kubectl delete namespace statefulset-demo --timeout=120s

    print_info "Waiting for cleanup to complete..."
    while kubectl get namespace statefulset-demo &> /dev/null; do
        echo -n "."
        sleep 2
    done
    echo ""

    print_success "Cleanup complete"
    cleanup_orphaned_pvs
}

# Interactive cleanup
interactive_cleanup() {
    if ! check_namespace; then
        print_info "Nothing to clean up"
        return 0
    fi

    show_resources

    echo ""
    echo "Cleanup Options:"
    echo "1. Delete everything (namespace and all resources)"
    echo "2. Delete StatefulSets only (keep PVCs)"
    echo "3. Delete StatefulSets and PVCs"
    echo "4. Cancel"
    echo ""

    read -p "Select option [1-4]: " option

    case $option in
        1)
            delete_namespace
            cleanup_orphaned_pvs
            ;;
        2)
            delete_statefulsets
            delete_pods
            print_info "PVCs preserved for reuse"
            ;;
        3)
            delete_statefulsets
            delete_pods
            delete_pvcs
            ;;
        4)
            print_info "Cleanup cancelled"
            return 0
            ;;
        *)
            print_error "Invalid option"
            return 1
            ;;
    esac

    print_success "Cleanup complete"
}

# Show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Cleanup StatefulSets and Storage examples

OPTIONS:
    -q, --quick     Quick cleanup (delete namespace without prompts)
    -h, --help      Show this help message

EXAMPLES:
    # Interactive cleanup (recommended)
    $0

    # Quick cleanup (no prompts)
    $0 --quick

EOF
}

# Main execution
main() {
    if [ "$1" = "-q" ] || [ "$1" = "--quick" ]; then
        quick_cleanup
    elif [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
        show_usage
    else
        interactive_cleanup
    fi
}

main "$@"
