#!/bin/bash

################################################################################
# Cleanup Debugging Scenarios Script
#
# This script removes deployed debugging scenarios
################################################################################

set -e

# Color definitions
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCENARIOS_DIR="$SCRIPT_DIR/../scenarios"

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

show_usage() {
    cat << EOF
Usage: $0 [SCENARIO_NUMBER | all]

Clean up debugging scenarios

ARGUMENTS:
    01-06  - Clean up specific scenario
    all    - Clean up all scenarios

EXAMPLES:
    # Clean up scenario 1
    $0 01

    # Clean up all scenarios
    $0 all

    # Interactive mode (no arguments)
    $0

EOF
}

cleanup_scenario() {
    local scenario_num="$1"
    local namespace="debug-scenario-${scenario_num}"

    print_header "Cleaning Up Scenario $scenario_num"

    if ! kubectl get namespace "$namespace" &> /dev/null; then
        print_warning "Namespace $namespace not found (already cleaned up?)"
        return 0
    fi

    print_info "Deleting namespace: $namespace"
    if kubectl delete namespace "$namespace" --timeout=60s; then
        print_success "Scenario $scenario_num cleaned up successfully"
    else
        print_error "Failed to clean up scenario $scenario_num"
        return 1
    fi
}

cleanup_all() {
    print_header "Cleaning Up All Scenarios"

    # Find all debug-scenario namespaces
    local namespaces=$(kubectl get namespaces -o jsonpath='{.items[?(@.metadata.labels.exercise=="debugging")].metadata.name}')

    if [ -z "$namespaces" ]; then
        print_info "No debugging scenarios found"
        return 0
    fi

    print_info "Found scenarios in namespaces: $namespaces"
    echo ""

    for ns in $namespaces; do
        print_info "Deleting namespace: $ns"
        kubectl delete namespace "$ns" --timeout=60s &
    done

    # Wait for all deletions to complete
    wait

    print_success "All scenarios cleaned up!"
}

list_deployed_scenarios() {
    print_header "Deployed Scenarios"

    local namespaces=$(kubectl get namespaces -o jsonpath='{.items[?(@.metadata.labels.exercise=="debugging")].metadata.name}')

    if [ -z "$namespaces" ]; then
        print_info "No debugging scenarios currently deployed"
        return 0
    fi

    for ns in $namespaces; do
        local scenario=$(echo "$ns" | sed 's/debug-scenario-//')
        local pod_count=$(kubectl get pods -n "$ns" --no-headers 2>/dev/null | wc -l)
        echo -e "  ${BLUE}Scenario $scenario${NC} - Namespace: $ns (Pods: $pod_count)"
    done

    echo ""
}

interactive_mode() {
    list_deployed_scenarios

    echo -e "${BLUE}Enter scenario number to clean up (or 'all' for all scenarios, 'q' to quit):${NC}"
    read -p "> " choice

    case $choice in
        [0-9][0-9])
            cleanup_scenario "$choice"
            ;;
        all)
            cleanup_all
            ;;
        q|quit|exit)
            exit 0
            ;;
        *)
            print_error "Invalid choice: $choice"
            exit 1
            ;;
    esac
}

# Check prerequisites
if ! command -v kubectl &> /dev/null; then
    print_error "kubectl is not installed"
    exit 1
fi

if ! kubectl cluster-info &> /dev/null; then
    print_error "Cannot connect to Kubernetes cluster"
    exit 1
fi

# Parse arguments
if [ $# -eq 0 ]; then
    interactive_mode
elif [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    show_usage
elif [ "$1" = "all" ]; then
    cleanup_all
elif [[ "$1" =~ ^[0-9]{2}$ ]]; then
    cleanup_scenario "$1"
else
    print_error "Invalid argument: $1"
    show_usage
    exit 1
fi
