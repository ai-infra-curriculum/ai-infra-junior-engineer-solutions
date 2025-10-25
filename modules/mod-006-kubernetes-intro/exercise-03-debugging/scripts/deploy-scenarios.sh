#!/bin/bash

################################################################################
# Deploy Debugging Scenarios Script
#
# This script deploys one or all debugging scenarios for practice
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

Deploy debugging scenarios for practice

SCENARIOS:
    01 - Image Pull Error
    02 - CrashLoopBackOff
    03 - Resource Constraints
    04 - Service Connectivity Issues
    05 - Configuration Issues
    06 - Liveness and Readiness Probe Issues
    all - Deploy all scenarios

EXAMPLES:
    # Deploy scenario 1
    $0 01

    # Deploy all scenarios
    $0 all

    # Interactive mode (no arguments)
    $0

EOF
}

deploy_scenario() {
    local scenario_num="$1"
    local scenario_file="$SCENARIOS_DIR/${scenario_num}*.yaml"

    if ! ls $scenario_file &> /dev/null; then
        print_error "Scenario $scenario_num not found"
        return 1
    fi

    local file=$(ls $scenario_file | head -1)
    local scenario_name=$(basename "$file" .yaml | sed 's/^[0-9]*-//')

    print_header "Deploying Scenario $scenario_num: $scenario_name"

    print_info "Applying manifests from: $(basename $file)"
    if kubectl apply -f "$file"; then
        print_success "Scenario $scenario_num deployed successfully"

        # Extract namespace from file
        local namespace=$(grep -m1 "namespace:" "$file" | awk '{print $2}' | tr -d '"')

        echo ""
        print_info "Scenario deployed to namespace: $namespace"
        print_info "View resources:"
        echo "  kubectl get all -n $namespace"
        echo ""
        print_info "Debug using:"
        echo "  $SCRIPT_DIR/debug-master.sh -n $namespace check-all"
        echo ""
    else
        print_error "Failed to deploy scenario $scenario_num"
        return 1
    fi
}

list_scenarios() {
    print_header "Available Scenarios"
    echo ""

    for file in "$SCENARIOS_DIR"/*.yaml; do
        if [ -f "$file" ]; then
            local num=$(basename "$file" | cut -d'-' -f1)
            local name=$(basename "$file" .yaml | sed 's/^[0-9]*-//' | tr '-' ' ')
            local issue=$(grep "^# Problem:" "$file" | sed 's/^# Problem: //')

            printf "${BLUE}%2s${NC} - ${GREEN}%-30s${NC} %s\n" "$num" "$name" "$issue"
        fi
    done

    echo ""
}

deploy_all() {
    print_header "Deploying All Scenarios"

    for file in "$SCENARIOS_DIR"/*.yaml; do
        if [ -f "$file" ]; then
            local num=$(basename "$file" | cut -d'-' -f1)
            deploy_scenario "$num"
            echo ""
        fi
    done

    print_success "All scenarios deployed!"
    echo ""
    print_info "View all namespaces:"
    echo "  kubectl get namespaces | grep debug-scenario"
    echo ""
    print_info "To clean up all scenarios:"
    echo "  $SCRIPT_DIR/cleanup-scenarios.sh all"
}

interactive_mode() {
    list_scenarios

    echo -e "${BLUE}Enter scenario number (or 'all' for all scenarios, 'q' to quit):${NC}"
    read -p "> " choice

    case $choice in
        [0-9][0-9])
            deploy_scenario "$choice"
            ;;
        all)
            deploy_all
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
    deploy_all
elif [[ "$1" =~ ^[0-9]{2}$ ]]; then
    deploy_scenario "$1"
else
    print_error "Invalid argument: $1"
    show_usage
    exit 1
fi
