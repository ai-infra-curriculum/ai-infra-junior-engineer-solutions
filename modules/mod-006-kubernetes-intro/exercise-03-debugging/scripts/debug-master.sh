#!/bin/bash

################################################################################
# Master Debugging Script
#
# This script provides comprehensive debugging capabilities for Kubernetes
# applications, covering common issues and troubleshooting scenarios.
################################################################################

set -e

# Color definitions
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Default values
NAMESPACE=""
POD_NAME=""
DEPLOYMENT_NAME=""
SERVICE_NAME=""
VERBOSE=false

# Helper functions
print_header() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_subheader() {
    echo ""
    echo -e "${CYAN}--- $1${NC}"
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
Usage: $0 [OPTIONS] [COMMAND]

Kubernetes debugging toolkit for common troubleshooting scenarios

OPTIONS:
    -n, --namespace NAMESPACE    Kubernetes namespace to debug
    -p, --pod POD_NAME          Pod name to debug
    -d, --deployment NAME       Deployment name to debug
    -s, --service NAME          Service name to debug
    -v, --verbose               Verbose output
    -h, --help                  Show this help message

COMMANDS:
    check-all          Run all diagnostic checks
    check-pods         Check pod status and health
    check-services     Check service configuration
    check-network      Check networking and connectivity
    check-resources    Check resource usage and constraints
    check-config       Check ConfigMaps and Secrets
    check-logs         View and analyze logs
    check-events       View recent events
    interactive        Interactive debugging mode

EXAMPLES:
    # Check all components in a namespace
    $0 -n my-app check-all

    # Debug specific pod
    $0 -n my-app -p my-pod-123 check-pods

    # Interactive mode
    $0 -n my-app interactive

EOF
}

# Parse command-line arguments
COMMAND=""
while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        -p|--pod)
            POD_NAME="$2"
            shift 2
            ;;
        -d|--deployment)
            DEPLOYMENT_NAME="$2"
            shift 2
            ;;
        -s|--service)
            SERVICE_NAME="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        check-all|check-pods|check-services|check-network|check-resources|check-config|check-logs|check-events|interactive)
            COMMAND="$1"
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Validate prerequisites
check_prerequisites() {
    if ! command -v kubectl &> /dev/null; then
        print_error "kubectl is not installed"
        exit 1
    fi

    if ! kubectl cluster-info &> /dev/null; then
        print_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi

    if [ -z "$NAMESPACE" ]; then
        print_error "Namespace is required. Use -n or --namespace"
        exit 1
    fi

    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        print_error "Namespace '$NAMESPACE' does not exist"
        exit 1
    fi
}

################################################################################
# Check Pod Status
################################################################################
check_pods() {
    print_header "Pod Status Check"

    if [ -n "$POD_NAME" ]; then
        # Check specific pod
        print_info "Checking pod: $POD_NAME"

        print_subheader "Pod Status"
        kubectl get pod "$POD_NAME" -n "$NAMESPACE" -o wide

        print_subheader "Pod Description"
        kubectl describe pod "$POD_NAME" -n "$NAMESPACE"

        print_subheader "Container Status"
        kubectl get pod "$POD_NAME" -n "$NAMESPACE" -o jsonpath='{.status.containerStatuses[*]}' | jq

        print_subheader "Pod Conditions"
        kubectl get pod "$POD_NAME" -n "$NAMESPACE" -o jsonpath='{.status.conditions[*]}' | jq

        # Check for common issues
        STATUS=$(kubectl get pod "$POD_NAME" -n "$NAMESPACE" -o jsonpath='{.status.phase}')
        case $STATUS in
            Pending)
                print_warning "Pod is Pending. Common causes:"
                echo "  - Insufficient resources"
                echo "  - Image pull errors"
                echo "  - Volume mount issues"
                echo "  - Node selector/affinity constraints"
                ;;
            Running)
                RESTARTS=$(kubectl get pod "$POD_NAME" -n "$NAMESPACE" -o jsonpath='{.status.containerStatuses[0].restartCount}')
                if [ "$RESTARTS" -gt 0 ]; then
                    print_warning "Pod has restarted $RESTARTS times"
                    echo "  Check liveness probe configuration"
                    echo "  Check application logs for crashes"
                fi
                print_success "Pod is Running"
                ;;
            Failed|CrashLoopBackOff)
                print_error "Pod is in Failed/CrashLoopBackOff state"
                echo "  Check logs: kubectl logs $POD_NAME -n $NAMESPACE --previous"
                echo "  Common causes:"
                echo "    - Application errors"
                echo "    - Missing dependencies"
                echo "    - Configuration issues"
                ;;
            *)
                print_warning "Pod status: $STATUS"
                ;;
        esac

    else
        # Check all pods in namespace
        print_info "Checking all pods in namespace: $NAMESPACE"

        print_subheader "Pod List"
        kubectl get pods -n "$NAMESPACE" -o wide

        print_subheader "Pod Issues Summary"

        # Count pods by status
        TOTAL=$(kubectl get pods -n "$NAMESPACE" --no-headers | wc -l)
        RUNNING=$(kubectl get pods -n "$NAMESPACE" --field-selector=status.phase=Running --no-headers | wc -l)
        PENDING=$(kubectl get pods -n "$NAMESPACE" --field-selector=status.phase=Pending --no-headers | wc -l)
        FAILED=$(kubectl get pods -n "$NAMESPACE" --field-selector=status.phase=Failed --no-headers | wc -l)

        echo "Total Pods:   $TOTAL"
        echo "Running:      $RUNNING"
        echo "Pending:      $PENDING"
        echo "Failed:       $FAILED"

        # Show pods with issues
        if [ "$PENDING" -gt 0 ]; then
            print_warning "Pending pods:"
            kubectl get pods -n "$NAMESPACE" --field-selector=status.phase=Pending
        fi

        if [ "$FAILED" -gt 0 ]; then
            print_error "Failed pods:"
            kubectl get pods -n "$NAMESPACE" --field-selector=status.phase=Failed
        fi

        # Check for high restart counts
        print_subheader "Pods with Restarts"
        kubectl get pods -n "$NAMESPACE" --no-headers | awk '$4 > 0 {print $0}'
    fi
}

################################################################################
# Check Services
################################################################################
check_services() {
    print_header "Service Check"

    if [ -n "$SERVICE_NAME" ]; then
        # Check specific service
        print_info "Checking service: $SERVICE_NAME"

        print_subheader "Service Details"
        kubectl get svc "$SERVICE_NAME" -n "$NAMESPACE" -o wide

        print_subheader "Service Description"
        kubectl describe svc "$SERVICE_NAME" -n "$NAMESPACE"

        print_subheader "Service Endpoints"
        kubectl get endpoints "$SERVICE_NAME" -n "$NAMESPACE"

        # Check if service has endpoints
        ENDPOINTS=$(kubectl get endpoints "$SERVICE_NAME" -n "$NAMESPACE" -o jsonpath='{.subsets[*].addresses[*].ip}')
        if [ -z "$ENDPOINTS" ]; then
            print_error "Service has NO endpoints!"
            echo "  Common causes:"
            echo "    - Label selector doesn't match any pods"
            echo "    - Pods are not ready (readiness probe failing)"
            echo "    - Pods don't exist"

            print_subheader "Service Selector"
            kubectl get svc "$SERVICE_NAME" -n "$NAMESPACE" -o jsonpath='{.spec.selector}' | jq

            print_subheader "Matching Pods"
            SELECTOR=$(kubectl get svc "$SERVICE_NAME" -n "$NAMESPACE" -o jsonpath='{.spec.selector}' | jq -r 'to_entries|map("\(.key)=\(.value)")|join(",")')
            kubectl get pods -n "$NAMESPACE" -l "$SELECTOR" --show-labels
        else
            print_success "Service has endpoints: $ENDPOINTS"
        fi

    else
        # Check all services
        print_info "Checking all services in namespace: $NAMESPACE"

        print_subheader "Service List"
        kubectl get svc -n "$NAMESPACE" -o wide

        print_subheader "Services Without Endpoints"
        for svc in $(kubectl get svc -n "$NAMESPACE" -o jsonpath='{.items[*].metadata.name}'); do
            ENDPOINTS=$(kubectl get endpoints "$svc" -n "$NAMESPACE" -o jsonpath='{.subsets[*].addresses[*].ip}')
            if [ -z "$ENDPOINTS" ]; then
                print_warning "Service $svc has no endpoints"
            fi
        done
    fi
}

################################################################################
# Check Network Connectivity
################################################################################
check_network() {
    print_header "Network Connectivity Check"

    print_subheader "Network Policies"
    if kubectl get networkpolicies -n "$NAMESPACE" &> /dev/null; then
        kubectl get networkpolicies -n "$NAMESPACE"
    else
        print_info "No NetworkPolicies found"
    fi

    if [ -n "$POD_NAME" ]; then
        print_subheader "Pod Network Info"
        kubectl get pod "$POD_NAME" -n "$NAMESPACE" -o jsonpath='{.status.podIP}'
        echo ""

        print_subheader "DNS Resolution Test"
        kubectl exec "$POD_NAME" -n "$NAMESPACE" -- nslookup kubernetes.default 2>/dev/null || \
            print_warning "DNS test failed (nslookup not available in container)"

        if [ -n "$SERVICE_NAME" ]; then
            print_subheader "Service Connectivity Test"
            print_info "Testing connection to $SERVICE_NAME..."
            kubectl exec "$POD_NAME" -n "$NAMESPACE" -- wget -O- --timeout=5 "http://$SERVICE_NAME" 2>/dev/null || \
                print_error "Cannot connect to service $SERVICE_NAME"
        fi
    fi

    print_subheader "CoreDNS Status"
    kubectl get pods -n kube-system -l k8s-app=kube-dns
}

################################################################################
# Check Resources
################################################################################
check_resources() {
    print_header "Resource Usage Check"

    print_subheader "Node Resources"
    kubectl top nodes 2>/dev/null || print_warning "Metrics server not available"

    print_subheader "Namespace Resource Quotas"
    kubectl get resourcequota -n "$NAMESPACE" 2>/dev/null || print_info "No resource quotas"

    if [ -n "$POD_NAME" ]; then
        print_subheader "Pod Resource Usage"
        kubectl top pod "$POD_NAME" -n "$NAMESPACE" 2>/dev/null || print_warning "Metrics not available"

        print_subheader "Pod Resource Requests/Limits"
        kubectl get pod "$POD_NAME" -n "$NAMESPACE" -o jsonpath='{.spec.containers[*].resources}' | jq
    else
        print_subheader "All Pods Resource Usage"
        kubectl top pods -n "$NAMESPACE" 2>/dev/null || print_warning "Metrics server not available"
    fi

    print_subheader "Recent Resource Events"
    kubectl get events -n "$NAMESPACE" --field-selector reason=OOMKilled,reason=Evicted --sort-by='.lastTimestamp' | tail -10
}

################################################################################
# Check Configuration
################################################################################
check_config() {
    print_header "Configuration Check"

    print_subheader "ConfigMaps"
    kubectl get configmaps -n "$NAMESPACE"

    print_subheader "Secrets"
    kubectl get secrets -n "$NAMESPACE"

    if [ -n "$POD_NAME" ]; then
        print_subheader "Pod Environment Variables"
        kubectl exec "$POD_NAME" -n "$NAMESPACE" -- env 2>/dev/null | sort || \
            print_warning "Cannot access environment variables"

        print_subheader "Pod Volume Mounts"
        kubectl get pod "$POD_NAME" -n "$NAMESPACE" -o jsonpath='{.spec.containers[*].volumeMounts}' | jq
    fi
}

################################################################################
# Check Logs
################################################################################
check_logs() {
    print_header "Log Analysis"

    if [ -z "$POD_NAME" ]; then
        print_error "Pod name is required for log checks. Use -p or --pod"
        return 1
    fi

    print_subheader "Current Logs (last 50 lines)"
    kubectl logs "$POD_NAME" -n "$NAMESPACE" --tail=50

    print_subheader "Previous Container Logs"
    kubectl logs "$POD_NAME" -n "$NAMESPACE" --previous --tail=50 2>/dev/null || \
        print_info "No previous container logs available"

    print_subheader "Log Analysis Summary"
    ERROR_COUNT=$(kubectl logs "$POD_NAME" -n "$NAMESPACE" --tail=1000 2>/dev/null | grep -ic error || echo 0)
    WARN_COUNT=$(kubectl logs "$POD_NAME" -n "$NAMESPACE" --tail=1000 2>/dev/null | grep -ic warn || echo 0)

    echo "Errors in last 1000 lines: $ERROR_COUNT"
    echo "Warnings in last 1000 lines: $WARN_COUNT"

    if [ "$ERROR_COUNT" -gt 0 ]; then
        print_subheader "Recent Errors"
        kubectl logs "$POD_NAME" -n "$NAMESPACE" --tail=1000 2>/dev/null | grep -i error | tail -10
    fi
}

################################################################################
# Check Events
################################################################################
check_events() {
    print_header "Event Check"

    print_subheader "Recent Events (last 20)"
    kubectl get events -n "$NAMESPACE" --sort-by='.lastTimestamp' | tail -20

    print_subheader "Warning Events"
    kubectl get events -n "$NAMESPACE" --field-selector type=Warning --sort-by='.lastTimestamp' | tail -10

    if [ -n "$POD_NAME" ]; then
        print_subheader "Events for Pod: $POD_NAME"
        kubectl get events -n "$NAMESPACE" --field-selector involvedObject.name="$POD_NAME" --sort-by='.lastTimestamp'
    fi
}

################################################################################
# Run All Checks
################################################################################
check_all() {
    print_header "Running All Diagnostic Checks"
    print_info "Namespace: $NAMESPACE"

    check_pods
    check_services
    check_network
    check_resources
    check_config
    check_events

    if [ -n "$POD_NAME" ]; then
        check_logs
    fi

    print_header "Diagnostic Check Complete"
}

################################################################################
# Interactive Mode
################################################################################
interactive_mode() {
    while true; do
        echo ""
        echo -e "${CYAN}========================================${NC}"
        echo -e "${CYAN}Interactive Debugging Menu${NC}"
        echo -e "${CYAN}========================================${NC}"
        echo "Namespace: $NAMESPACE"
        [ -n "$POD_NAME" ] && echo "Pod: $POD_NAME"
        [ -n "$SERVICE_NAME" ] && echo "Service: $SERVICE_NAME"
        echo ""
        echo "1. Check pod status"
        echo "2. Check services"
        echo "3. Check network"
        echo "4. Check resources"
        echo "5. Check configuration"
        echo "6. View logs"
        echo "7. View events"
        echo "8. Run all checks"
        echo "9. Change pod/service"
        echo "0. Exit"
        echo ""
        read -p "Select option: " option

        case $option in
            1) check_pods ;;
            2) check_services ;;
            3) check_network ;;
            4) check_resources ;;
            5) check_config ;;
            6) check_logs ;;
            7) check_events ;;
            8) check_all ;;
            9)
                read -p "Enter pod name (leave empty to check all): " POD_NAME
                read -p "Enter service name (optional): " SERVICE_NAME
                ;;
            0) break ;;
            *) print_error "Invalid option" ;;
        esac
    done
}

################################################################################
# Main
################################################################################
check_prerequisites

if [ -z "$COMMAND" ]; then
    show_usage
    exit 1
fi

case $COMMAND in
    check-all) check_all ;;
    check-pods) check_pods ;;
    check-services) check_services ;;
    check-network) check_network ;;
    check-resources) check_resources ;;
    check-config) check_config ;;
    check-logs) check_logs ;;
    check-events) check_events ;;
    interactive) interactive_mode ;;
    *)
        print_error "Unknown command: $COMMAND"
        show_usage
        exit 1
        ;;
esac
