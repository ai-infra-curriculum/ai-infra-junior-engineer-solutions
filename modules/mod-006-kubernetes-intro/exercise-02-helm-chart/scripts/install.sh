#!/bin/bash

################################################################################
# Helm Chart Installation Script
#
# This script installs the Flask application Helm chart with support for
# different environments (dev, staging, production).
################################################################################

set -e

# Color definitions
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
RELEASE_NAME="flask-app"
NAMESPACE="default"
ENVIRONMENT="dev"
CHART_DIR="../flask-app"
DRY_RUN=false
WAIT=true
TIMEOUT="5m"
CREATE_NAMESPACE=true

# Helper functions
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
Usage: $0 [OPTIONS]

Install Flask application Helm chart

OPTIONS:
    -n, --name NAME          Release name (default: flask-app)
    -s, --namespace NAMESPACE Kubernetes namespace (default: default)
    -e, --environment ENV    Environment: dev, staging, prod (default: dev)
    -c, --chart PATH         Chart directory path (default: ../flask-app)
    -d, --dry-run            Perform a dry-run installation
    --no-wait                Don't wait for resources to be ready
    --no-create-namespace    Don't create namespace if it doesn't exist
    -t, --timeout DURATION   Timeout for installation (default: 5m)
    -h, --help               Show this help message

EXAMPLES:
    # Install in development environment
    $0 -e dev

    # Install in production with custom release name
    $0 -n my-app -e prod -s production

    # Dry-run installation
    $0 -e prod --dry-run

EOF
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--name)
            RELEASE_NAME="$2"
            shift 2
            ;;
        -s|--namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -c|--chart)
            CHART_DIR="$2"
            shift 2
            ;;
        -d|--dry-run)
            DRY_RUN=true
            shift
            ;;
        --no-wait)
            WAIT=false
            shift
            ;;
        --no-create-namespace)
            CREATE_NAMESPACE=false
            shift
            ;;
        -t|--timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Validate environment
if [[ ! "$ENVIRONMENT" =~ ^(dev|staging|prod)$ ]]; then
    print_error "Invalid environment: $ENVIRONMENT. Must be dev, staging, or prod."
    exit 1
fi

# Navigate to script directory
cd "$(dirname "$0")"

################################################################################
# Prerequisites Check
################################################################################
print_header "Checking Prerequisites"

if ! command -v helm &> /dev/null; then
    print_error "Helm is not installed. Please install Helm first."
    exit 1
fi
print_success "Helm is installed ($(helm version --short))"

if ! command -v kubectl &> /dev/null; then
    print_error "kubectl is not installed. Please install kubectl first."
    exit 1
fi
print_success "kubectl is installed"

# Check cluster connectivity
if ! kubectl cluster-info &> /dev/null; then
    print_error "Cannot connect to Kubernetes cluster"
    exit 1
fi
print_success "Connected to Kubernetes cluster"

# Verify chart exists
if [ ! -d "$CHART_DIR" ]; then
    print_error "Chart directory not found: $CHART_DIR"
    exit 1
fi
print_success "Chart directory found: $CHART_DIR"

################################################################################
# Environment Configuration
################################################################################
print_header "Environment Configuration"

echo "Release Name:  $RELEASE_NAME"
echo "Namespace:     $NAMESPACE"
echo "Environment:   $ENVIRONMENT"
echo "Chart Path:    $CHART_DIR"
echo "Dry Run:       $DRY_RUN"
echo "Wait:          $WAIT"
echo "Timeout:       $TIMEOUT"

# Determine values file
case $ENVIRONMENT in
    dev)
        VALUES_FILE="$CHART_DIR/values-dev.yaml"
        ;;
    staging)
        VALUES_FILE="$CHART_DIR/values-staging.yaml"
        if [ ! -f "$VALUES_FILE" ]; then
            print_warning "Staging values file not found, using production values with reduced resources"
            VALUES_FILE="$CHART_DIR/values-prod.yaml"
        fi
        ;;
    prod)
        VALUES_FILE="$CHART_DIR/values-prod.yaml"
        ;;
esac

if [ ! -f "$VALUES_FILE" ]; then
    print_error "Values file not found: $VALUES_FILE"
    exit 1
fi
print_success "Values file: $VALUES_FILE"

################################################################################
# Pre-installation Checks
################################################################################
print_header "Pre-installation Checks"

# Check if release already exists
if helm list -n "$NAMESPACE" 2>/dev/null | grep -q "^$RELEASE_NAME"; then
    print_error "Release '$RELEASE_NAME' already exists in namespace '$NAMESPACE'"
    print_info "Use upgrade.sh script to upgrade existing release"
    exit 1
fi
print_success "Release name is available"

# Check if namespace exists
if kubectl get namespace "$NAMESPACE" &> /dev/null; then
    print_info "Namespace '$NAMESPACE' already exists"
else
    if [ "$CREATE_NAMESPACE" = true ]; then
        print_info "Namespace '$NAMESPACE' will be created"
    else
        print_error "Namespace '$NAMESPACE' does not exist and --no-create-namespace was specified"
        exit 1
    fi
fi

################################################################################
# Update Dependencies
################################################################################
print_header "Updating Chart Dependencies"

print_info "Checking for chart dependencies..."
if grep -q "dependencies:" "$CHART_DIR/Chart.yaml"; then
    print_info "Updating dependencies..."
    if helm dependency update "$CHART_DIR" 2>&1 | grep -v "WARNING"; then
        print_success "Dependencies updated"
    else
        print_warning "Could not update all dependencies (proceeding anyway)"
    fi
else
    print_info "No dependencies to update"
fi

################################################################################
# Validate Chart
################################################################################
print_header "Validating Chart"

print_info "Linting chart..."
if helm lint "$CHART_DIR" -f "$VALUES_FILE" > /dev/null 2>&1; then
    print_success "Chart validation passed"
else
    print_error "Chart validation failed"
    helm lint "$CHART_DIR" -f "$VALUES_FILE"
    exit 1
fi

################################################################################
# Security Checks
################################################################################
if [ "$ENVIRONMENT" = "prod" ]; then
    print_header "Production Security Checks"

    # Check for default secret key
    if grep -q "change-me-in-production" "$VALUES_FILE"; then
        print_error "Default secret key detected in production values file!"
        print_error "Please update flask.secretKey before deploying to production"
        exit 1
    fi
    print_success "No default secret keys found"

    # Check debug mode
    if grep -q "debug: true" "$VALUES_FILE"; then
        print_warning "Debug mode is enabled in production values file"
        read -p "Continue anyway? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    print_success "Debug mode is disabled"
fi

################################################################################
# Installation
################################################################################
print_header "Installing Helm Chart"

# Build helm install command
HELM_CMD="helm install $RELEASE_NAME $CHART_DIR"
HELM_CMD="$HELM_CMD --namespace $NAMESPACE"
HELM_CMD="$HELM_CMD --values $VALUES_FILE"

if [ "$CREATE_NAMESPACE" = true ]; then
    HELM_CMD="$HELM_CMD --create-namespace"
fi

if [ "$WAIT" = true ]; then
    HELM_CMD="$HELM_CMD --wait --timeout $TIMEOUT"
fi

if [ "$DRY_RUN" = true ]; then
    HELM_CMD="$HELM_CMD --dry-run --debug"
fi

print_info "Executing: $HELM_CMD"
echo ""

if eval "$HELM_CMD"; then
    if [ "$DRY_RUN" = true ]; then
        print_success "Dry-run installation completed successfully"
    else
        print_success "Installation completed successfully"
    fi
else
    print_error "Installation failed"
    exit 1
fi

################################################################################
# Post-installation
################################################################################
if [ "$DRY_RUN" = false ]; then
    print_header "Post-installation Information"

    # Show release status
    echo ""
    print_info "Release Status:"
    helm status "$RELEASE_NAME" -n "$NAMESPACE"

    echo ""
    print_info "Deployed Resources:"
    kubectl get all -n "$NAMESPACE" -l "app.kubernetes.io/instance=$RELEASE_NAME"

    echo ""
    print_info "Useful Commands:"
    echo ""
    echo "  # Check pod status"
    echo "  kubectl get pods -n $NAMESPACE -l app.kubernetes.io/instance=$RELEASE_NAME"
    echo ""
    echo "  # View logs"
    echo "  kubectl logs -n $NAMESPACE -l app.kubernetes.io/instance=$RELEASE_NAME --tail=100 -f"
    echo ""
    echo "  # Access the application (port-forward)"
    echo "  kubectl port-forward -n $NAMESPACE svc/$RELEASE_NAME-flask-app 8080:80"
    echo ""
    echo "  # Uninstall the release"
    echo "  helm uninstall $RELEASE_NAME -n $NAMESPACE"
    echo ""

    print_success "Installation complete!"
else
    print_success "Dry-run complete!"
fi
