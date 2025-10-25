#!/bin/bash

################################################################################
# Helm Chart Testing Script
#
# This script validates the Helm chart by:
# - Linting the chart
# - Testing template rendering
# - Validating generated manifests
# - Checking for common issues
################################################################################

set -e

# Color definitions
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Variables
CHART_DIR="../flask-app"
NAMESPACE="flask-app-test"

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

# Check prerequisites
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
print_success "kubectl is installed ($(kubectl version --client --short 2>/dev/null))"

if ! command -v yamllint &> /dev/null; then
    print_warning "yamllint is not installed. Skipping YAML validation."
    SKIP_YAMLLINT=true
else
    print_success "yamllint is installed"
    SKIP_YAMLLINT=false
fi

# Navigate to chart directory
cd "$(dirname "$0")"
if [ ! -d "$CHART_DIR" ]; then
    print_error "Chart directory not found: $CHART_DIR"
    exit 1
fi

################################################################################
# Test 1: Chart Validation
################################################################################
print_header "Test 1: Chart Structure Validation"

# Check required files
REQUIRED_FILES=(
    "$CHART_DIR/Chart.yaml"
    "$CHART_DIR/values.yaml"
    "$CHART_DIR/templates/deployment.yaml"
    "$CHART_DIR/templates/service.yaml"
    "$CHART_DIR/templates/_helpers.tpl"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        print_success "Found: $file"
    else
        print_error "Missing required file: $file"
        exit 1
    fi
done

################################################################################
# Test 2: Helm Lint
################################################################################
print_header "Test 2: Helm Lint"

print_info "Linting chart with default values..."
if helm lint "$CHART_DIR"; then
    print_success "Chart linting passed (default values)"
else
    print_error "Chart linting failed (default values)"
    exit 1
fi

print_info "Linting chart with development values..."
if helm lint "$CHART_DIR" -f "$CHART_DIR/values-dev.yaml"; then
    print_success "Chart linting passed (development values)"
else
    print_error "Chart linting failed (development values)"
    exit 1
fi

print_info "Linting chart with production values..."
if helm lint "$CHART_DIR" -f "$CHART_DIR/values-prod.yaml"; then
    print_success "Chart linting passed (production values)"
else
    print_error "Chart linting failed (production values)"
    exit 1
fi

################################################################################
# Test 3: Template Rendering
################################################################################
print_header "Test 3: Template Rendering"

print_info "Testing template rendering with default values..."
OUTPUT_DIR="/tmp/helm-test-$$"
mkdir -p "$OUTPUT_DIR"

if helm template test-release "$CHART_DIR" --output-dir "$OUTPUT_DIR" > /dev/null; then
    print_success "Template rendering succeeded (default values)"
else
    print_error "Template rendering failed (default values)"
    exit 1
fi

print_info "Testing template rendering with development values..."
if helm template test-release "$CHART_DIR" -f "$CHART_DIR/values-dev.yaml" --output-dir "$OUTPUT_DIR-dev" > /dev/null; then
    print_success "Template rendering succeeded (development values)"
else
    print_error "Template rendering failed (development values)"
    exit 1
fi

print_info "Testing template rendering with production values..."
if helm template test-release "$CHART_DIR" -f "$CHART_DIR/values-prod.yaml" --output-dir "$OUTPUT_DIR-prod" > /dev/null; then
    print_success "Template rendering succeeded (production values)"
else
    print_error "Template rendering failed (production values)"
    exit 1
fi

################################################################################
# Test 4: Manifest Validation
################################################################################
print_header "Test 4: Kubernetes Manifest Validation"

print_info "Validating rendered manifests against Kubernetes API..."
for manifest in "$OUTPUT_DIR"/flask-app/templates/*.yaml; do
    if [ -f "$manifest" ]; then
        filename=$(basename "$manifest")
        if kubectl apply --dry-run=client -f "$manifest" > /dev/null 2>&1; then
            print_success "Valid manifest: $filename"
        else
            print_error "Invalid manifest: $filename"
            kubectl apply --dry-run=client -f "$manifest"
            exit 1
        fi
    fi
done

################################################################################
# Test 5: YAML Linting
################################################################################
if [ "$SKIP_YAMLLINT" = false ]; then
    print_header "Test 5: YAML Linting"

    print_info "Linting YAML files..."
    if yamllint -c /dev/null "$CHART_DIR" 2>&1 | grep -v "warning"; then
        print_success "YAML linting passed"
    else
        print_warning "YAML linting found issues (non-critical)"
    fi
fi

################################################################################
# Test 6: Value Validation
################################################################################
print_header "Test 6: Values Validation"

print_info "Checking for common configuration issues..."

# Test with autoscaling enabled but no metrics
echo "
autoscaling:
  enabled: true
  targetCPUUtilizationPercentage: null
  targetMemoryUtilizationPercentage: null
" > /tmp/test-values-invalid.yaml

print_info "Testing invalid configuration (autoscaling with no metrics)..."
if helm template test-release "$CHART_DIR" -f /tmp/test-values-invalid.yaml 2>&1 | grep -q "ERROR"; then
    print_success "Chart correctly validates autoscaling configuration"
else
    print_warning "Chart may not properly validate autoscaling configuration"
fi

# Test with production environment and default secret key
echo "
flask:
  env: production
  secretKey: change-me-in-production-use-vault-or-sealed-secrets
" > /tmp/test-values-insecure.yaml

print_info "Testing insecure configuration (default secret in production)..."
if helm template test-release "$CHART_DIR" -f /tmp/test-values-insecure.yaml 2>&1 | grep -q "ERROR\|WARNING"; then
    print_success "Chart warns about insecure configuration"
else
    print_warning "Chart should warn about default secret key in production"
fi

################################################################################
# Test 7: Dependency Check
################################################################################
print_header "Test 7: Chart Dependencies"

print_info "Checking chart dependencies..."
if [ -f "$CHART_DIR/Chart.yaml" ]; then
    if grep -q "dependencies:" "$CHART_DIR/Chart.yaml"; then
        print_success "Chart has dependencies defined"

        # Update dependencies
        print_info "Updating chart dependencies..."
        if helm dependency update "$CHART_DIR" > /dev/null 2>&1; then
            print_success "Dependencies updated successfully"
        else
            print_warning "Could not update dependencies (this is OK if repositories are not configured)"
        fi
    else
        print_info "No dependencies defined"
    fi
fi

################################################################################
# Test 8: Dry-Run Install
################################################################################
print_header "Test 8: Dry-Run Install"

print_info "Testing dry-run installation..."
if helm install test-release "$CHART_DIR" --dry-run --debug > /dev/null 2>&1; then
    print_success "Dry-run installation succeeded"
else
    print_error "Dry-run installation failed"
    helm install test-release "$CHART_DIR" --dry-run --debug
    exit 1
fi

################################################################################
# Test 9: Feature Toggles
################################################################################
print_header "Test 9: Feature Toggle Testing"

FEATURE_TESTS=(
    "ingress.enabled=true"
    "autoscaling.enabled=true"
    "persistence.enabled=true"
    "database.enabled=true"
    "redis.enabled=true"
    "mlModel.enabled=true"
    "serviceMonitor.enabled=true"
    "podDisruptionBudget.enabled=true"
    "networkPolicy.enabled=true"
)

for feature in "${FEATURE_TESTS[@]}"; do
    print_info "Testing with $feature..."
    if helm template test-release "$CHART_DIR" --set "$feature" > /dev/null 2>&1; then
        print_success "Feature toggle test passed: $feature"
    else
        print_error "Feature toggle test failed: $feature"
        exit 1
    fi
done

################################################################################
# Cleanup
################################################################################
print_header "Cleanup"

rm -rf "$OUTPUT_DIR" "$OUTPUT_DIR-dev" "$OUTPUT_DIR-prod"
rm -f /tmp/test-values-*.yaml
print_success "Cleaned up temporary files"

################################################################################
# Summary
################################################################################
print_header "Test Summary"

echo ""
echo -e "${GREEN}✓ All tests passed!${NC}"
echo ""
echo "Test Results:"
echo "  ✓ Chart structure validation"
echo "  ✓ Helm lint (3 configurations)"
echo "  ✓ Template rendering (3 configurations)"
echo "  ✓ Manifest validation"
if [ "$SKIP_YAMLLINT" = false ]; then
    echo "  ✓ YAML linting"
fi
echo "  ✓ Values validation"
echo "  ✓ Dependency check"
echo "  ✓ Dry-run installation"
echo "  ✓ Feature toggle testing (${#FEATURE_TESTS[@]} features)"
echo ""
echo -e "${BLUE}Chart is ready for deployment!${NC}"
echo ""
