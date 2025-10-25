#!/bin/bash

# test-configs.sh
# Comprehensive testing script for ConfigMaps and Secrets
# Tests various configuration patterns and usage scenarios

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
NAMESPACE="config-demo"
TEST_RESULTS=()
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_failure() {
    echo -e "${RED}[✗]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_header() {
    echo ""
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo ""
}

print_test_header() {
    echo ""
    echo -e "${BLUE}--- $1 ---${NC}"
}

# Function to record test result
record_test() {
    local test_name=$1
    local passed=$2
    local message=${3:-""}

    TOTAL_TESTS=$((TOTAL_TESTS + 1))

    if [ "$passed" = true ]; then
        PASSED_TESTS=$((PASSED_TESTS + 1))
        print_success "$test_name"
        TEST_RESULTS+=("✓ $test_name")
    else
        FAILED_TESTS=$((FAILED_TESTS + 1))
        print_failure "$test_name"
        if [ -n "$message" ]; then
            echo "  Reason: $message"
        fi
        TEST_RESULTS+=("✗ $test_name")
    fi
}

# Function to wait for pod to be ready
wait_for_pod() {
    local pod_name=$1
    local timeout=${2:-30}

    kubectl wait --for=condition=Ready pod "$pod_name" \
        -n "$NAMESPACE" --timeout="${timeout}s" &> /dev/null
}

# Test 1: ConfigMap Existence
test_configmap_existence() {
    print_test_header "Test 1: ConfigMap Existence"

    local configmaps=("app-config-literals" "app-config-files" "app-config-dev" "app-config-prod" "app-scripts")

    for cm in "${configmaps[@]}"; do
        if kubectl get configmap "$cm" -n "$NAMESPACE" &> /dev/null; then
            record_test "ConfigMap '$cm' exists" true
        else
            record_test "ConfigMap '$cm' exists" false "ConfigMap not found"
        fi
    done
}

# Test 2: Secret Existence
test_secret_existence() {
    print_test_header "Test 2: Secret Existence"

    local secrets=("app-secrets" "docker-registry-secret" "tls-secret" "ssh-auth-secret" "basic-auth-secret")

    for secret in "${secrets[@]}"; do
        if kubectl get secret "$secret" -n "$NAMESPACE" &> /dev/null; then
            record_test "Secret '$secret' exists" true
        else
            record_test "Secret '$secret' exists" false "Secret not found"
        fi
    done
}

# Test 3: ConfigMap Data Integrity
test_configmap_data() {
    print_test_header "Test 3: ConfigMap Data Integrity"

    # Check if specific keys exist in ConfigMap
    local app_name=$(kubectl get configmap app-config-literals -n "$NAMESPACE" \
        -o jsonpath='{.data.APP_NAME}' 2>/dev/null)

    if [ -n "$app_name" ]; then
        record_test "ConfigMap 'app-config-literals' contains APP_NAME" true
    else
        record_test "ConfigMap 'app-config-literals' contains APP_NAME" false "Key not found"
    fi

    # Check if nginx.conf exists in app-config-files
    local nginx_conf=$(kubectl get configmap app-config-files -n "$NAMESPACE" \
        -o jsonpath='{.data.nginx\.conf}' 2>/dev/null)

    if [ -n "$nginx_conf" ]; then
        record_test "ConfigMap 'app-config-files' contains nginx.conf" true
    else
        record_test "ConfigMap 'app-config-files' contains nginx.conf" false "Key not found"
    fi
}

# Test 4: Secret Data Integrity
test_secret_data() {
    print_test_header "Test 4: Secret Data Integrity"

    # Check if specific keys exist in Secret
    local db_user=$(kubectl get secret app-secrets -n "$NAMESPACE" \
        -o jsonpath='{.data.database-username}' 2>/dev/null)

    if [ -n "$db_user" ]; then
        # Decode base64
        local decoded=$(echo "$db_user" | base64 -d 2>/dev/null)
        if [ "$decoded" = "appuser" ]; then
            record_test "Secret 'app-secrets' database-username is correct" true
        else
            record_test "Secret 'app-secrets' database-username is correct" false "Value mismatch"
        fi
    else
        record_test "Secret 'app-secrets' database-username exists" false "Key not found"
    fi
}

# Test 5: Pod Using ConfigMap as Environment Variables
test_configmap_env_vars() {
    print_test_header "Test 5: ConfigMap as Environment Variables"

    local pod_name="pod-configmap-env"

    if kubectl get pod "$pod_name" -n "$NAMESPACE" &> /dev/null; then
        # Wait for pod to be ready
        if wait_for_pod "$pod_name" 30; then
            # Check pod logs for expected output
            local logs=$(kubectl logs "$pod_name" -n "$NAMESPACE" 2>/dev/null)

            if echo "$logs" | grep -q "APP_NAME:"; then
                record_test "Pod can access ConfigMap as env vars" true
            else
                record_test "Pod can access ConfigMap as env vars" false "Expected output not found in logs"
            fi
        else
            record_test "Pod '$pod_name' is ready" false "Pod not ready within timeout"
        fi
    else
        record_test "Pod '$pod_name' exists" false "Pod not found"
    fi
}

# Test 6: Pod Using Secret as Environment Variables
test_secret_env_vars() {
    print_test_header "Test 6: Secret as Environment Variables"

    local pod_name="pod-secret-env"

    if kubectl get pod "$pod_name" -n "$NAMESPACE" &> /dev/null; then
        if wait_for_pod "$pod_name" 30; then
            local logs=$(kubectl logs "$pod_name" -n "$NAMESPACE" 2>/dev/null)

            if echo "$logs" | grep -q "Database Username:"; then
                record_test "Pod can access Secret as env vars" true
            else
                record_test "Pod can access Secret as env vars" false "Expected output not found"
            fi
        else
            record_test "Pod '$pod_name' is ready" false "Pod not ready"
        fi
    else
        record_test "Pod '$pod_name' exists" false "Pod not found"
    fi
}

# Test 7: Pod Using ConfigMap as Volume
test_configmap_volume() {
    print_test_header "Test 7: ConfigMap as Volume Mount"

    local pod_name="pod-configmap-volume"

    if kubectl get pod "$pod_name" -n "$NAMESPACE" &> /dev/null; then
        if wait_for_pod "$pod_name" 30; then
            # Check if files are mounted
            local file_check=$(kubectl exec "$pod_name" -n "$NAMESPACE" -- \
                ls /etc/config/nginx.conf 2>/dev/null)

            if [ -n "$file_check" ]; then
                record_test "ConfigMap mounted as volume" true
            else
                record_test "ConfigMap mounted as volume" false "File not found in expected location"
            fi
        else
            record_test "Pod '$pod_name' is ready" false "Pod not ready"
        fi
    else
        record_test "Pod '$pod_name' exists" false "Pod not found"
    fi
}

# Test 8: Pod Using Secret as Volume
test_secret_volume() {
    print_test_header "Test 8: Secret as Volume Mount"

    local pod_name="pod-secret-volume"

    if kubectl get pod "$pod_name" -n "$NAMESPACE" &> /dev/null; then
        if wait_for_pod "$pod_name" 30; then
            # Check if secret files are mounted
            local file_check=$(kubectl exec "$pod_name" -n "$NAMESPACE" -- \
                ls /etc/secrets/database-username 2>/dev/null)

            if [ -n "$file_check" ]; then
                record_test "Secret mounted as volume" true
            else
                record_test "Secret mounted as volume" false "File not found"
            fi
        else
            record_test "Pod '$pod_name' is ready" false "Pod not ready"
        fi
    else
        record_test "Pod '$pod_name' exists" false "Pod not found"
    fi
}

# Test 9: Secret File Permissions
test_secret_permissions() {
    print_test_header "Test 9: Secret File Permissions"

    local pod_name="pod-secret-permissions"

    if kubectl get pod "$pod_name" -n "$NAMESPACE" &> /dev/null; then
        if wait_for_pod "$pod_name" 30; then
            # Check file permissions
            local perms=$(kubectl exec "$pod_name" -n "$NAMESPACE" -- \
                stat -c '%a' /etc/secrets/database-username 2>/dev/null)

            if [ "$perms" = "400" ]; then
                record_test "Secret files have correct permissions (400)" true
            else
                record_test "Secret files have correct permissions (400)" false "Permissions: $perms"
            fi
        else
            record_test "Pod '$pod_name' is ready" false "Pod not ready"
        fi
    else
        record_test "Pod '$pod_name' exists" false "Pod not found"
    fi
}

# Test 10: Deployment Using ConfigMaps
test_deployment_configmap() {
    print_test_header "Test 10: Deployment Using ConfigMaps"

    local deployment="demo-app"

    if kubectl get deployment "$deployment" -n "$NAMESPACE" &> /dev/null; then
        # Check if deployment is available
        local available=$(kubectl get deployment "$deployment" -n "$NAMESPACE" \
            -o jsonpath='{.status.availableReplicas}' 2>/dev/null)

        if [ -n "$available" ] && [ "$available" -gt 0 ]; then
            record_test "Deployment '$deployment' is running" true
        else
            record_test "Deployment '$deployment' is running" false "No available replicas"
        fi
    else
        record_test "Deployment '$deployment' exists" false "Deployment not found"
    fi
}

# Test 11: Deployment Using Secrets
test_deployment_secret() {
    print_test_header "Test 11: Deployment Using Secrets"

    local deployment="secure-app"

    if kubectl get deployment "$deployment" -n "$NAMESPACE" &> /dev/null; then
        local available=$(kubectl get deployment "$deployment" -n "$NAMESPACE" \
            -o jsonpath='{.status.availableReplicas}' 2>/dev/null)

        if [ -n "$available" ] && [ "$available" -gt 0 ]; then
            record_test "Deployment '$deployment' is running" true

            # Get a pod from the deployment
            local pod=$(kubectl get pods -n "$NAMESPACE" -l app=secure-app \
                -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)

            if [ -n "$pod" ]; then
                # Check if secrets are accessible
                local secret_check=$(kubectl exec "$pod" -n "$NAMESPACE" -- \
                    cat /etc/secrets/database-username 2>/dev/null)

                if [ -n "$secret_check" ]; then
                    record_test "Deployment can access Secrets" true
                else
                    record_test "Deployment can access Secrets" false "Cannot read secret file"
                fi
            fi
        else
            record_test "Deployment '$deployment' is running" false "No available replicas"
        fi
    else
        record_test "Deployment '$deployment' exists" false "Deployment not found"
    fi
}

# Test 12: EnvFrom ConfigMap
test_envfrom_configmap() {
    print_test_header "Test 12: envFrom ConfigMap"

    local pod_name="pod-configmap-envfrom"

    if kubectl get pod "$pod_name" -n "$NAMESPACE" &> /dev/null; then
        if wait_for_pod "$pod_name" 30; then
            local logs=$(kubectl logs "$pod_name" -n "$NAMESPACE" 2>/dev/null)

            if echo "$logs" | grep -q "APP_NAME" && echo "$logs" | grep -q "LOG_LEVEL"; then
                record_test "envFrom imports all ConfigMap keys" true
            else
                record_test "envFrom imports all ConfigMap keys" false "Expected keys not found"
            fi
        else
            record_test "Pod '$pod_name' is ready" false "Pod not ready"
        fi
    else
        record_test "Pod '$pod_name' exists" false "Pod not found"
    fi
}

# Test 13: EnvFrom Secret
test_envfrom_secret() {
    print_test_header "Test 13: envFrom Secret"

    local pod_name="pod-secret-envfrom"

    if kubectl get pod "$pod_name" -n "$NAMESPACE" &> /dev/null; then
        if wait_for_pod "$pod_name" 30; then
            local logs=$(kubectl logs "$pod_name" -n "$NAMESPACE" 2>/dev/null)

            if echo "$logs" | grep -q "database-username"; then
                record_test "envFrom imports all Secret keys" true
            else
                record_test "envFrom imports all Secret keys" false "Expected output not found"
            fi
        else
            record_test "Pod '$pod_name' is ready" false "Pod not ready"
        fi
    else
        record_test "Pod '$pod_name' exists" false "Pod not found"
    fi
}

# Test 14: Selective Key Mounting
test_selective_mounting() {
    print_test_header "Test 14: Selective Key Mounting"

    local pod_name="pod-configmap-selective"

    if kubectl get pod "$pod_name" -n "$NAMESPACE" &> /dev/null; then
        if wait_for_pod "$pod_name" 30; then
            # Check that only selected file is present
            local file_count=$(kubectl exec "$pod_name" -n "$NAMESPACE" -- \
                ls /etc/config/ 2>/dev/null | wc -l)

            if [ "$file_count" -eq 1 ]; then
                record_test "Only selected keys are mounted" true
            else
                record_test "Only selected keys are mounted" false "Found $file_count files, expected 1"
            fi
        else
            record_test "Pod '$pod_name' is ready" false "Pod not ready"
        fi
    else
        record_test "Pod '$pod_name' exists" false "Pod not found"
    fi
}

# Test 15: ConfigMap Update Detection
test_configmap_update() {
    print_test_header "Test 15: ConfigMap Update Detection"

    local configmap="app-config-literals"
    local original_value=$(kubectl get configmap "$configmap" -n "$NAMESPACE" \
        -o jsonpath='{.data.LOG_LEVEL}' 2>/dev/null)

    if [ -n "$original_value" ]; then
        record_test "Can read original ConfigMap value" true

        print_info "Original LOG_LEVEL: $original_value"
        print_info "Note: ConfigMap updates propagate to mounted volumes after ~60s"
        print_info "Environment variables do NOT update automatically"
    else
        record_test "Can read original ConfigMap value" false "Cannot read value"
    fi
}

# Function to print summary
print_summary() {
    print_header "Test Summary"

    echo "Total Tests: $TOTAL_TESTS"
    echo -e "Passed: ${GREEN}$PASSED_TESTS${NC}"
    echo -e "Failed: ${RED}$FAILED_TESTS${NC}"
    echo ""

    if [ $FAILED_TESTS -eq 0 ]; then
        echo -e "${GREEN}All tests passed!${NC} ✓"
    else
        echo -e "${YELLOW}Some tests failed.${NC}"
        echo ""
        echo "Failed tests:"
        for result in "${TEST_RESULTS[@]}"; do
            if [[ $result == ✗* ]]; then
                echo "  $result"
            fi
        done
    fi
    echo ""

    # Calculate success rate
    if [ $TOTAL_TESTS -gt 0 ]; then
        success_rate=$((PASSED_TESTS * 100 / TOTAL_TESTS))
        echo "Success rate: ${success_rate}%"
    fi
}

# Function to show diagnostic information
show_diagnostics() {
    print_header "Diagnostic Information"

    print_info "ConfigMaps in namespace:"
    kubectl get configmaps -n "$NAMESPACE" 2>/dev/null || print_warning "No ConfigMaps found"
    echo ""

    print_info "Secrets in namespace:"
    kubectl get secrets -n "$NAMESPACE" 2>/dev/null || print_warning "No Secrets found"
    echo ""

    print_info "Pods in namespace:"
    kubectl get pods -n "$NAMESPACE" 2>/dev/null || print_warning "No Pods found"
    echo ""

    print_info "Events (last 10):"
    kubectl get events -n "$NAMESPACE" --sort-by='.lastTimestamp' | tail -10 2>/dev/null || true
    echo ""
}

# Main execution
main() {
    print_header "ConfigMaps and Secrets Test Suite"

    print_info "Testing namespace: $NAMESPACE"
    echo ""

    # Run all tests
    test_configmap_existence
    test_secret_existence
    test_configmap_data
    test_secret_data
    test_configmap_env_vars
    test_secret_env_vars
    test_configmap_volume
    test_secret_volume
    test_secret_permissions
    test_deployment_configmap
    test_deployment_secret
    test_envfrom_configmap
    test_envfrom_secret
    test_selective_mounting
    test_configmap_update

    # Print summary
    print_summary

    # Show diagnostics if there were failures
    if [ $FAILED_TESTS -gt 0 ]; then
        show_diagnostics
    fi

    # Exit with appropriate code
    if [ $FAILED_TESTS -eq 0 ]; then
        exit 0
    else
        exit 1
    fi
}

# Run main function
main "$@"
