#!/bin/bash
# Test script for Exercise 01
# Validates that all deployed resources are working correctly

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

NAMESPACE="exercise-01"
FAILED_TESTS=0

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Testing Exercise 01 Deployments${NC}"
echo -e "${GREEN}========================================${NC}"

# Test function
test_command() {
    local description=$1
    local command=$2
    local expected=$3

    echo -e "\n${YELLOW}Test: $description${NC}"
    if eval "$command" &>/dev/null; then
        echo -e "${GREEN}✓ PASS${NC}"
        return 0
    else
        echo -e "${RED}✗ FAIL${NC}"
        ((FAILED_TESTS++))
        return 1
    fi
}

# Switch to correct namespace
kubectl config set-context --current --namespace="$NAMESPACE" &>/dev/null

# Test 1: Deployments exist and are ready
echo -e "\n${YELLOW}=== Testing Deployments ===${NC}"
test_command "nginx-web deployment exists" \
    "kubectl get deployment nginx-web"

test_command "nginx-web deployment is ready" \
    "kubectl wait --for=condition=available --timeout=30s deployment/nginx-web"

test_command "nginx-custom deployment exists" \
    "kubectl get deployment nginx-custom"

test_command "nginx-custom deployment is ready" \
    "kubectl wait --for=condition=available --timeout=30s deployment/nginx-custom"

# Test 2: Pods are running
echo -e "\n${YELLOW}=== Testing Pods ===${NC}"
NGINX_PODS=$(kubectl get pods -l app=nginx --field-selector=status.phase=Running --no-headers 2>/dev/null | wc -l)
echo -e "${YELLOW}nginx-web pods running: $NGINX_PODS${NC}"
if [ "$NGINX_PODS" -ge 1 ]; then
    echo -e "${GREEN}✓ PASS: At least 1 nginx pod running${NC}"
else
    echo -e "${RED}✗ FAIL: No nginx pods running${NC}"
    ((FAILED_TESTS++))
fi

CUSTOM_PODS=$(kubectl get pods -l app=nginx-custom --field-selector=status.phase=Running --no-headers 2>/dev/null | wc -l)
echo -e "${YELLOW}nginx-custom pods running: $CUSTOM_PODS${NC}"
if [ "$CUSTOM_PODS" -ge 1 ]; then
    echo -e "${GREEN}✓ PASS: At least 1 nginx-custom pod running${NC}"
else
    echo -e "${RED}✗ FAIL: No nginx-custom pods running${NC}"
    ((FAILED_TESTS++))
fi

# Test 3: Services exist
echo -e "\n${YELLOW}=== Testing Services ===${NC}"
test_command "nginx-service exists" \
    "kubectl get service nginx-service"

test_command "nginx-custom-service exists" \
    "kubectl get service nginx-custom-service"

# Test 4: Endpoints are populated
echo -e "\n${YELLOW}=== Testing Endpoints ===${NC}"
NGINX_ENDPOINTS=$(kubectl get endpoints nginx-service -o jsonpath='{.subsets[0].addresses[*].ip}' 2>/dev/null | wc -w)
echo -e "${YELLOW}nginx-service endpoints: $NGINX_ENDPOINTS${NC}"
if [ "$NGINX_ENDPOINTS" -ge 1 ]; then
    echo -e "${GREEN}✓ PASS: Service has endpoints${NC}"
else
    echo -e "${RED}✗ FAIL: Service has no endpoints${NC}"
    ((FAILED_TESTS++))
fi

CUSTOM_ENDPOINTS=$(kubectl get endpoints nginx-custom-service -o jsonpath='{.subsets[0].addresses[*].ip}' 2>/dev/null | wc -w)
echo -e "${YELLOW}nginx-custom-service endpoints: $CUSTOM_ENDPOINTS${NC}"
if [ "$CUSTOM_ENDPOINTS" -ge 1 ]; then
    echo -e "${GREEN}✓ PASS: Custom service has endpoints${NC}"
else
    echo -e "${RED}✗ FAIL: Custom service has no endpoints${NC}"
    ((FAILED_TESTS++))
fi

# Test 5: ConfigMap exists
echo -e "\n${YELLOW}=== Testing ConfigMap ===${NC}"
test_command "nginx-html ConfigMap exists" \
    "kubectl get configmap nginx-html"

# Test 6: HTTP connectivity test (NodePort)
echo -e "\n${YELLOW}=== Testing HTTP Connectivity ===${NC}"
echo -e "${YELLOW}Testing NodePort service (nginx-custom)...${NC}"
if curl -s -o /dev/null -w "%{http_code}" http://localhost:30081 2>/dev/null | grep -q "200"; then
    echo -e "${GREEN}✓ PASS: NodePort service accessible${NC}"
else
    echo -e "${YELLOW}⚠ WARNING: NodePort test failed${NC}"
    echo -e "${YELLOW}This may fail if running on a remote cluster${NC}"
fi

# Test 7: Pod health checks
echo -e "\n${YELLOW}=== Testing Health Probes ===${NC}"
POD_NAME=$(kubectl get pods -l app=nginx -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
if [ -n "$POD_NAME" ]; then
    READY=$(kubectl get pod "$POD_NAME" -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}' 2>/dev/null)
    if [ "$READY" = "True" ]; then
        echo -e "${GREEN}✓ PASS: Pod $POD_NAME is ready${NC}"
    else
        echo -e "${RED}✗ FAIL: Pod $POD_NAME not ready${NC}"
        ((FAILED_TESTS++))
    fi
fi

# Test 8: Resource limits are set
echo -e "\n${YELLOW}=== Testing Resource Limits ===${NC}"
if [ -n "$POD_NAME" ]; then
    CPU_LIMIT=$(kubectl get pod "$POD_NAME" -o jsonpath='{.spec.containers[0].resources.limits.cpu}' 2>/dev/null)
    MEM_LIMIT=$(kubectl get pod "$POD_NAME" -o jsonpath='{.spec.containers[0].resources.limits.memory}' 2>/dev/null)

    if [ -n "$CPU_LIMIT" ] && [ -n "$MEM_LIMIT" ]; then
        echo -e "${GREEN}✓ PASS: Resource limits configured (CPU: $CPU_LIMIT, Memory: $MEM_LIMIT)${NC}"
    else
        echo -e "${RED}✗ FAIL: Resource limits not configured${NC}"
        ((FAILED_TESTS++))
    fi
fi

# Summary
echo -e "\n${GREEN}========================================${NC}"
if [ $FAILED_TESTS -eq 0 ]; then
    echo -e "${GREEN}All Tests Passed! ✓${NC}"
    echo -e "${GREEN}========================================${NC}"
    exit 0
else
    echo -e "${RED}$FAILED_TESTS Test(s) Failed ✗${NC}"
    echo -e "${RED}========================================${NC}"
    echo -e "\n${YELLOW}Troubleshooting:${NC}"
    echo "  View pod status: kubectl get pods"
    echo "  View pod logs: kubectl logs <pod-name>"
    echo "  Describe pod: kubectl describe pod <pod-name>"
    echo "  View events: kubectl get events --sort-by='.lastTimestamp'"
    exit 1
fi
