#!/bin/bash
# Cleanup script for Exercise 01
# Removes all resources created during the exercise

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

NAMESPACE="exercise-01"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Cleanup Exercise 01 Resources${NC}"
echo -e "${GREEN}========================================${NC}"

# Check if namespace exists
if ! kubectl get namespace "$NAMESPACE" &>/dev/null; then
    echo -e "${YELLOW}Namespace '$NAMESPACE' doesn't exist. Nothing to clean up.${NC}"
    exit 0
fi

echo -e "\n${YELLOW}Current resources in namespace '$NAMESPACE':${NC}"
kubectl get all -n "$NAMESPACE"

echo -e "\n${RED}This will delete ALL resources in namespace '$NAMESPACE'${NC}"
read -p "Are you sure you want to continue? (y/N): " -n 1 -r
echo

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cleanup cancelled."
    exit 0
fi

# Delete specific resources first (graceful deletion)
echo -e "\n${YELLOW}Deleting deployments...${NC}"
kubectl delete deployment --all -n "$NAMESPACE" --grace-period=30 2>/dev/null || true
echo -e "${GREEN}✓ Deployments deleted${NC}"

echo -e "\n${YELLOW}Deleting services...${NC}"
kubectl delete service --all -n "$NAMESPACE" --grace-period=30 2>/dev/null || true
echo -e "${GREEN}✓ Services deleted${NC}"

echo -e "\n${YELLOW}Deleting configmaps...${NC}"
kubectl delete configmap --all -n "$NAMESPACE" 2>/dev/null || true
echo -e "${GREEN}✓ ConfigMaps deleted${NC}"

echo -e "\n${YELLOW}Deleting pods...${NC}"
kubectl delete pod --all -n "$NAMESPACE" --grace-period=30 2>/dev/null || true
echo -e "${GREEN}✓ Pods deleted${NC}"

# Wait for pods to terminate
echo -e "\n${YELLOW}Waiting for pods to terminate...${NC}"
kubectl wait --for=delete pod --all -n "$NAMESPACE" --timeout=60s 2>/dev/null || true

# Delete any remaining resources
echo -e "\n${YELLOW}Deleting any remaining resources...${NC}"
kubectl delete all --all -n "$NAMESPACE" --grace-period=0 --force 2>/dev/null || true

# Option to delete namespace
echo -e "\n${YELLOW}Do you want to delete the namespace '$NAMESPACE' as well?${NC}"
read -p "(y/N): " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Deleting namespace...${NC}"
    kubectl delete namespace "$NAMESPACE" --grace-period=30 --timeout=60s
    echo -e "${GREEN}✓ Namespace deleted${NC}"

    # Switch back to default namespace
    kubectl config set-context --current --namespace=default
    echo -e "${GREEN}✓ Switched to 'default' namespace${NC}"
else
    echo -e "${YELLOW}Namespace '$NAMESPACE' kept (resources deleted)${NC}"
fi

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Cleanup Complete!${NC}"
echo -e "${GREEN}========================================${NC}"

# Verify cleanup
echo -e "\n${YELLOW}Remaining resources:${NC}"
if kubectl get namespace "$NAMESPACE" &>/dev/null; then
    kubectl get all -n "$NAMESPACE"
else
    echo "Namespace '$NAMESPACE' has been deleted"
fi

echo -e "\n${YELLOW}To start fresh:${NC}"
echo "./scripts/setup.sh"
echo "./scripts/deploy.sh"
