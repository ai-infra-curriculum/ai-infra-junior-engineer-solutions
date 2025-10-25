#!/bin/bash
# Setup script for Exercise 01: First Kubernetes Deployment
# This script sets up the namespace and verifies the environment

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

NAMESPACE="exercise-01"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Exercise 01: Environment Setup${NC}"
echo -e "${GREEN}========================================${NC}"

# Function to check command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check kubectl
echo -e "\n${YELLOW}Checking prerequisites...${NC}"
if ! command_exists kubectl; then
    echo -e "${RED}❌ kubectl not found. Please install kubectl.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ kubectl found${NC}"

# Check cluster connection
echo -e "\n${YELLOW}Checking cluster connection...${NC}"
if ! kubectl cluster-info &>/dev/null; then
    echo -e "${RED}❌ Cannot connect to Kubernetes cluster${NC}"
    echo -e "${YELLOW}Please ensure:${NC}"
    echo "  - Docker Desktop Kubernetes is enabled, OR"
    echo "  - Minikube is running, OR"
    echo "  - You have valid kubeconfig for a cluster"
    exit 1
fi
echo -e "${GREEN}✓ Connected to cluster${NC}"

# Display cluster info
kubectl cluster-info

# Check nodes
echo -e "\n${YELLOW}Checking cluster nodes...${NC}"
NODE_COUNT=$(kubectl get nodes --no-headers 2>/dev/null | wc -l)
if [ "$NODE_COUNT" -eq 0 ]; then
    echo -e "${RED}❌ No nodes found in cluster${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Found $NODE_COUNT node(s)${NC}"
kubectl get nodes

# Create namespace
echo -e "\n${YELLOW}Creating namespace: $NAMESPACE${NC}"
if kubectl get namespace "$NAMESPACE" &>/dev/null; then
    echo -e "${YELLOW}⚠ Namespace '$NAMESPACE' already exists${NC}"
    read -p "Do you want to delete and recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Deleting existing namespace..."
        kubectl delete namespace "$NAMESPACE" --grace-period=30 --timeout=60s
        echo "Waiting for namespace to be fully deleted..."
        sleep 5
        kubectl create namespace "$NAMESPACE"
        echo -e "${GREEN}✓ Namespace recreated${NC}"
    else
        echo "Using existing namespace"
    fi
else
    kubectl create namespace "$NAMESPACE"
    echo -e "${GREEN}✓ Namespace created${NC}"
fi

# Set namespace as default for current context
echo -e "\n${YELLOW}Setting default namespace...${NC}"
kubectl config set-context --current --namespace="$NAMESPACE"
echo -e "${GREEN}✓ Default namespace set to: $NAMESPACE${NC}"

# Verify namespace
CURRENT_NS=$(kubectl config view --minify --output 'jsonpath={..namespace}')
echo -e "${GREEN}Current namespace: $CURRENT_NS${NC}"

# Check if metrics-server is available (for kubectl top)
echo -e "\n${YELLOW}Checking metrics server...${NC}"
if kubectl top nodes &>/dev/null; then
    echo -e "${GREEN}✓ Metrics server available${NC}"
    kubectl top nodes
else
    echo -e "${YELLOW}⚠ Metrics server not available${NC}"
    echo -e "${YELLOW}Some commands like 'kubectl top' will not work${NC}"
    echo -e "${YELLOW}This is optional for the exercise${NC}"
fi

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Setup Complete!${NC}"
echo -e "${GREEN}========================================${NC}"

echo -e "\n${YELLOW}Next steps:${NC}"
echo "1. Deploy nginx: kubectl apply -f manifests/01-nginx-deployment.yaml"
echo "2. Expose with service: kubectl apply -f manifests/02-nginx-service-clusterip.yaml"
echo "3. Follow the README.md for the complete exercise"

echo -e "\n${YELLOW}Quick commands:${NC}"
echo "  View all resources: kubectl get all"
echo "  View pods: kubectl get pods"
echo "  View logs: kubectl logs <pod-name>"
echo "  Delete all: kubectl delete all --all"
