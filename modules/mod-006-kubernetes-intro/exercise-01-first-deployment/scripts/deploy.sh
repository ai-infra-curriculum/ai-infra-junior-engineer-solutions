#!/bin/bash
# Deployment script for Exercise 01
# Deploys all resources in the correct order

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

NAMESPACE="exercise-01"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Deploying Exercise 01 Resources${NC}"
echo -e "${GREEN}========================================${NC}"

# Check if we're in the right namespace
CURRENT_NS=$(kubectl config view --minify --output 'jsonpath={..namespace}')
if [ "$CURRENT_NS" != "$NAMESPACE" ]; then
    echo -e "${YELLOW}⚠ Current namespace is '$CURRENT_NS', switching to '$NAMESPACE'${NC}"
    kubectl config set-context --current --namespace="$NAMESPACE"
fi

# Check if manifests directory exists
if [ ! -d "manifests" ]; then
    echo -e "${RED}❌ manifests directory not found${NC}"
    echo "Please run this script from the exercise-01-first-deployment directory"
    exit 1
fi

# Deploy resources in order
echo -e "\n${YELLOW}Step 1: Deploying nginx Deployment...${NC}"
kubectl apply -f manifests/01-nginx-deployment.yaml
echo -e "${GREEN}✓ Deployment created${NC}"

echo -e "\n${YELLOW}Waiting for deployment to be ready...${NC}"
kubectl wait --for=condition=available --timeout=60s deployment/nginx-web
echo -e "${GREEN}✓ Deployment ready${NC}"

echo -e "\n${YELLOW}Step 2: Creating ClusterIP Service...${NC}"
kubectl apply -f manifests/02-nginx-service-clusterip.yaml
echo -e "${GREEN}✓ Service created${NC}"

echo -e "\n${YELLOW}Step 3: Creating ConfigMap for custom HTML...${NC}"
kubectl apply -f manifests/04-nginx-custom-html-configmap.yaml
echo -e "${GREEN}✓ ConfigMap created${NC}"

echo -e "\n${YELLOW}Step 4: Deploying custom nginx with ConfigMap...${NC}"
kubectl apply -f manifests/05-nginx-with-configmap.yaml
echo -e "${GREEN}✓ Custom nginx deployed${NC}"

echo -e "\n${YELLOW}Waiting for custom nginx deployment...${NC}"
kubectl wait --for=condition=available --timeout=60s deployment/nginx-custom
echo -e "${GREEN}✓ Custom nginx ready${NC}"

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Deployment Complete!${NC}"
echo -e "${GREEN}========================================${NC}"

# Display status
echo -e "\n${YELLOW}Current Status:${NC}"
kubectl get deployments
echo ""
kubectl get services
echo ""
kubectl get pods

echo -e "\n${YELLOW}Access the applications:${NC}"
echo "1. nginx-web (ClusterIP):"
echo "   kubectl port-forward deployment/nginx-web 8080:80"
echo "   Then visit: http://localhost:8080"
echo ""
echo "2. nginx-custom (NodePort):"
echo "   curl http://localhost:30081"
echo "   Or visit in browser: http://localhost:30081"

echo -e "\n${YELLOW}View logs:${NC}"
echo "kubectl logs -l app=nginx"

echo -e "\n${YELLOW}Scale deployment:${NC}"
echo "kubectl scale deployment nginx-web --replicas=5"

echo -e "\n${YELLOW}Update image (rolling update):${NC}"
echo "kubectl set image deployment/nginx-web nginx=nginx:1.22"

echo -e "\n${YELLOW}Cleanup:${NC}"
echo "./scripts/cleanup.sh"
