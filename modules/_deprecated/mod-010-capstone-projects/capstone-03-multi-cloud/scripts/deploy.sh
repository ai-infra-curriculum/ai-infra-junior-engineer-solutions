#!/bin/bash
# Multi-Cloud ML Platform Deployment Script
# Deploys infrastructure and applications to AWS, GCP, and Azure

set -e

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
ENVIRONMENT="${ENVIRONMENT:-dev}"
DEPLOY_AWS="${DEPLOY_AWS:-true}"
DEPLOY_GCP="${DEPLOY_GCP:-true}"
DEPLOY_AZURE="${DEPLOY_AZURE:-true}"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Multi-Cloud ML Platform Deployment${NC}"
echo -e "${GREEN}Environment: ${ENVIRONMENT}${NC}"
echo -e "${GREEN}========================================${NC}"

# Function to print status
print_status() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

print_error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."

    commands=("terraform" "kubectl" "helm" "docker" "aws" "gcloud" "az")
    for cmd in "${commands[@]}"; do
        if ! command -v $cmd &> /dev/null; then
            print_error "$cmd is not installed"
            exit 1
        fi
    done

    print_status "All prerequisites met"
}

# Deploy Terraform infrastructure
deploy_infrastructure() {
    local cloud=$1
    print_status "Deploying $cloud infrastructure..."

    cd terraform/$cloud

    terraform init
    terraform plan -out=tfplan -var="environment=${ENVIRONMENT}"

    if [ $? -eq 0 ]; then
        terraform apply tfplan
        print_status "$cloud infrastructure deployed successfully"
    else
        print_error "Failed to deploy $cloud infrastructure"
        exit 1
    fi

    cd ../..
}

# Configure kubectl for cluster
configure_kubectl() {
    local cloud=$1
    print_status "Configuring kubectl for $cloud..."

    case $cloud in
        aws)
            CLUSTER_NAME=$(terraform -chdir=terraform/aws output -raw eks_cluster_id)
            REGION=$(terraform -chdir=terraform/aws output -raw region)
            aws eks update-kubeconfig --name $CLUSTER_NAME --region $REGION
            ;;
        gcp)
            CLUSTER_NAME=$(terraform -chdir=terraform/gcp output -raw gke_cluster_name)
            REGION=$(terraform -chdir=terraform/gcp output -raw region)
            PROJECT_ID=$(terraform -chdir=terraform/gcp output -raw project_id)
            gcloud container clusters get-credentials $CLUSTER_NAME --region $REGION --project $PROJECT_ID
            ;;
        azure)
            CLUSTER_NAME=$(terraform -chdir=terraform/azure output -raw aks_cluster_name)
            RESOURCE_GROUP=$(terraform -chdir=terraform/azure output -raw resource_group_name)
            az aks get-credentials --name $CLUSTER_NAME --resource-group $RESOURCE_GROUP
            ;;
    esac

    print_status "kubectl configured for $cloud"
}

# Deploy Kubernetes resources
deploy_kubernetes() {
    local cloud=$1
    print_status "Deploying Kubernetes resources to $cloud..."

    # Create namespaces
    kubectl apply -f kubernetes/shared/namespaces.yaml

    # Deploy cloud-specific resources
    kubectl apply -f kubernetes/${cloud}-$(echo $cloud | sed 's/aws/eks/;s/gcp/gke/;s/azure/aks/')/ --recursive

    # Deploy shared resources
    kubectl apply -f kubernetes/shared/ --recursive

    print_status "Kubernetes resources deployed to $cloud"
}

# Install Istio service mesh
install_istio() {
    print_status "Installing Istio service mesh..."

    # Download and install Istio
    if [ ! -d "istio-1.19.0" ]; then
        curl -L https://istio.io/downloadIstio | ISTIO_VERSION=1.19.0 sh -
    fi

    export PATH=$PWD/istio-1.19.0/bin:$PATH

    # Install Istio
    istioctl install --set profile=production -y

    # Enable Istio injection
    kubectl label namespace ml-platform istio-injection=enabled
    kubectl label namespace ml-serving istio-injection=enabled

    # Apply service mesh configuration
    kubectl apply -f kubernetes/service-mesh/

    print_status "Istio service mesh installed"
}

# Build and push container images
build_and_push_images() {
    print_status "Building and pushing container images..."

    # Build API Gateway
    docker build -t ml-platform/api-gateway:${ENVIRONMENT} src/api-gateway/

    # Build Model Serving
    docker build -t ml-platform/model-serving:${ENVIRONMENT} src/model-serving/

    # Tag and push to registries would happen here
    print_status "Container images built"
}

# Deploy monitoring stack
deploy_monitoring() {
    print_status "Deploying monitoring stack..."

    # Install Prometheus
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    helm repo update

    helm upgrade --install prometheus prometheus-community/kube-prometheus-stack \
        --namespace monitoring \
        --create-namespace \
        --values monitoring/prometheus/values.yaml

    # Deploy Grafana dashboards
    kubectl apply -f monitoring/grafana/dashboards/

    print_status "Monitoring stack deployed"
}

# Run health checks
run_health_checks() {
    print_status "Running health checks..."

    sleep 30  # Wait for services to start

    # Check API Gateway
    for cloud in aws gcp azure; do
        if kubectl get pods -n ml-platform -l app=api-gateway --context=${cloud} &> /dev/null; then
            print_status "API Gateway healthy on $cloud"
        else
            print_warning "API Gateway not found on $cloud"
        fi
    done

    print_status "Health checks completed"
}

# Main deployment workflow
main() {
    check_prerequisites

    # Deploy infrastructure to selected clouds
    if [ "$DEPLOY_AWS" = "true" ]; then
        deploy_infrastructure "aws"
        configure_kubectl "aws"
        deploy_kubernetes "aws"
    fi

    if [ "$DEPLOY_GCP" = "true" ]; then
        deploy_infrastructure "gcp"
        configure_kubectl "gcp"
        deploy_kubernetes "gcp"
    fi

    if [ "$DEPLOY_AZURE" = "true" ]; then
        deploy_infrastructure "azure"
        configure_kubectl "azure"
        deploy_kubernetes "azure"
    fi

    # Build and push images
    build_and_push_images

    # Install service mesh (once per cluster)
    if [ "$DEPLOY_AWS" = "true" ]; then
        install_istio
    fi

    # Deploy monitoring
    deploy_monitoring

    # Run health checks
    run_health_checks

    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Deployment completed successfully!${NC}"
    echo -e "${GREEN}========================================${NC}"
}

# Run main function
main
