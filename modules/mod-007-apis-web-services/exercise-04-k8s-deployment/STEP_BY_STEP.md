# Step-by-Step Guide: Kubernetes Deployment from CI/CD

## Overview
Automate Kubernetes deployments from GitHub Actions using kubectl, deploy ML applications with proper manifests, and implement rolling updates with health checks.

## Phase 1: Kubernetes Manifest Setup (15 minutes)

### Create Deployment Manifest
Create `k8s/deployment.yaml`:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-app
  labels:
    app: ml-app
    version: v1
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ml-app
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  template:
    metadata:
      labels:
        app: ml-app
        version: v1
    spec:
      containers:
      - name: ml-app
        image: ghcr.io/username/ml-app:latest
        imagePullPolicy: Always
        ports:
        - containerPort: 5000
          name: http
        env:
        - name: ENVIRONMENT
          value: "production"
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 5000
          initialDelaySeconds: 10
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /health
            port: 5000
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 2
```

### Create Service Manifest
Create `k8s/service.yaml`:
```yaml
apiVersion: v1
kind: Service
metadata:
  name: ml-app-service
  labels:
    app: ml-app
spec:
  type: LoadBalancer
  selector:
    app: ml-app
  ports:
  - port: 80
    targetPort: 5000
    protocol: TCP
    name: http
  sessionAffinity: ClientIP
```

### Create ConfigMap
Create `k8s/configmap.yaml`:
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: ml-app-config
data:
  LOG_LEVEL: "INFO"
  MODEL_PATH: "/models/model.pkl"
  PREDICTION_TIMEOUT: "30"
```

**Validation**: Validate manifests: `kubectl apply --dry-run=client -f k8s/`

## Phase 2: Local Kubernetes Testing (15 minutes)

### Setup Minikube/Kind
```bash
# Install minikube (if not installed)
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube

# Start local cluster
minikube start --driver=docker --cpus=2 --memory=4096

# Verify cluster
kubectl cluster-info
kubectl get nodes
```

### Deploy to Local Cluster
```bash
# Apply manifests
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# Verify deployment
kubectl get pods
kubectl get deployments
kubectl get services

# Check rollout status
kubectl rollout status deployment/ml-app
```

### Test Application
```bash
# Port forward for testing
kubectl port-forward service/ml-app-service 8080:80

# In another terminal, test endpoint
curl http://localhost:8080/health

# Check logs
kubectl logs -l app=ml-app --tail=50

# Describe pod for debugging
kubectl describe pod -l app=ml-app
```

**Validation**: Application responds to health checks and serves requests.

## Phase 3: GitHub Actions Kubernetes Deploy (15 minutes)

### Create Kubernetes Deploy Workflow
Create `.github/workflows/k8s-deploy.yml`:
```yaml
name: Deploy to Kubernetes

on:
  push:
    branches: [ main ]
  workflow_dispatch:

env:
  IMAGE_NAME: ghcr.io/${{ github.repository }}

jobs:
  deploy:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write

    steps:
    - name: Checkout code
      uses: actions/checkout@v3

    - name: Set up kubectl
      uses: azure/setup-kubectl@v3
      with:
        version: 'v1.28.0'

    - name: Configure kubectl
      run: |
        mkdir -p $HOME/.kube
        echo "${{ secrets.KUBE_CONFIG }}" | base64 -d > $HOME/.kube/config
        chmod 600 $HOME/.kube/config

    - name: Verify cluster access
      run: |
        kubectl cluster-info
        kubectl get nodes

    - name: Update image tag
      run: |
        export IMAGE_TAG=${{ github.sha }}
        sed -i "s|image: .*|image: ${{ env.IMAGE_NAME }}:${IMAGE_TAG}|g" k8s/deployment.yaml

    - name: Deploy to Kubernetes
      run: |
        kubectl apply -f k8s/configmap.yaml
        kubectl apply -f k8s/service.yaml
        kubectl apply -f k8s/deployment.yaml

    - name: Wait for rollout
      run: |
        kubectl rollout status deployment/ml-app --timeout=5m

    - name: Verify deployment
      run: |
        kubectl get pods -l app=ml-app
        kubectl get services ml-app-service

    - name: Run smoke tests
      run: |
        SERVICE_IP=$(kubectl get service ml-app-service -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
        curl -f http://${SERVICE_IP}/health || exit 1
```

**Validation**: Push to main branch triggers deployment workflow.

## Phase 4: Advanced Deployment Patterns (15 minutes)

### Create Kustomization
Create `k8s/base/kustomization.yaml`:
```yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

resources:
  - deployment.yaml
  - service.yaml
  - configmap.yaml

commonLabels:
  app.kubernetes.io/name: ml-app
  app.kubernetes.io/managed-by: kustomize

images:
  - name: ml-app
    newName: ghcr.io/username/ml-app
    newTag: latest
```

### Create Environment Overlays
Create `k8s/overlays/production/kustomization.yaml`:
```yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

bases:
  - ../../base

replicas:
  - name: ml-app
    count: 5

patches:
  - patch: |-
      - op: replace
        path: /spec/template/spec/containers/0/resources/limits/memory
        value: "1Gi"
    target:
      kind: Deployment
      name: ml-app
```

### Deploy with Kustomize
```bash
# Build and view
kubectl kustomize k8s/overlays/production

# Apply
kubectl apply -k k8s/overlays/production

# Verify
kubectl get deployment ml-app -o yaml
```

**Validation**: Production configuration overrides base settings correctly.

## Phase 5: Helm Chart Creation (15 minutes)

### Create Helm Chart Structure
```bash
# Create chart
mkdir -p helm/ml-app
cd helm/ml-app

# Create Chart.yaml
cat > Chart.yaml << 'EOF'
apiVersion: v2
name: ml-app
description: ML Application Helm Chart
type: application
version: 1.0.0
appVersion: "1.0"
EOF
```

### Create Values File
Create `helm/ml-app/values.yaml`:
```yaml
replicaCount: 3

image:
  repository: ghcr.io/username/ml-app
  pullPolicy: IfNotPresent
  tag: "latest"

service:
  type: LoadBalancer
  port: 80
  targetPort: 5000

resources:
  limits:
    cpu: 500m
    memory: 512Mi
  requests:
    cpu: 250m
    memory: 256Mi

autoscaling:
  enabled: false
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 80

env:
  LOG_LEVEL: INFO
  ENVIRONMENT: production
```

### Create Template
Create `helm/ml-app/templates/deployment.yaml`:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ include "ml-app.fullname" . }}
  labels:
    {{- include "ml-app.labels" . | nindent 4 }}
spec:
  replicas: {{ .Values.replicaCount }}
  selector:
    matchLabels:
      {{- include "ml-app.selectorLabels" . | nindent 6 }}
  template:
    metadata:
      labels:
        {{- include "ml-app.selectorLabels" . | nindent 8 }}
    spec:
      containers:
      - name: {{ .Chart.Name }}
        image: "{{ .Values.image.repository }}:{{ .Values.image.tag | default .Chart.AppVersion }}"
        imagePullPolicy: {{ .Values.image.pullPolicy }}
        ports:
        - containerPort: {{ .Values.service.targetPort }}
        resources:
          {{- toYaml .Values.resources | nindent 12 }}
```

### Deploy with Helm
```bash
# Install chart
helm install ml-app ./helm/ml-app

# Upgrade
helm upgrade ml-app ./helm/ml-app --set replicaCount=5

# Rollback
helm rollback ml-app 1
```

**Validation**: `helm list` shows successful deployment.

## Phase 6: CI/CD with ArgoCD (10 minutes)

### Create ArgoCD Application
Create `argocd/application.yaml`:
```yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: ml-app
  namespace: argocd
spec:
  project: default
  source:
    repoURL: https://github.com/username/ml-app.git
    targetRevision: main
    path: k8s/overlays/production
  destination:
    server: https://kubernetes.default.svc
    namespace: ml-app
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
    syncOptions:
      - CreateNamespace=true
```

### Apply ArgoCD Application
```bash
# Install ArgoCD
kubectl create namespace argocd
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml

# Apply application
kubectl apply -f argocd/application.yaml

# Get admin password
kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath="{.data.password}" | base64 -d

# Port forward UI
kubectl port-forward svc/argocd-server -n argocd 8080:443
```

**Validation**: Access ArgoCD UI at https://localhost:8080 and verify sync status.

## Summary

You've implemented comprehensive Kubernetes deployment automation with:
- **Production-ready manifests** with health checks, resource limits, and rolling updates
- **GitHub Actions integration** for automated kubectl deployments
- **Kustomize overlays** for environment-specific configurations
- **Helm charts** for templated, reusable deployments
- **ArgoCD GitOps** for declarative, automated synchronization
- **Deployment verification** with rollout status and smoke tests

This infrastructure enables zero-downtime deployments with automated rollback capabilities and full deployment history tracking.
