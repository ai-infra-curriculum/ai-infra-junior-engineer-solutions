# Exercise 04: Kubernetes Deployment Automation

## Overview

Automate deployment of Docker images to Kubernetes using CI/CD pipelines. This exercise builds on Exercise 03 (Docker CI/CD) and Module 006 (Kubernetes Introduction) to create production-ready automated deployment workflows.

## Learning Objectives

- Deploy Docker images to Kubernetes
- Create Kubernetes deployment manifests
- Build Helm charts for applications
- Automate deployments with GitHub Actions
- Implement rolling updates and rollbacks
- Set up GitOps workflows
- Configure secrets management
- Monitor deployments

## Prerequisites

- Kubernetes cluster access (minikube, kind, or cloud provider)
- kubectl configured
- Helm 3 installed
- Completed Exercise 03 (Docker CI/CD)
- Familiarity with Module 006 (Kubernetes Introduction)

## Project Structure

```
exercise-04-k8s-deployment/
├── manifests/
│   ├── namespace.yaml              # Namespace definition
│   ├── deployment.yaml             # ML API deployment
│   ├── service.yaml                # Service for ML API
│   ├── ingress.yaml                # Ingress configuration
│   ├── configmap.yaml              # Configuration
│   ├── secret.yaml                 # Secrets (template)
│   ├── hpa.yaml                    # Horizontal Pod Autoscaler
│   └── pdb.yaml                    # Pod Disruption Budget
├── helm-chart/
│   └── ml-api/
│       ├── Chart.yaml              # Helm chart metadata
│       ├── values.yaml             # Default values
│       ├── values-dev.yaml         # Dev environment
│       ├── values-prod.yaml        # Production environment
│       └── templates/
│           ├── deployment.yaml
│           ├── service.yaml
│           ├── ingress.yaml
│           ├── configmap.yaml
│           ├── secret.yaml
│           ├── hpa.yaml
│           └── _helpers.tpl
├── scripts/
│   ├── deploy.sh                   # Manual deployment script
│   ├── rollback.sh                 # Rollback script
│   └── test-deployment.sh          # Test deployed application
├── .github/
│   └── workflows/
│       ├── deploy-dev.yml          # Deploy to dev on PR merge
│       ├── deploy-prod.yml         # Deploy to prod on release
│       └── k8s-tests.yml           # Test K8s manifests
└── README.md
```

## Quick Start

### 1. Deploy with kubectl

```bash
# Create namespace
kubectl apply -f manifests/namespace.yaml

# Deploy application
kubectl apply -f manifests/

# Check status
kubectl get pods -n ml-api
kubectl get svc -n ml-api
```

### 2. Deploy with Helm

```bash
# Install chart
helm install ml-api ./helm-chart/ml-api -n ml-api --create-namespace

# Upgrade chart
helm upgrade ml-api ./helm-chart/ml-api -n ml-api

# Rollback
helm rollback ml-api -n ml-api
```

### 3. Deploy with Script

```bash
# Deploy to dev
./scripts/deploy.sh --environment dev

# Deploy to prod
./scripts/deploy.sh --environment prod --image-tag v1.2.0
```

## Kubernetes Manifests

### Deployment

Deploys the ML API with:
- 3 replicas for high availability
- Resource limits
- Liveness and readiness probes
- Rolling update strategy

### Service

Exposes the deployment:
- ClusterIP service
- Port 80 → 8000
- Selector matches deployment pods

### Ingress

Routes external traffic:
- Host-based routing
- TLS termination
- Path-based routing for API

### HorizontalPodAutoscaler

Automatically scales based on:
- CPU utilization (70%)
- Memory utilization (80%)
- Custom metrics (requests/sec)

## Helm Chart

### Chart Structure

```yaml
# Chart.yaml
apiVersion: v2
name: ml-api
description: ML API Helm Chart
version: 1.0.0
appVersion: "1.0.0"
```

### Values

```yaml
# values.yaml
replicaCount: 3

image:
  repository: ghcr.io/username/ml-api
  tag: "latest"
  pullPolicy: IfNotPresent

service:
  type: ClusterIP
  port: 80

ingress:
  enabled: true
  className: nginx
  hosts:
    - host: ml-api.example.com
      paths:
        - path: /
          pathType: Prefix

resources:
  requests:
    memory: "512Mi"
    cpu: "250m"
  limits:
    memory: "1Gi"
    cpu: "500m"

autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
```

### Environment-Specific Values

```bash
# Development
helm install ml-api ./helm-chart/ml-api -f values-dev.yaml

# Production
helm install ml-api ./helm-chart/ml-api -f values-prod.yaml
```

## CI/CD Workflows

### Deploy to Dev (on PR merge)

```yaml
name: Deploy to Dev

on:
  push:
    branches: [develop]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Configure kubectl
        uses: azure/k8s-set-context@v3
        with:
          method: kubeconfig
          kubeconfig: ${{ secrets.KUBECONFIG_DEV }}

      - name: Deploy with Helm
        run: |
          helm upgrade --install ml-api ./helm-chart/ml-api \
            --namespace ml-api-dev \
            --create-namespace \
            --values helm-chart/ml-api/values-dev.yaml \
            --set image.tag=${{ github.sha }} \
            --wait

      - name: Test deployment
        run: ./scripts/test-deployment.sh
```

### Deploy to Production (on release)

```yaml
name: Deploy to Production

on:
  release:
    types: [published]

jobs:
  deploy:
    runs-on: ubuntu-latest
    environment: production
    steps:
      - uses: actions/checkout@v4

      - name: Configure kubectl
        uses: azure/k8s-set-context@v3
        with:
          method: kubeconfig
          kubeconfig: ${{ secrets.KUBECONFIG_PROD }}

      - name: Deploy with Helm
        run: |
          helm upgrade --install ml-api ./helm-chart/ml-api \
            --namespace ml-api \
            --values helm-chart/ml-api/values-prod.yaml \
            --set image.tag=${{ github.event.release.tag_name }} \
            --wait \
            --timeout 10m

      - name: Smoke tests
        run: ./scripts/test-deployment.sh --environment prod
```

## Deployment Strategies

### Rolling Update (Default)

```yaml
strategy:
  type: RollingUpdate
  rollingUpdate:
    maxUnavailable: 1
    maxSurge: 1
```

### Blue-Green Deployment

```bash
# Deploy green version
helm install ml-api-green ./helm-chart/ml-api \
  --set image.tag=v2.0.0 \
  --set service.selector.version=green

# Switch traffic
kubectl patch service ml-api -p '{"spec":{"selector":{"version":"green"}}}'

# Remove blue version
helm uninstall ml-api-blue
```

### Canary Deployment

```yaml
# Deploy canary with fewer replicas
replicaCount: 1
labels:
  version: canary

# Main deployment
replicaCount: 9
labels:
  version: stable
```

## Secrets Management

### Using Kubernetes Secrets

```bash
# Create secret from file
kubectl create secret generic ml-api-secrets \
  --from-file=model-key=./model.key \
  --namespace ml-api

# Create from literals
kubectl create secret generic ml-api-secrets \
  --from-literal=api-key=your-api-key \
  --namespace ml-api
```

### Using External Secrets Operator

```yaml
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: ml-api-secrets
spec:
  secretStoreRef:
    name: aws-secrets-manager
  target:
    name: ml-api-secrets
  data:
    - secretKey: api-key
      remoteRef:
        key: prod/ml-api/api-key
```

### Using Sealed Secrets

```bash
# Install sealed-secrets controller
helm install sealed-secrets sealed-secrets/sealed-secrets -n kube-system

# Create sealed secret
kubeseal --format yaml < secret.yaml > sealed-secret.yaml

# Apply sealed secret (safe to commit)
kubectl apply -f sealed-secret.yaml
```

## Monitoring Deployments

### Check Deployment Status

```bash
# Watch deployment progress
kubectl rollout status deployment/ml-api -n ml-api

# View deployment history
kubectl rollout history deployment/ml-api -n ml-api

# Describe deployment
kubectl describe deployment ml-api -n ml-api
```

### View Logs

```bash
# Logs from all pods
kubectl logs -l app=ml-api -n ml-api

# Follow logs
kubectl logs -f deployment/ml-api -n ml-api

# Previous pod logs
kubectl logs -l app=ml-api --previous -n ml-api
```

### Health Checks

```bash
# Check pod health
kubectl get pods -n ml-api

# Port forward for local testing
kubectl port-forward svc/ml-api 8000:80 -n ml-api

# Test health endpoint
curl http://localhost:8000/health
```

## Rollback Procedures

### Helm Rollback

```bash
# List releases
helm list -n ml-api

# View history
helm history ml-api -n ml-api

# Rollback to previous
helm rollback ml-api -n ml-api

# Rollback to specific revision
helm rollback ml-api 3 -n ml-api
```

### kubectl Rollback

```bash
# Rollback to previous
kubectl rollout undo deployment/ml-api -n ml-api

# Rollback to specific revision
kubectl rollout undo deployment/ml-api --to-revision=2 -n ml-api
```

## GitOps with ArgoCD

### ArgoCD Application

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: ml-api
  namespace: argocd
spec:
  project: default
  source:
    repoURL: https://github.com/username/ml-api
    targetRevision: HEAD
    path: helm-chart/ml-api
    helm:
      valueFiles:
        - values-prod.yaml
  destination:
    server: https://kubernetes.default.svc
    namespace: ml-api
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
```

## Best Practices

### 1. Resource Management

✅ Always set resource requests and limits
✅ Use HPA for automatic scaling
✅ Set Pod Disruption Budgets for availability

### 2. Health Checks

✅ Implement liveness probes
✅ Implement readiness probes
✅ Set appropriate timeouts

### 3. Updates

✅ Use rolling updates
✅ Test in dev/staging first
✅ Have rollback plan ready
✅ Monitor during rollout

### 4. Security

✅ Use secrets for sensitive data
✅ Apply least privilege RBAC
✅ Use Network Policies
✅ Scan images for vulnerabilities
✅ Keep Kubernetes updated

### 5. Observability

✅ Centralized logging
✅ Metrics collection (Prometheus)
✅ Distributed tracing
✅ Alerting on failures

## Common Commands

```bash
# Deploy
kubectl apply -f manifests/
helm install ml-api ./helm-chart/ml-api

# Update
kubectl set image deployment/ml-api ml-api=ml-api:v2
helm upgrade ml-api ./helm-chart/ml-api

# Scale
kubectl scale deployment/ml-api --replicas=5

# Rollback
kubectl rollout undo deployment/ml-api
helm rollback ml-api

# Debug
kubectl logs -f deployment/ml-api
kubectl exec -it deployment/ml-api -- /bin/bash
kubectl describe pod <pod-name>

# Clean up
kubectl delete -f manifests/
helm uninstall ml-api
```

## Troubleshooting

### Pods not starting

```bash
# Check pod status
kubectl get pods -n ml-api

# Describe pod
kubectl describe pod <pod-name> -n ml-api

# Check events
kubectl get events -n ml-api --sort-by='.lastTimestamp'

# Check logs
kubectl logs <pod-name> -n ml-api
```

### Image pull errors

```bash
# Check image pull secrets
kubectl get secrets -n ml-api

# Create image pull secret
kubectl create secret docker-registry regcred \
  --docker-server=ghcr.io \
  --docker-username=username \
  --docker-password=$GITHUB_TOKEN \
  -n ml-api
```

### Service not accessible

```bash
# Check service
kubectl get svc -n ml-api

# Check endpoints
kubectl get endpoints -n ml-api

# Test from within cluster
kubectl run test --rm -it --image=curlimages/curl -- \
  curl http://ml-api.ml-api.svc.cluster.local
```

## Resources

- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Helm Documentation](https://helm.sh/docs/)
- [ArgoCD Documentation](https://argo-cd.readthedocs.io/)
- Module 006: Kubernetes Introduction (completed exercises)

## Next Steps

After completing this exercise:

1. ✅ Understand Kubernetes deployments
2. ✅ Create Helm charts
3. ✅ Automate deployments with CI/CD
4. ✅ Implement rolling updates
5. ✅ Handle rollbacks

**Move on to**: Exercise 05 - Model Versioning & Artifact Management
