# Step-by-Step Implementation Guide: Kubernetes Deployment Automation

## Overview

Automate Kubernetes deployments with CI/CD! Learn automated deployments, GitOps, Helm in CI/CD, deployment strategies, and production rollout automation.

**Time**: 2-3 hours | **Difficulty**: Advanced

---

## Learning Objectives

✅ Deploy to Kubernetes from CI/CD
✅ Implement GitOps workflows
✅ Use Helm in pipelines
✅ Automate deployment verification
✅ Implement canary deployments
✅ Handle deployment rollbacks
✅ Manage multiple environments

---

## Kubernetes Deploy Workflow

```.github/workflows/deploy.yml
name: Deploy to Kubernetes

on:
  push:
    branches: [main]
  workflow_dispatch:
    inputs:
      environment:
        description: 'Environment to deploy'
        required: true
        type: choice
        options:
          - development
          - staging
          - production

jobs:
  deploy:
    runs-on: ubuntu-latest
    environment: ${{ github.event.inputs.environment || 'development' }}

    steps:
      - uses: actions/checkout@v4

      - name: Configure kubectl
        uses: azure/k8s-set-context@v3
        with:
          method: kubeconfig
          kubeconfig: ${{ secrets.KUBE_CONFIG }}

      - name: Deploy with kubectl
        run: |
          kubectl apply -f k8s/namespace.yaml
          kubectl apply -f k8s/configmap.yaml
          kubectl apply -f k8s/secret.yaml
          kubectl apply -f k8s/deployment.yaml
          kubectl apply -f k8s/service.yaml
          kubectl apply -f k8s/ingress.yaml

      - name: Wait for rollout
        run: |
          kubectl rollout status deployment/ml-api -n production
          kubectl wait --for=condition=available --timeout=300s deployment/ml-api -n production

      - name: Run smoke tests
        run: |
          ENDPOINT=$(kubectl get ingress ml-api -n production -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
          curl -f https://$ENDPOINT/health || exit 1

      - name: Rollback on failure
        if: failure()
        run: kubectl rollout undo deployment/ml-api -n production
```

---

## Helm Deployment

```.github/workflows/helm-deploy.yml
name: Helm Deploy

on:
  push:
    tags: ['v*']

jobs:
  deploy:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Install Helm
        uses: azure/setup-helm@v3

      - name: Configure kubectl
        uses: azure/k8s-set-context@v3
        with:
          kubeconfig: ${{ secrets.KUBE_CONFIG }}

      - name: Deploy with Helm
        run: |
          helm upgrade --install ml-api ./charts/ml-api \
            --namespace production \
            --create-namespace \
            --set image.tag=${{ github.ref_name }} \
            --set ingress.enabled=true \
            --set autoscaling.enabled=true \
            --values values/production.yaml \
            --wait \
            --timeout 5m

      - name: Test deployment
        run: helm test ml-api -n production
```

---

## GitOps with ArgoCD

### Application Manifest

```yaml
# argocd/application.yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: ml-api
  namespace: argocd
spec:
  project: default
  source:
    repoURL: https://github.com/org/ml-api
    targetRevision: main
    path: k8s/overlays/production
  destination:
    server: https://kubernetes.default.svc
    namespace: production
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
    syncOptions:
    - CreateNamespace=true
```

---

## Canary Deployment

```.github/workflows/canary.yml
name: Canary Deployment

on:
  workflow_dispatch:
    inputs:
      traffic_percentage:
        description: 'Canary traffic %'
        required: true
        default: '10'

jobs:
  canary:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Deploy canary
        run: |
          kubectl apply -f k8s/canary-deployment.yaml

      - name: Set traffic split
        run: |
          kubectl patch ingress ml-api -p '{
            "metadata": {
              "annotations": {
                "nginx.ingress.kubernetes.io/canary": "true",
                "nginx.ingress.kubernetes.io/canary-weight": "${{ github.event.inputs.traffic_percentage }}"
              }
            }
          }'

      - name: Monitor metrics
        run: |
          # Monitor error rate, latency for 10 minutes
          ./scripts/monitor-canary.sh 10

      - name: Promote or rollback
        run: |
          if [ $METRICS_OK ]; then
            kubectl apply -f k8s/production-deployment.yaml
            kubectl delete -f k8s/canary-deployment.yaml
          else
            kubectl delete -f k8s/canary-deployment.yaml
            exit 1
          fi
```

---

## Best Practices

✅ Use GitOps for declarative deployments
✅ Implement automated health checks
✅ Use Helm for package management
✅ Separate configs per environment
✅ Implement canary/blue-green deployments
✅ Automate rollback on failure
✅ Monitor deployments with metrics
✅ Use deployment gates and approvals
✅ Test in lower environments first
✅ Implement proper RBAC

---

**Kubernetes Deployment Automation mastered!** ☸️
