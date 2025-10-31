# ArgoCD GitOps Configuration

This directory contains ArgoCD Application manifests for deploying the ML API using GitOps principles.

## Overview

ArgoCD provides declarative, GitOps continuous delivery for Kubernetes. These manifests define how ArgoCD should deploy and manage the ML API application.

## Files

- **application.yaml** - Production ArgoCD Application
- **application-dev.yaml** - Development ArgoCD Application

## Prerequisites

1. **Install ArgoCD**:
```bash
kubectl create namespace argocd
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml
```

2. **Access ArgoCD UI**:
```bash
# Port forward to ArgoCD server
kubectl port-forward svc/argocd-server -n argocd 8080:443

# Get admin password
kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath="{.data.password}" | base64 -d
```

3. **Install ArgoCD CLI** (optional):
```bash
brew install argocd
# or
curl -sSL -o /usr/local/bin/argocd https://github.com/argoproj/argo-cd/releases/latest/download/argocd-linux-amd64
chmod +x /usr/local/bin/argocd
```

## Usage

### Deploy Development Application

```bash
# Apply ArgoCD Application manifest
kubectl apply -f argocd/application-dev.yaml

# Watch sync status
argocd app get ml-api-dev

# Manual sync if needed
argocd app sync ml-api-dev
```

### Deploy Production Application

```bash
# Apply ArgoCD Application manifest
kubectl apply -f argocd/application.yaml

# Watch sync status
argocd app get ml-api

# Manual sync (automatic sync is enabled)
argocd app sync ml-api
```

## Configuration

### Automatic Sync

Both applications are configured with automatic sync:

```yaml
syncPolicy:
  automated:
    prune: true      # Remove old resources
    selfHeal: true   # Auto-fix drift
```

This means:
- ArgoCD will automatically deploy changes from Git
- Resources not in Git will be removed (prune)
- Manual changes to Kubernetes will be reverted (selfHeal)

### Sync Waves

For complex deployments with dependencies, use sync waves:

```yaml
metadata:
  annotations:
    argocd.argoproj.io/sync-wave: "1"
```

Lower numbers sync first. Useful for:
- Wave 0: Namespaces, CRDs
- Wave 1: ConfigMaps, Secrets
- Wave 2: Deployments, Services

### Image Updates

Two approaches for image updates:

#### 1. GitOps Flow (Recommended)
```bash
# CI/CD updates Git repository with new image tag
# ArgoCD detects change and syncs automatically
```

#### 2. ArgoCD Image Updater
```yaml
metadata:
  annotations:
    argocd-image-updater.argoproj.io/image-list: ml-api=ghcr.io/your-org/ml-api
    argocd-image-updater.argoproj.io/ml-api.update-strategy: semver
```

## Monitoring

### Check Application Status

```bash
# List all applications
argocd app list

# Get application details
argocd app get ml-api

# View sync history
argocd app history ml-api

# View application resources
argocd app resources ml-api
```

### Watch Application

```bash
# Watch application status
watch argocd app get ml-api

# Follow sync progress
argocd app wait ml-api --sync
```

## Rollback

### Using ArgoCD

```bash
# List history
argocd app history ml-api

# Rollback to specific revision
argocd app rollback ml-api 5

# Rollback to previous revision
argocd app rollback ml-api
```

### Using Git

```bash
# Revert Git commit
git revert HEAD
git push

# ArgoCD will automatically sync the revert
```

## Multi-Environment Strategy

### Option 1: Branch-Based

```yaml
# Dev tracks develop branch
targetRevision: develop

# Prod tracks main branch
targetRevision: main
```

### Option 2: Tag-Based

```yaml
# Dev uses latest from branch
targetRevision: HEAD

# Prod uses specific tags
targetRevision: v1.2.3
```

### Option 3: Directory-Based

```
├── environments/
│   ├── dev/
│   │   └── values.yaml
│   ├── staging/
│   │   └── values.yaml
│   └── prod/
│       └── values.yaml
```

## App of Apps Pattern

For managing multiple applications:

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: ml-platform
spec:
  source:
    path: argocd/
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
```

This deploys all applications in the argocd/ directory.

## Notifications

Configure notifications for sync events:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: argocd-notifications-cm
data:
  service.slack: |
    token: $slack-token
  template.app-deployed: |
    message: Application {{.app.metadata.name}} deployed!
  trigger.on-deployed: |
    - when: app.status.operationState.phase in ['Succeeded']
      send: [app-deployed]
```

## Security Best Practices

1. **Use Private Repositories**:
```bash
argocd repo add https://github.com/your-org/ml-api \
  --username your-username \
  --password $GITHUB_TOKEN
```

2. **Use RBAC**:
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: argocd-rbac-cm
data:
  policy.csv: |
    p, role:dev, applications, sync, default/ml-api-dev, allow
    g, dev-team, role:dev
```

3. **Use Projects**:
```yaml
apiVersion: argoproj.io/v1alpha1
kind: AppProject
metadata:
  name: ml-platform
spec:
  sourceRepos:
  - https://github.com/your-org/ml-api
  destinations:
  - namespace: ml-api*
    server: https://kubernetes.default.svc
```

## Troubleshooting

### Application OutOfSync

```bash
# Check diff
argocd app diff ml-api

# Force sync
argocd app sync ml-api --force
```

### Sync Failed

```bash
# Check sync result
argocd app get ml-api

# View logs
kubectl logs -n argocd deployment/argocd-application-controller

# Retry sync
argocd app sync ml-api --retry-limit 5
```

### Resource Not Syncing

```bash
# Check if resource is ignored
argocd app get ml-api --show-params

# Remove ignore annotation
kubectl annotate app ml-api argocd.argoproj.io/compare-options-
```

## Resources

- [ArgoCD Documentation](https://argo-cd.readthedocs.io/)
- [GitOps Best Practices](https://www.gitops.tech/)
- [ArgoCD Best Practices](https://argoproj.github.io/argo-cd/user-guide/best_practices/)
