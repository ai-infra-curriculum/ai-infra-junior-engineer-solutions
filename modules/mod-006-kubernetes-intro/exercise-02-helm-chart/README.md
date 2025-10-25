# Exercise 02: Helm Chart for Flask Application

## Overview

This exercise demonstrates how to package, deploy, and manage a Flask application on Kubernetes using Helm, the de facto package manager for Kubernetes. You'll learn how to create production-ready Helm charts with support for multiple environments, dependency management, and operational best practices.

## Learning Objectives

By completing this exercise, you will:

1. Understand Helm chart structure and components
2. Create reusable Kubernetes templates with Go templating
3. Implement environment-specific configurations (dev, staging, prod)
4. Manage chart dependencies (PostgreSQL, Redis)
5. Use Helm hooks and lifecycle management
6. Implement security best practices in Helm charts
7. Test and validate Helm charts
8. Deploy and upgrade applications using Helm

## Prerequisites

- Completed Exercise 01 (First Kubernetes Deployment)
- Kubernetes cluster (kind, minikube, or cloud provider)
- Helm 3.x installed
- kubectl configured
- Basic understanding of:
  - Kubernetes resources (Deployments, Services, ConfigMaps, Secrets)
  - YAML syntax
  - Basic Go templating (helpful but not required)

## Architecture

This Helm chart deploys a Flask web application with optional ML inference capabilities, supporting:

- **Application Layer**: Flask application with uWSGI
- **Data Layer**: Optional PostgreSQL database and Redis cache
- **Configuration**: Environment-based configuration (dev/staging/prod)
- **Observability**: Prometheus metrics and health checks
- **Networking**: Ingress with TLS support
- **Storage**: Persistent volumes for model storage
- **Scaling**: Horizontal Pod Autoscaling (HPA)
- **High Availability**: Pod Disruption Budgets (PDB) and anti-affinity rules

```
┌─────────────────────────────────────────────────────────────┐
│                         Ingress                              │
│                  (with TLS termination)                      │
└───────────────────────┬─────────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────────┐
│                       Service                                │
│                    (ClusterIP/LB)                            │
└───────────────────────┬─────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
┌───────▼─────┐  ┌─────▼──────┐  ┌────▼───────┐
│   Pod 1     │  │   Pod 2    │  │   Pod 3    │
│  (Flask)    │  │  (Flask)   │  │  (Flask)   │
└──────┬──────┘  └──────┬─────┘  └─────┬──────┘
       │                │              │
       └────────────────┼──────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
┌───────▼─────┐  ┌─────▼──────┐  ┌────▼───────┐
│ PostgreSQL  │  │   Redis    │  │    PVC     │
│  (optional) │  │ (optional) │  │  (storage) │
└─────────────┘  └────────────┘  └────────────┘
```

## Directory Structure

```
exercise-02-helm-chart/
├── README.md                    # This file
├── STEP_BY_STEP.md             # Detailed implementation guide
├── flask-app/                   # Helm chart directory
│   ├── Chart.yaml              # Chart metadata and dependencies
│   ├── values.yaml             # Default configuration values
│   ├── values-dev.yaml         # Development environment overrides
│   ├── values-prod.yaml        # Production environment overrides
│   ├── .helmignore             # Files to exclude from chart package
│   └── templates/              # Kubernetes manifest templates
│       ├── _helpers.tpl        # Template helper functions
│       ├── NOTES.txt           # Post-installation notes
│       ├── deployment.yaml     # Deployment template
│       ├── service.yaml        # Service template
│       ├── serviceaccount.yaml # ServiceAccount template
│       ├── configmap.yaml      # ConfigMap template
│       ├── secret.yaml         # Secret template
│       ├── ingress.yaml        # Ingress template (conditional)
│       ├── hpa.yaml            # HorizontalPodAutoscaler (conditional)
│       ├── pdb.yaml            # PodDisruptionBudget (conditional)
│       ├── pvc.yaml            # PersistentVolumeClaim (conditional)
│       ├── networkpolicy.yaml  # NetworkPolicy (conditional)
│       └── servicemonitor.yaml # ServiceMonitor for Prometheus (conditional)
└── scripts/                     # Automation scripts
    ├── test-chart.sh           # Chart validation and testing
    ├── install.sh              # Chart installation
    └── upgrade.sh              # Chart upgrade with rollback
```

## Helm Chart Components

### 1. Chart.yaml

Defines chart metadata including:
- Chart name, version, and description
- Application version
- Dependencies (PostgreSQL, Redis)
- Maintainer information
- Keywords and annotations

### 2. values.yaml

Default configuration values including:
- Image configuration (repository, tag, pullPolicy)
- Replica count
- Resource requests and limits
- Flask application settings
- Database and Redis configuration
- ML model settings
- Health check configuration
- Autoscaling parameters
- Security contexts
- Service and Ingress configuration

### 3. templates/

Go templates that generate Kubernetes manifests:

#### Core Resources
- **deployment.yaml**: Application deployment with health checks
- **service.yaml**: Service to expose the application
- **serviceaccount.yaml**: Service account for pod identity
- **configmap.yaml**: Non-sensitive configuration data
- **secret.yaml**: Sensitive data (passwords, keys)

#### Optional Resources (Conditional)
- **ingress.yaml**: HTTP(S) routing with TLS
- **hpa.yaml**: Auto-scaling based on CPU/memory
- **pdb.yaml**: Ensure minimum availability during disruptions
- **pvc.yaml**: Persistent storage
- **networkpolicy.yaml**: Network segmentation
- **servicemonitor.yaml**: Prometheus integration

### 4. Helper Functions (_helpers.tpl)

Reusable template functions:
- `flask-app.name`: Generate chart name
- `flask-app.fullname`: Generate fully qualified name
- `flask-app.labels`: Common labels
- `flask-app.selectorLabels`: Pod selector labels
- `flask-app.databaseURL`: Database connection string
- `flask-app.redisURL`: Redis connection string
- `flask-app.validateValues`: Input validation

## Quick Start

### 1. Test the Chart

Validate chart syntax and templates:

```bash
cd scripts
./test-chart.sh
```

This will:
- Lint the chart
- Validate YAML syntax
- Test template rendering
- Check Kubernetes manifest validity
- Test feature toggles

### 2. Install in Development

Deploy with development settings:

```bash
./install.sh --environment dev --namespace dev-flask-app
```

This creates:
- 1 replica (low resource usage)
- Debug mode enabled
- Local PostgreSQL and Redis (via subcharts)
- Relaxed security settings
- No ingress (use port-forward)

### 3. Access the Application

Port-forward to access locally:

```bash
kubectl port-forward -n dev-flask-app svc/flask-app-flask-app 8080:80
```

Then visit: http://localhost:8080

### 4. Upgrade the Release

Modify values and upgrade:

```bash
./upgrade.sh --environment dev --namespace dev-flask-app
```

### 5. Cleanup

Uninstall the release:

```bash
helm uninstall flask-app -n dev-flask-app
kubectl delete namespace dev-flask-app
```

## Environment-Specific Deployments

### Development Environment

**File**: `values-dev.yaml`

**Characteristics**:
- 1 replica
- Debug mode enabled
- Low resource limits (100m CPU, 128Mi memory)
- Local database and Redis (via subcharts)
- Relaxed security
- No autoscaling
- No ingress

**Usage**:
```bash
helm install flask-app ./flask-app -f values-dev.yaml -n dev
```

### Production Environment

**File**: `values-prod.yaml`

**Characteristics**:
- 3+ replicas with autoscaling
- Debug mode disabled
- Higher resource limits (500m CPU, 512Mi memory)
- External managed database and Redis
- Strict security (non-root, read-only filesystem)
- Autoscaling enabled (3-20 replicas)
- Ingress with TLS
- Pod Disruption Budget
- Network policies
- Prometheus monitoring

**Usage**:
```bash
helm install flask-app ./flask-app -f values-prod.yaml -n production
```

## Advanced Features

### 1. Chart Dependencies

The chart declares optional dependencies on:

- **PostgreSQL** (Bitnami chart)
- **Redis** (Bitnami chart)

Update dependencies:
```bash
helm dependency update ./flask-app
```

This downloads dependency charts to `charts/` directory.

Enable/disable in values:
```yaml
postgresql:
  enabled: true  # Use subchart
  # OR
  enabled: false # Use external database

database:
  host: external-postgres.example.com
  port: 5432
```

### 2. Autoscaling

Enable HPA based on CPU and memory:

```yaml
autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 20
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80
```

### 3. Ingress with TLS

Enable ingress for external access:

```yaml
ingress:
  enabled: true
  className: nginx
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
  hosts:
    - host: app.example.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: app-tls
      hosts:
        - app.example.com
```

### 4. Persistent Storage

Enable persistence for model storage:

```yaml
persistence:
  enabled: true
  size: 10Gi
  storageClass: fast-ssd
  mountPath: /data
```

### 5. Network Policies

Restrict network traffic:

```yaml
networkPolicy:
  enabled: true
  ingress:
    - from:
        - namespaceSelector:
            matchLabels:
              name: ingress-nginx
  egress:
    - to:
        - namespaceSelector: {}
      ports:
        - protocol: TCP
          port: 5432  # PostgreSQL
```

### 6. Prometheus Monitoring

Enable ServiceMonitor for Prometheus Operator:

```yaml
serviceMonitor:
  enabled: true
  interval: 30s
  scrapeTimeout: 10s
```

## Helm Commands Reference

### Chart Management

```bash
# Lint chart
helm lint ./flask-app

# Test template rendering
helm template test-release ./flask-app --debug

# Package chart
helm package ./flask-app

# Update dependencies
helm dependency update ./flask-app

# Show chart values
helm show values ./flask-app
```

### Release Management

```bash
# Install release
helm install <release-name> ./flask-app -f values.yaml -n <namespace>

# Upgrade release
helm upgrade <release-name> ./flask-app -f values.yaml -n <namespace>

# Rollback release
helm rollback <release-name> <revision> -n <namespace>

# List releases
helm list -n <namespace>

# Get release status
helm status <release-name> -n <namespace>

# Show release history
helm history <release-name> -n <namespace>

# Get release values
helm get values <release-name> -n <namespace>

# Get release manifest
helm get manifest <release-name> -n <namespace>

# Uninstall release
helm uninstall <release-name> -n <namespace>
```

### Debugging

```bash
# Dry-run install
helm install <release-name> ./flask-app --dry-run --debug

# Show rendered templates
helm template <release-name> ./flask-app --debug

# Show values with overrides
helm install <release-name> ./flask-app --dry-run --debug -f custom-values.yaml
```

## Template Syntax Guide

### Basic Variable Substitution

```yaml
# Access values
image: {{ .Values.image.repository }}:{{ .Values.image.tag }}

# Access chart metadata
version: {{ .Chart.Version }}

# Access release info
name: {{ .Release.Name }}
namespace: {{ .Release.Namespace }}
```

### Conditionals

```yaml
{{- if .Values.ingress.enabled }}
apiVersion: networking.k8s.io/v1
kind: Ingress
# ...
{{- end }}
```

### Loops

```yaml
{{- range .Values.extraEnv }}
- name: {{ .name }}
  value: {{ .value }}
{{- end }}
```

### Helper Functions

```yaml
# Call helper function
labels:
  {{- include "flask-app.labels" . | nindent 4 }}

# Define helper function (in _helpers.tpl)
{{- define "flask-app.labels" -}}
app.kubernetes.io/name: {{ include "flask-app.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}
```

### Built-in Functions

```yaml
# String manipulation
name: {{ .Values.name | upper }}
name: {{ .Values.name | lower }}
name: {{ .Values.name | quote }}

# Type conversion
port: {{ .Values.port | toString }}

# YAML/JSON conversion
config:
  {{- toYaml .Values.config | nindent 2 }}

# Base64 encoding
data:
  password: {{ .Values.password | b64enc }}

# Default values
replicas: {{ .Values.replicas | default 3 }}
```

## Security Best Practices

### 1. Secret Management

**Don't store secrets in values.yaml**:
```yaml
# ❌ Bad
flask:
  secretKey: "my-secret-key"

# ✅ Good - Use existing secret
flask:
  existingSecret: flask-app-secrets
```

Create external secrets:
```bash
kubectl create secret generic flask-app-secrets \
  --from-literal=secret-key="$(openssl rand -base64 32)"
```

Or use secret managers:
- HashiCorp Vault
- AWS Secrets Manager
- Sealed Secrets
- External Secrets Operator

### 2. Security Contexts

Always run as non-root:
```yaml
securityContext:
  runAsNonRoot: true
  runAsUser: 1000
  capabilities:
    drop:
      - ALL
  readOnlyRootFilesystem: true
  allowPrivilegeEscalation: false
```

### 3. Network Policies

Restrict network traffic:
```yaml
networkPolicy:
  enabled: true
  ingress:
    - from:
        - namespaceSelector:
            matchLabels:
              name: ingress-nginx
  egress:
    - to:
        - namespaceSelector:
            matchLabels:
              name: database
```

### 4. RBAC

Use ServiceAccounts with minimal permissions:
```yaml
serviceAccount:
  create: true
  name: flask-app
```

## Troubleshooting

### Chart Won't Install

```bash
# Check for validation errors
helm lint ./flask-app

# Test template rendering
helm template test-release ./flask-app --debug

# Dry-run install
helm install test-release ./flask-app --dry-run --debug
```

### Pods Not Starting

```bash
# Check pod status
kubectl get pods -n <namespace> -l app.kubernetes.io/instance=<release-name>

# View pod logs
kubectl logs -n <namespace> <pod-name>

# Describe pod for events
kubectl describe pod -n <namespace> <pod-name>

# Check deployment status
kubectl rollout status deployment -n <namespace> <deployment-name>
```

### Upgrade Failed

```bash
# Check upgrade history
helm history <release-name> -n <namespace>

# Rollback to previous version
helm rollback <release-name> <revision> -n <namespace>

# Rollback to previous version (auto)
helm rollback <release-name> -n <namespace>
```

### Values Not Applied

```bash
# Check current values
helm get values <release-name> -n <namespace>

# Show all values (including defaults)
helm get values <release-name> -n <namespace> --all

# Test with specific values
helm template <release-name> ./flask-app -f custom-values.yaml --debug
```

### Template Errors

```bash
# Show full template output
helm template <release-name> ./flask-app --debug

# Test specific template
helm template <release-name> ./flask-app -s templates/deployment.yaml
```

## Testing Strategy

### 1. Syntax Validation

```bash
# Lint chart
helm lint ./flask-app

# Validate YAML
yamllint ./flask-app
```

### 2. Template Testing

```bash
# Render templates
helm template test-release ./flask-app

# Test with different values
helm template test-release ./flask-app -f values-dev.yaml
helm template test-release ./flask-app -f values-prod.yaml
```

### 3. Manifest Validation

```bash
# Validate against Kubernetes API
helm template test-release ./flask-app | kubectl apply --dry-run=client -f -
```

### 4. Integration Testing

```bash
# Install in test namespace
helm install test-release ./flask-app -n test --create-namespace

# Run smoke tests
kubectl run test-pod --rm -it --image=curlimages/curl -- \
  curl http://test-release-flask-app.test.svc.cluster.local/health

# Cleanup
helm uninstall test-release -n test
kubectl delete namespace test
```

### 5. Automated Testing

Use the provided test script:
```bash
./scripts/test-chart.sh
```

## Best Practices

### 1. Chart Design

- **Use semantic versioning**: Update chart version on changes
- **Document all values**: Add comments to values.yaml
- **Provide sensible defaults**: Chart should work with minimal configuration
- **Make features optional**: Use conditionals for optional resources
- **Support multiple environments**: Provide environment-specific values files

### 2. Templates

- **Use helper functions**: Keep templates DRY
- **Add comments**: Explain complex template logic
- **Validate inputs**: Check required values
- **Use consistent naming**: Follow Kubernetes naming conventions
- **Format output**: Use nindent for proper YAML formatting

### 3. Values

- **Organize logically**: Group related values
- **Use nested structure**: Don't flatten unnecessarily
- **Add descriptions**: Comment each value
- **Provide examples**: Show usage in comments
- **Set safe defaults**: Default to secure, production-ready settings

### 4. Security

- **Don't commit secrets**: Use external secret management
- **Run as non-root**: Set security contexts
- **Use read-only filesystems**: Where possible
- **Enable network policies**: Restrict traffic
- **Scan images**: Use trusted, scanned container images

### 5. Operations

- **Test before deploying**: Always test charts thoroughly
- **Use atomic upgrades**: Enable automatic rollback on failure
- **Monitor deployments**: Watch rollout status
- **Keep history**: Don't use --force carelessly
- **Document changes**: Maintain CHANGELOG

## Learning Resources

### Official Documentation
- [Helm Documentation](https://helm.sh/docs/)
- [Helm Best Practices](https://helm.sh/docs/chart_best_practices/)
- [Go Template Language](https://pkg.go.dev/text/template)

### Chart Development
- [Chart Template Guide](https://helm.sh/docs/chart_template_guide/)
- [Chart Hooks](https://helm.sh/docs/topics/charts_hooks/)
- [Dependency Management](https://helm.sh/docs/helm/helm_dependency/)

### Examples
- [Helm Charts Repository](https://github.com/helm/charts)
- [Bitnami Charts](https://github.com/bitnami/charts)
- [Artifact Hub](https://artifacthub.io/)

## Next Steps

After completing this exercise, you should:

1. Understand Helm chart structure and components
2. Be able to create production-ready Helm charts
3. Know how to manage releases and rollbacks
4. Understand environment-specific configurations
5. Be familiar with Helm best practices

**Continue to**: Exercise 03 - Debugging Kubernetes Applications

## Support

For questions or issues with this exercise:
- Review the [STEP_BY_STEP.md](./STEP_BY_STEP.md) guide
- Check the troubleshooting section above
- Consult Helm documentation
- Review example charts in the learning repository

---

**Exercise Type**: Hands-on Implementation
**Difficulty**: Intermediate
**Estimated Time**: 3-4 hours
**Prerequisites**: Exercise 01, Basic Kubernetes knowledge
