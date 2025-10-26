# Step-by-Step Implementation Guide: Helm Charts

## Overview

Master Helm, the package manager for Kubernetes! Learn to create production-ready charts, manage dependencies, configure multi-environment deployments, and implement best practices for ML application packaging.

**Time**: 2-3 hours | **Difficulty**: Intermediate

---

## Learning Objectives

✅ Understand Helm architecture and concepts
✅ Create custom Helm charts from scratch
✅ Use templates and values for flexibility
✅ Manage chart dependencies
✅ Implement multi-environment configurations
✅ Package and distribute charts
✅ Upgrade and rollback releases
✅ Use helper functions and chart hooks
✅ Implement production-ready ML chart patterns

---

## Prerequisites

```bash
# Install Helm 3
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# Verify installation
helm version

# Add popular chart repositories
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo add stable https://charts.helm.sh/stable
helm repo update

# List repos
helm repo list
```

---

## Phase 1: Helm Basics

### Key Concepts

**Chart**: Package containing all resource definitions
**Release**: Instance of a chart running in a cluster
**Repository**: Place to store and share charts
**Values**: Configuration parameters for a chart

### Chart Structure

```
flask-app/
├── Chart.yaml           # Chart metadata
├── values.yaml          # Default configuration values
├── charts/              # Chart dependencies
├── templates/           # Kubernetes resource templates
│   ├── NOTES.txt       # Post-install notes
│   ├── _helpers.tpl    # Template helpers
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── configmap.yaml
│   ├── secret.yaml
│   ├── ingress.yaml
│   ├── hpa.yaml
│   ├── pvc.yaml
│   ├── networkpolicy.yaml
│   └── servicemonitor.yaml
└── .helmignore         # Files to ignore
```

---

## Phase 2: Create Chart Metadata

### Chart.yaml

```yaml
apiVersion: v2
name: flask-app
description: A production-ready Helm chart for Flask ML applications
type: application
version: 1.0.0
appVersion: "1.0.0"

keywords:
  - flask
  - python
  - ml
  - machine-learning
  - inference

maintainers:
  - name: AI Infrastructure Team
    email: ai-infra@example.com

home: https://github.com/ai-infra-curriculum/flask-app
sources:
  - https://github.com/ai-infra-curriculum/flask-app

# Chart dependencies
dependencies:
  - name: postgresql
    version: "12.x.x"
    repository: https://charts.bitnami.com/bitnami
    condition: postgresql.enabled
    tags:
      - database
  - name: redis
    version: "17.x.x"
    repository: https://charts.bitnami.com/bitnami
    condition: redis.enabled
    tags:
      - cache

icon: https://raw.githubusercontent.com/docker-library/docs/master/python/logo.png

annotations:
  category: Application
  license: MIT
```

### values.yaml (Core Settings)

```yaml
# Replica count
replicaCount: 2

# Image configuration
image:
  repository: tiangolo/uwsgi-nginx-flask
  pullPolicy: IfNotPresent
  tag: "python3.9"

# Service configuration
service:
  type: ClusterIP
  port: 80
  targetPort: 80

# Resource limits
resources:
  limits:
    cpu: 200m
    memory: 256Mi
  requests:
    cpu: 100m
    memory: 128Mi

# Autoscaling
autoscaling:
  enabled: false
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80

# Flask app config
flask:
  env: production
  debug: false
  logLevel: "INFO"
  secretKey: "change-me-in-production"

# ML Model config
mlModel:
  enabled: false
  modelPath: "/models/model.pkl"
  modelVersion: "1.0.0"
  modelType: "sklearn"
  batchSize: 32

# Health checks
healthCheck:
  livenessProbe:
    enabled: true
    path: /health
    initialDelaySeconds: 30
    periodSeconds: 10
  readinessProbe:
    enabled: true
    path: /ready
    initialDelaySeconds: 10
    periodSeconds: 5

# Dependencies
postgresql:
  enabled: false
  auth:
    username: "flaskuser"
    password: "flaskpass"
    database: "flaskdb"

redis:
  enabled: false
  architecture: standalone
  auth:
    enabled: false
```

---

## Phase 3: Create Template Helpers

### templates/_helpers.tpl

```yaml
{{/*
Expand the name of the chart
*/}}
{{- define "flask-app.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name
*/}}
{{- define "flask-app.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label
*/}}
{{- define "flask-app.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "flask-app.labels" -}}
helm.sh/chart: {{ include "flask-app.chart" . }}
{{ include "flask-app.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "flask-app.selectorLabels" -}}
app.kubernetes.io/name: {{ include "flask-app.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account
*/}}
{{- define "flask-app.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "flask-app.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Database connection string
*/}}
{{- define "flask-app.databaseUrl" -}}
{{- if .Values.postgresql.enabled }}
postgresql://{{ .Values.postgresql.auth.username }}:{{ .Values.postgresql.auth.password }}@{{ include "flask-app.fullname" . }}-postgresql:5432/{{ .Values.postgresql.auth.database }}
{{- else }}
postgresql://{{ .Values.database.username }}:{{ .Values.database.password }}@{{ .Values.database.host }}:{{ .Values.database.port }}/{{ .Values.database.name }}
{{- end }}
{{- end }}

{{/*
Redis URL
*/}}
{{- define "flask-app.redisUrl" -}}
{{- if .Values.redis.enabled }}
redis://{{ include "flask-app.fullname" . }}-redis-master:6379/{{ .Values.redis.database }}
{{- else }}
redis://{{ .Values.redis.host }}:{{ .Values.redis.port }}/{{ .Values.redis.database }}
{{- end }}
{{- end }}
```

---

## Phase 4: Deployment Template

### templates/deployment.yaml

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ include "flask-app.fullname" . }}
  labels:
    {{- include "flask-app.labels" . | nindent 4 }}
spec:
  {{- if not .Values.autoscaling.enabled }}
  replicas: {{ .Values.replicaCount }}
  {{- end }}
  selector:
    matchLabels:
      {{- include "flask-app.selectorLabels" . | nindent 6 }}
  strategy:
    {{- toYaml .Values.strategy | nindent 4 }}
  template:
    metadata:
      annotations:
        checksum/config: {{ include (print $.Template.BasePath "/configmap.yaml") . | sha256sum }}
        checksum/secret: {{ include (print $.Template.BasePath "/secret.yaml") . | sha256sum }}
        {{- with .Values.podAnnotations }}
        {{- toYaml . | nindent 8 }}
        {{- end }}
      labels:
        {{- include "flask-app.selectorLabels" . | nindent 8 }}
    spec:
      {{- with .Values.imagePullSecrets }}
      imagePullSecrets:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      serviceAccountName: {{ include "flask-app.serviceAccountName" . }}
      securityContext:
        {{- toYaml .Values.podSecurityContext | nindent 8 }}
      {{- with .Values.initContainers }}
      initContainers:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      containers:
      - name: {{ .Chart.Name }}
        securityContext:
          {{- toYaml .Values.securityContext | nindent 12 }}
        image: "{{ .Values.image.repository }}:{{ .Values.image.tag | default .Chart.AppVersion }}"
        imagePullPolicy: {{ .Values.image.pullPolicy }}
        ports:
        - name: http
          containerPort: {{ .Values.service.targetPort }}
          protocol: TCP
        env:
        - name: FLASK_ENV
          value: {{ .Values.flask.env }}
        - name: FLASK_DEBUG
          value: {{ .Values.flask.debug | quote }}
        - name: LOG_LEVEL
          value: {{ .Values.flask.logLevel }}
        {{- if .Values.database.enabled }}
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: {{ include "flask-app.fullname" . }}
              key: database-url
        {{- end }}
        {{- if .Values.redis.enabled }}
        - name: REDIS_URL
          value: {{ include "flask-app.redisUrl" . }}
        {{- end }}
        {{- if .Values.mlModel.enabled }}
        - name: MODEL_PATH
          value: {{ .Values.mlModel.modelPath }}
        - name: MODEL_VERSION
          value: {{ .Values.mlModel.modelVersion }}
        - name: MODEL_TYPE
          value: {{ .Values.mlModel.modelType }}
        {{- end }}
        {{- with .Values.extraEnv }}
        {{- toYaml . | nindent 8 }}
        {{- end }}
        {{- with .Values.extraEnvFrom }}
        envFrom:
          {{- toYaml . | nindent 8 }}
        {{- end }}
        {{- if .Values.healthCheck.livenessProbe.enabled }}
        livenessProbe:
          httpGet:
            path: {{ .Values.healthCheck.livenessProbe.path }}
            port: http
          initialDelaySeconds: {{ .Values.healthCheck.livenessProbe.initialDelaySeconds }}
          periodSeconds: {{ .Values.healthCheck.livenessProbe.periodSeconds }}
          timeoutSeconds: {{ .Values.healthCheck.livenessProbe.timeoutSeconds }}
          failureThreshold: {{ .Values.healthCheck.livenessProbe.failureThreshold }}
        {{- end }}
        {{- if .Values.healthCheck.readinessProbe.enabled }}
        readinessProbe:
          httpGet:
            path: {{ .Values.healthCheck.readinessProbe.path }}
            port: http
          initialDelaySeconds: {{ .Values.healthCheck.readinessProbe.initialDelaySeconds }}
          periodSeconds: {{ .Values.healthCheck.readinessProbe.periodSeconds }}
          timeoutSeconds: {{ .Values.healthCheck.readinessProbe.timeoutSeconds }}
          failureThreshold: {{ .Values.healthCheck.readinessProbe.failureThreshold }}
        {{- end }}
        resources:
          {{- toYaml .Values.resources | nindent 12 }}
        volumeMounts:
        {{- if .Values.persistence.enabled }}
        - name: data
          mountPath: {{ .Values.persistence.mountPath }}
        {{- end }}
        {{- with .Values.extraVolumeMounts }}
        {{- toYaml . | nindent 8 }}
        {{- end }}
        {{- with .Values.lifecycle }}
        lifecycle:
          {{- toYaml . | nindent 12 }}
        {{- end }}
      {{- with .Values.sidecars }}
      {{- toYaml . | nindent 6 }}
      {{- end }}
      volumes:
      {{- if .Values.persistence.enabled }}
      - name: data
        persistentVolumeClaim:
          claimName: {{ include "flask-app.fullname" . }}
      {{- end }}
      {{- with .Values.extraVolumes }}
      {{- toYaml . | nindent 6 }}
      {{- end }}
      {{- with .Values.nodeSelector }}
      nodeSelector:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      {{- with .Values.affinity }}
      affinity:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      {{- with .Values.tolerations }}
      tolerations:
        {{- toYaml . | nindent 8 }}
      {{- end }}
```

---

## Phase 5: Service and Ingress

### templates/service.yaml

```yaml
apiVersion: v1
kind: Service
metadata:
  name: {{ include "flask-app.fullname" . }}
  labels:
    {{- include "flask-app.labels" . | nindent 4 }}
  {{- with .Values.service.annotations }}
  annotations:
    {{- toYaml . | nindent 4 }}
  {{- end }}
spec:
  type: {{ .Values.service.type }}
  ports:
  - port: {{ .Values.service.port }}
    targetPort: http
    protocol: TCP
    name: http
  selector:
    {{- include "flask-app.selectorLabels" . | nindent 4 }}
```

### templates/ingress.yaml

```yaml
{{- if .Values.ingress.enabled -}}
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: {{ include "flask-app.fullname" . }}
  labels:
    {{- include "flask-app.labels" . | nindent 4 }}
  {{- with .Values.ingress.annotations }}
  annotations:
    {{- toYaml . | nindent 4 }}
  {{- end }}
spec:
  {{- if .Values.ingress.className }}
  ingressClassName: {{ .Values.ingress.className }}
  {{- end }}
  {{- if .Values.ingress.tls }}
  tls:
    {{- range .Values.ingress.tls }}
    - hosts:
        {{- range .hosts }}
        - {{ . | quote }}
        {{- end }}
      secretName: {{ .secretName }}
    {{- end }}
  {{- end }}
  rules:
    {{- range .Values.ingress.hosts }}
    - host: {{ .host | quote }}
      http:
        paths:
          {{- range .paths }}
          - path: {{ .path }}
            pathType: {{ .pathType }}
            backend:
              service:
                name: {{ include "flask-app.fullname" $ }}
                port:
                  number: {{ $.Values.service.port }}
          {{- end }}
    {{- end }}
{{- end }}
```

---

## Phase 6: ConfigMaps and Secrets

### templates/configmap.yaml

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: {{ include "flask-app.fullname" . }}
  labels:
    {{- include "flask-app.labels" . | nindent 4 }}
data:
  FLASK_ENV: {{ .Values.flask.env | quote }}
  LOG_LEVEL: {{ .Values.flask.logLevel | quote }}
  APP_NAME: {{ .Values.flask.appName | quote }}
  {{- if .Values.mlModel.enabled }}
  MODEL_TYPE: {{ .Values.mlModel.modelType | quote }}
  MODEL_VERSION: {{ .Values.mlModel.modelVersion | quote }}
  BATCH_SIZE: {{ .Values.mlModel.batchSize | quote }}
  {{- end }}
```

### templates/secret.yaml

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: {{ include "flask-app.fullname" . }}
  labels:
    {{- include "flask-app.labels" . | nindent 4 }}
type: Opaque
data:
  flask-secret-key: {{ .Values.flask.secretKey | b64enc | quote }}
  {{- if .Values.database.enabled }}
  database-url: {{ include "flask-app.databaseUrl" . | b64enc | quote }}
  {{- end }}
```

---

## Phase 7: Install and Manage Charts

### Install Chart

```bash
# Dry run (test without installing)
helm install flask-app ./flask-app --dry-run --debug

# Install with default values
helm install flask-app ./flask-app

# Install with custom values
helm install flask-app ./flask-app \
  --set replicaCount=3 \
  --set image.tag=v2.0.0

# Install with values file
helm install flask-app ./flask-app \
  -f values-production.yaml

# Install in specific namespace
helm install flask-app ./flask-app \
  --namespace production \
  --create-namespace
```

### List Releases

```bash
# List all releases
helm list

# List in all namespaces
helm list --all-namespaces

# List releases in specific namespace
helm list -n production
```

### Upgrade Release

```bash
# Upgrade with new values
helm upgrade flask-app ./flask-app \
  --set replicaCount=5

# Upgrade with values file
helm upgrade flask-app ./flask-app \
  -f values-production.yaml

# Force upgrade (recreate resources)
helm upgrade flask-app ./flask-app --force

# Install or upgrade (idempotent)
helm upgrade --install flask-app ./flask-app
```

### Rollback Release

```bash
# Show revision history
helm history flask-app

# Rollback to previous version
helm rollback flask-app

# Rollback to specific revision
helm rollback flask-app 2

# Rollback with dry-run
helm rollback flask-app 2 --dry-run
```

### Uninstall Release

```bash
# Uninstall release
helm uninstall flask-app

# Keep release history
helm uninstall flask-app --keep-history
```

---

## Phase 8: Multi-Environment Configuration

### values-dev.yaml

```yaml
replicaCount: 1

image:
  tag: "dev"
  pullPolicy: Always

flask:
  env: development
  debug: true
  logLevel: DEBUG

resources:
  limits:
    cpu: 100m
    memory: 128Mi
  requests:
    cpu: 50m
    memory: 64Mi

postgresql:
  enabled: true

redis:
  enabled: true
```

### values-staging.yaml

```yaml
replicaCount: 2

image:
  tag: "staging"

flask:
  env: staging
  debug: false
  logLevel: INFO

resources:
  limits:
    cpu: 200m
    memory: 256Mi
  requests:
    cpu: 100m
    memory: 128Mi

autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 5

ingress:
  enabled: true
  hosts:
    - host: flask-staging.example.com
      paths:
        - path: /
          pathType: Prefix
```

### values-production.yaml

```yaml
replicaCount: 3

image:
  repository: registry.company.com/flask-app
  tag: "v1.0.0"
  pullPolicy: IfNotPresent

flask:
  env: production
  debug: false
  logLevel: WARNING
  secretKey: "{{ .Values.externalSecrets.flaskSecretKey }}"

resources:
  limits:
    cpu: 500m
    memory: 512Mi
  requests:
    cpu: 250m
    memory: 256Mi

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 20
  targetCPUUtilizationPercentage: 70

ingress:
  enabled: true
  className: nginx
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
  hosts:
    - host: api.example.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: flask-app-tls
      hosts:
        - api.example.com

persistence:
  enabled: true
  size: 10Gi
  storageClass: fast-ssd

podDisruptionBudget:
  enabled: true
  minAvailable: 2

networkPolicy:
  enabled: true

serviceMonitor:
  enabled: true
  interval: 30s
```

### Deploy to Environments

```bash
# Development
helm install flask-dev ./flask-app \
  -f values-dev.yaml \
  -n development

# Staging
helm install flask-staging ./flask-app \
  -f values-staging.yaml \
  -n staging

# Production
helm install flask-prod ./flask-app \
  -f values-production.yaml \
  -n production
```

---

## Phase 9: Testing Charts

### Lint Chart

```bash
# Lint chart
helm lint ./flask-app

# Lint with values
helm lint ./flask-app -f values-production.yaml
```

### Test Installation

```bash
# Template and view output
helm template flask-app ./flask-app

# Template with values
helm template flask-app ./flask-app \
  -f values-production.yaml

# Show only specific resource
helm template flask-app ./flask-app \
  -s templates/deployment.yaml
```

### Test Script

```bash
#!/bin/bash
# scripts/test-chart.sh

set -e

echo "🧪 Testing Helm Chart..."

# Lint
echo "Running helm lint..."
helm lint ./flask-app

# Test all environments
for env in dev staging production; do
    echo ""
    echo "Testing $env environment..."
    helm template flask-app ./flask-app \
        -f values-$env.yaml \
        --validate
done

echo ""
echo "✅ All tests passed!"
```

---

## Phase 10: Package and Distribute

### Package Chart

```bash
# Package chart
helm package ./flask-app

# Package with specific version
helm package ./flask-app --version 1.2.0

# Package with dependencies
helm dependency update ./flask-app
helm package ./flask-app
```

### Create Chart Repository

```bash
# Create index file
helm repo index ./charts

# Update existing index
helm repo index ./charts --merge index.yaml

# Serve charts locally
helm serve --repo-path ./charts
```

### Use Chart from Repository

```bash
# Add custom repository
helm repo add myrepo https://charts.example.com

# Update repositories
helm repo update

# Install from repository
helm install flask-app myrepo/flask-app

# Search charts
helm search repo flask
```

---

## Best Practices

✅ Use semantic versioning (semver)
✅ Document all values in values.yaml
✅ Use helper templates for common patterns
✅ Implement proper health checks
✅ Set resource limits
✅ Use ConfigMaps/Secrets for configuration
✅ Test charts in all environments
✅ Version control your charts
✅ Sign charts for security (helm sign)
✅ Use chart dependencies wisely
✅ Implement upgrade/rollback strategies
✅ Add NOTES.txt for post-install instructions

---

## Troubleshooting

```bash
# Debug installation
helm install flask-app ./flask-app --debug --dry-run

# Get release status
helm status flask-app

# Get values for release
helm get values flask-app

# Get manifest for release
helm get manifest flask-app

# Get all release info
helm get all flask-app

# Check chart dependencies
helm dependency list ./flask-app
```

---

**Helm charts mastered!** 📦

**Next Exercise**: Debugging Kubernetes applications
