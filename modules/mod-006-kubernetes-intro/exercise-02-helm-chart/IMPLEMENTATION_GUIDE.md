# Implementation Guide: Helm Charts for ML Applications

## Table of Contents

1. [Introduction](#introduction)
2. [Helm Installation and Basics](#helm-installation-and-basics)
3. [Creating Custom Helm Charts](#creating-custom-helm-charts)
4. [Templates and Values Files](#templates-and-values-files)
5. [Chart Dependencies](#chart-dependencies)
6. [Helm Hooks for ML Workflows](#helm-hooks-for-ml-workflows)
7. [Chart Testing and Validation](#chart-testing-and-validation)
8. [Production ML Chart Deployment](#production-ml-chart-deployment)

---

## Introduction

This implementation guide provides a comprehensive walkthrough for creating, testing, and deploying Helm charts specifically designed for Machine Learning inference workloads. You'll learn how to package ML applications with all their dependencies, manage model versions, and deploy across multiple environments with confidence.

### What is Helm?

Helm is the package manager for Kubernetes, often described as "apt/yum for Kubernetes." It simplifies the deployment and management of complex Kubernetes applications through:

- **Templating**: Generate Kubernetes manifests with reusable templates
- **Versioning**: Track application and configuration versions
- **Rollbacks**: Easy rollback to previous working states
- **Dependencies**: Manage application dependencies declaratively
- **Reusability**: Share charts across teams and organizations

### Why Helm for ML Applications?

ML applications have unique requirements that Helm addresses well:

- **Model Versioning**: Different model versions require different configurations
- **Complex Dependencies**: Database, cache, model storage, monitoring
- **Environment Variations**: Dev models vs production models
- **Resource Management**: GPU allocation, memory requirements vary by model
- **A/B Testing**: Deploy multiple model versions simultaneously
- **Blue/Green Deployments**: Zero-downtime model updates

### Learning Objectives

By completing this guide, you will:

1. Install and configure Helm 3.x
2. Create production-ready Helm charts for ML inference
3. Implement Go templating for dynamic configuration
4. Manage chart dependencies (databases, caches, storage)
5. Use Helm hooks for ML-specific lifecycle management
6. Test and validate charts thoroughly
7. Deploy ML models across multiple environments
8. Implement model versioning and rollback strategies

---

## Helm Installation and Basics

### Installing Helm

#### Option 1: Using Package Manager (Linux)

```bash
# Download Helm installation script
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# Verify installation
helm version
```

#### Option 2: Using Homebrew (macOS)

```bash
brew install helm

# Verify installation
helm version
```

#### Option 3: Manual Installation

```bash
# Download specific version
wget https://get.helm.sh/helm-v3.13.0-linux-amd64.tar.gz

# Extract archive
tar -zxvf helm-v3.13.0-linux-amd64.tar.gz

# Move to PATH
sudo mv linux-amd64/helm /usr/local/bin/helm

# Verify installation
helm version
```

### Helm Architecture Overview

Helm 3 simplified architecture (no Tiller):

```
┌─────────────────────────────────────────────────────────┐
│                    Helm Client (CLI)                    │
│          helm install, upgrade, rollback, etc.          │
└────────────────────────┬────────────────────────────────┘
                         │
                         │ Direct API calls
                         │
┌────────────────────────▼────────────────────────────────┐
│              Kubernetes API Server                      │
│                                                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │ Deployments │  │  Services   │  │ ConfigMaps  │    │
│  └─────────────┘  └─────────────┘  └─────────────┘    │
│                                                          │
│  Release information stored as Kubernetes Secrets       │
└──────────────────────────────────────────────────────────┘
```

### Essential Helm Commands

#### Repository Management

```bash
# Add official Helm stable repository
helm repo add stable https://charts.helm.sh/stable

# Add Bitnami repository (popular for databases)
helm repo add bitnami https://charts.bitnami.com/bitnami

# Update repository cache
helm repo update

# List added repositories
helm repo list

# Search for charts
helm search repo postgresql
helm search hub tensorflow
```

#### Chart Operations

```bash
# Show chart information
helm show chart bitnami/postgresql
helm show values bitnami/postgresql
helm show all bitnami/postgresql

# Download chart without installing
helm pull bitnami/postgresql
helm pull bitnami/postgresql --untar

# Inspect chart structure
helm show values bitnami/postgresql > values.yaml
```

#### Release Management

```bash
# Install a release
helm install my-release bitnami/postgresql

# Install with custom values
helm install my-release bitnami/postgresql -f custom-values.yaml

# Install in specific namespace
helm install my-release bitnami/postgresql -n production --create-namespace

# List releases
helm list
helm list -n production
helm list --all-namespaces

# Get release status
helm status my-release
helm status my-release -n production

# Get release values
helm get values my-release
helm get values my-release --all  # Include defaults

# Get release manifest
helm get manifest my-release

# Show release history
helm history my-release
```

#### Upgrading and Rolling Back

```bash
# Upgrade release
helm upgrade my-release bitnami/postgresql -f new-values.yaml

# Upgrade with atomic flag (auto-rollback on failure)
helm upgrade my-release bitnami/postgresql --atomic --timeout 5m

# Rollback to previous version
helm rollback my-release

# Rollback to specific revision
helm rollback my-release 3

# Uninstall release
helm uninstall my-release
helm uninstall my-release --keep-history  # Keep release history
```

### Helm Chart Structure

A basic Helm chart directory structure:

```
my-ml-chart/
├── Chart.yaml              # Chart metadata
├── values.yaml             # Default configuration values
├── charts/                 # Dependency charts
├── templates/              # Kubernetes manifest templates
│   ├── NOTES.txt          # Post-installation notes
│   ├── _helpers.tpl       # Template helpers
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── configmap.yaml
│   └── ingress.yaml
├── .helmignore            # Files to exclude from packaging
└── README.md              # Chart documentation
```

### Creating Your First Chart

```bash
# Create a new chart
helm create ml-inference

# This generates a complete chart structure
cd ml-inference

# View generated files
tree .

# Test template rendering
helm template test-release . --debug

# Lint the chart
helm lint .

# Install the chart locally
helm install test-release . --dry-run --debug

# Actually install it
helm install test-release . --namespace test --create-namespace
```

---

## Creating Custom Helm Charts

### Chart.yaml - Chart Metadata

The `Chart.yaml` file defines chart metadata and dependencies.

**File: `flask-app/Chart.yaml`**

```yaml
apiVersion: v2
name: ml-inference
description: Production-ready ML inference service with model management
type: application

# Chart version (SemVer 2)
version: 1.0.0

# Application version
appVersion: "1.0.0"

# Chart keywords for discoverability
keywords:
  - machine-learning
  - ml
  - inference
  - tensorflow
  - pytorch
  - scikit-learn
  - flask
  - fastapi

# Maintainer information
maintainers:
  - name: ML Infrastructure Team
    email: ml-infra@company.com
    url: https://ml-team.company.com

# Project URLs
home: https://github.com/company/ml-inference-chart
sources:
  - https://github.com/company/ml-inference-service

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

# Chart icon
icon: https://example.com/ml-icon.png

# Additional metadata
annotations:
  category: MachineLearning
  license: Apache-2.0
  artifacthub.io/changes: |
    - kind: added
      description: Support for TensorFlow Serving
    - kind: added
      description: GPU resource management
    - kind: added
      description: Model versioning support
  artifacthub.io/containsSecurityUpdates: "false"
  artifacthub.io/prerelease: "false"
  artifacthub.io/recommendations: |
    - url: https://artifacthub.io/packages/helm/bitnami/postgresql
    - url: https://artifacthub.io/packages/helm/bitnami/redis
```

### Understanding Chart Versioning

**Chart Version vs App Version:**

- **Chart Version**: Version of the Helm chart itself (configuration changes)
- **App Version**: Version of the application being deployed (code/model changes)

**Examples:**

```yaml
# Scenario 1: New model version, same chart
version: 1.0.0        # No change
appVersion: "2.1.0"   # New model version

# Scenario 2: Chart configuration update
version: 1.1.0        # Chart updated (minor change)
appVersion: "2.1.0"   # Same model

# Scenario 3: Breaking chart change
version: 2.0.0        # Major chart version bump
appVersion: "2.1.0"   # Model version unchanged
```

### .helmignore - Excluding Files

The `.helmignore` file specifies files to exclude from the chart package.

**File: `flask-app/.helmignore`**

```
# Development files
.git/
.gitignore
.DS_Store

# CI/CD files
.github/
.gitlab-ci.yml
Jenkinsfile

# Documentation
*.md
docs/

# Testing
tests/
test-*.yaml
*_test.go

# IDE files
.vscode/
.idea/
*.swp
*.swo
*~

# Temporary files
*.tmp
*.bak
.helmignore

# Environment files
.env
.env.*

# Build artifacts
dist/
build/
*.tar.gz
```

### values.yaml - Default Configuration

The `values.yaml` file contains default configuration values for your chart.

**File: `flask-app/values.yaml`** (ML-focused excerpt)

```yaml
# ============================================================================
# ML Inference Service Configuration
# ============================================================================

# Image configuration
image:
  repository: company/ml-inference
  pullPolicy: IfNotPresent
  # Defaults to chart appVersion if not set
  tag: ""

# Image pull secrets for private registries
imagePullSecrets: []
# - name: regcred

# Service account
serviceAccount:
  create: true
  annotations: {}
  name: ""

# Replica configuration
replicaCount: 3

# Autoscaling configuration
autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 20
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80

# ============================================================================
# ML Model Configuration
# ============================================================================

model:
  # Model name and version
  name: "sentiment-classifier"
  version: "v1.2.0"

  # Model storage backend
  storage:
    type: "s3"  # Options: s3, gcs, azure, pvc
    s3:
      bucket: "ml-models"
      region: "us-west-2"
      prefix: "models"
      endpoint: ""  # For S3-compatible storage

    # Alternative: Persistent Volume
    pvc:
      enabled: false
      size: 50Gi
      storageClass: "fast-ssd"
      mountPath: /models

  # Model framework
  framework: "tensorflow"  # Options: tensorflow, pytorch, sklearn, onnx

  # Model serving parameters
  serving:
    batchSize: 32
    maxBatchDelay: 100  # milliseconds
    workers: 4
    threads: 2

  # Model warmup (preload on startup)
  warmup:
    enabled: true
    samples: 10

# ============================================================================
# Application Configuration
# ============================================================================

flask:
  # Flask configuration
  env: production
  debug: false
  secretKey: ""  # Set via secret or external secret manager

  # Application settings
  maxContentLength: 16777216  # 16MB
  jsonSortKeys: false

  # Logging
  logLevel: INFO
  logFormat: json

# API Configuration
api:
  # Rate limiting
  rateLimit:
    enabled: true
    requestsPerMinute: 100
    burstSize: 20

  # CORS
  cors:
    enabled: true
    origins:
      - "https://app.company.com"
      - "https://admin.company.com"

  # Request timeout
  timeout: 30

# ============================================================================
# Resource Configuration
# ============================================================================

# GPU configuration
gpu:
  enabled: false
  count: 1
  type: "nvidia.com/gpu"  # or "amd.com/gpu"

# Resource limits and requests
resources:
  limits:
    cpu: 2000m
    memory: 4Gi
    # nvidia.com/gpu: 1  # Conditionally added based on gpu.enabled
  requests:
    cpu: 1000m
    memory: 2Gi
    # nvidia.com/gpu: 1

# ============================================================================
# Dependencies Configuration
# ============================================================================

# PostgreSQL configuration (for tracking predictions, metadata)
postgresql:
  enabled: true
  auth:
    username: mlapp
    password: ""  # Set via secret
    database: mlapp
  primary:
    persistence:
      enabled: true
      size: 10Gi

# Redis configuration (for caching predictions)
redis:
  enabled: true
  auth:
    enabled: true
    password: ""  # Set via secret
  master:
    persistence:
      enabled: true
      size: 5Gi

# External database (when postgresql.enabled=false)
database:
  host: ""
  port: 5432
  name: mlapp
  username: mlapp
  password: ""
  sslMode: require

# External Redis (when redis.enabled=false)
cache:
  host: ""
  port: 6379
  password: ""
  database: 0

# ============================================================================
# Networking Configuration
# ============================================================================

service:
  type: ClusterIP
  port: 80
  targetPort: 8080
  annotations: {}

ingress:
  enabled: false
  className: nginx
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "100"
  hosts:
    - host: ml-api.company.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: ml-api-tls
      hosts:
        - ml-api.company.com

# ============================================================================
# Health Checks
# ============================================================================

healthChecks:
  liveness:
    enabled: true
    initialDelaySeconds: 30
    periodSeconds: 10
    timeoutSeconds: 5
    failureThreshold: 3
    path: /health/live

  readiness:
    enabled: true
    initialDelaySeconds: 10
    periodSeconds: 5
    timeoutSeconds: 3
    failureThreshold: 3
    path: /health/ready

  startup:
    enabled: true
    initialDelaySeconds: 0
    periodSeconds: 10
    timeoutSeconds: 5
    failureThreshold: 30
    path: /health/startup

# ============================================================================
# Security Configuration
# ============================================================================

securityContext:
  runAsNonRoot: true
  runAsUser: 1000
  fsGroup: 1000
  capabilities:
    drop:
      - ALL
  readOnlyRootFilesystem: true
  allowPrivilegeEscalation: false

# Network policies
networkPolicy:
  enabled: false
  ingress:
    - from:
        - namespaceSelector:
            matchLabels:
              name: ingress-nginx
  egress:
    - to:
        - namespaceSelector: {}

# Pod Disruption Budget
podDisruptionBudget:
  enabled: true
  minAvailable: 1
  # maxUnavailable: 1

# ============================================================================
# Monitoring Configuration
# ============================================================================

metrics:
  enabled: true
  port: 9090
  path: /metrics

serviceMonitor:
  enabled: false
  interval: 30s
  scrapeTimeout: 10s
  labels: {}

# ============================================================================
# Persistence Configuration
# ============================================================================

persistence:
  enabled: false
  size: 10Gi
  storageClass: ""
  accessMode: ReadWriteOnce
  mountPath: /data

# ============================================================================
# Additional Configuration
# ============================================================================

# Pod annotations
podAnnotations:
  prometheus.io/scrape: "true"
  prometheus.io/port: "9090"
  prometheus.io/path: "/metrics"

# Pod labels
podLabels:
  app.kubernetes.io/component: inference

# Node selection
nodeSelector: {}

# Tolerations
tolerations: []

# Affinity rules
affinity: {}
  # podAntiAffinity:
  #   preferredDuringSchedulingIgnoredDuringExecution:
  #     - weight: 100
  #       podAffinityTerm:
  #         labelSelector:
  #           matchExpressions:
  #             - key: app.kubernetes.io/name
  #               operator: In
  #               values:
  #                 - ml-inference
  #         topologyKey: kubernetes.io/hostname
```

---

## Templates and Values Files

### Template Basics

Helm uses Go templating with Sprig functions to generate Kubernetes manifests dynamically.

#### Basic Syntax

```yaml
# Variable substitution
image: {{ .Values.image.repository }}:{{ .Values.image.tag }}

# Default values
replicas: {{ .Values.replicaCount | default 3 }}

# Conditionals
{{- if .Values.ingress.enabled }}
apiVersion: networking.k8s.io/v1
kind: Ingress
# ...
{{- end }}

# Loops
{{- range .Values.extraEnv }}
- name: {{ .name }}
  value: {{ .value }}
{{- end }}

# Including other templates
labels:
  {{- include "ml-inference.labels" . | nindent 4 }}
```

### Helper Template (_helpers.tpl)

Helper functions for reusable template snippets.

**File: `flask-app/templates/_helpers.tpl`**

```yaml
{{/*
Expand the name of the chart.
*/}}
{{- define "ml-inference.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "ml-inference.fullname" -}}
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
Create chart name and version as used by the chart label.
*/}}
{{- define "ml-inference.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "ml-inference.labels" -}}
helm.sh/chart: {{ include "ml-inference.chart" . }}
{{ include "ml-inference.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/component: inference
ml.company.com/model-name: {{ .Values.model.name }}
ml.company.com/model-version: {{ .Values.model.version }}
ml.company.com/framework: {{ .Values.model.framework }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "ml-inference.selectorLabels" -}}
app.kubernetes.io/name: {{ include "ml-inference.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "ml-inference.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "ml-inference.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Database connection string
*/}}
{{- define "ml-inference.databaseURL" -}}
{{- if .Values.postgresql.enabled }}
{{- printf "postgresql://%s:%s@%s-postgresql:5432/%s" .Values.postgresql.auth.username .Values.postgresql.auth.password .Release.Name .Values.postgresql.auth.database }}
{{- else }}
{{- printf "postgresql://%s:%s@%s:%d/%s?sslmode=%s" .Values.database.username .Values.database.password .Values.database.host (int .Values.database.port) .Values.database.name .Values.database.sslMode }}
{{- end }}
{{- end }}

{{/*
Redis connection string
*/}}
{{- define "ml-inference.redisURL" -}}
{{- if .Values.redis.enabled }}
{{- printf "redis://:%s@%s-redis-master:6379/0" .Values.redis.auth.password .Release.Name }}
{{- else }}
{{- printf "redis://:%s@%s:%d/%d" .Values.cache.password .Values.cache.host (int .Values.cache.port) (int .Values.cache.database) }}
{{- end }}
{{- end }}

{{/*
Model storage configuration
*/}}
{{- define "ml-inference.modelStorageConfig" -}}
{{- if eq .Values.model.storage.type "s3" }}
MODEL_STORAGE_TYPE: "s3"
MODEL_S3_BUCKET: {{ .Values.model.storage.s3.bucket | quote }}
MODEL_S3_REGION: {{ .Values.model.storage.s3.region | quote }}
MODEL_S3_PREFIX: {{ .Values.model.storage.s3.prefix | quote }}
{{- if .Values.model.storage.s3.endpoint }}
MODEL_S3_ENDPOINT: {{ .Values.model.storage.s3.endpoint | quote }}
{{- end }}
{{- else if eq .Values.model.storage.type "pvc" }}
MODEL_STORAGE_TYPE: "local"
MODEL_PATH: {{ .Values.model.storage.pvc.mountPath | quote }}
{{- end }}
{{- end }}

{{/*
Validate required values
*/}}
{{- define "ml-inference.validateValues" -}}
{{- if and (not .Values.postgresql.enabled) (not .Values.database.host) }}
  {{- fail "Either postgresql.enabled must be true or database.host must be set" }}
{{- end }}
{{- if and .Values.gpu.enabled (lt (int .Values.gpu.count) 1) }}
  {{- fail "gpu.count must be at least 1 when gpu.enabled is true" }}
{{- end }}
{{- end }}
```

### Deployment Template

**File: `flask-app/templates/deployment.yaml`** (ML-optimized)

```yaml
{{- include "ml-inference.validateValues" . -}}
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ include "ml-inference.fullname" . }}
  labels:
    {{- include "ml-inference.labels" . | nindent 4 }}
spec:
  {{- if not .Values.autoscaling.enabled }}
  replicas: {{ .Values.replicaCount }}
  {{- end }}
  selector:
    matchLabels:
      {{- include "ml-inference.selectorLabels" . | nindent 6 }}
  template:
    metadata:
      annotations:
        checksum/config: {{ include (print $.Template.BasePath "/configmap.yaml") . | sha256sum }}
        checksum/secret: {{ include (print $.Template.BasePath "/secret.yaml") . | sha256sum }}
        {{- with .Values.podAnnotations }}
        {{- toYaml . | nindent 8 }}
        {{- end }}
      labels:
        {{- include "ml-inference.selectorLabels" . | nindent 8 }}
        {{- with .Values.podLabels }}
        {{- toYaml . | nindent 8 }}
        {{- end }}
    spec:
      {{- with .Values.imagePullSecrets }}
      imagePullSecrets:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      serviceAccountName: {{ include "ml-inference.serviceAccountName" . }}
      securityContext:
        {{- toYaml .Values.securityContext | nindent 8 }}

      # Init container for model download
      initContainers:
        - name: model-downloader
          image: amazon/aws-cli:latest
          command:
            - sh
            - -c
            - |
              echo "Downloading model {{ .Values.model.name }} version {{ .Values.model.version }}"
              aws s3 sync s3://{{ .Values.model.storage.s3.bucket }}/{{ .Values.model.storage.s3.prefix }}/{{ .Values.model.name }}/{{ .Values.model.version }} /models
              echo "Model download complete"
          env:
            - name: AWS_REGION
              value: {{ .Values.model.storage.s3.region | quote }}
          volumeMounts:
            - name: model-storage
              mountPath: /models
          resources:
            requests:
              cpu: 100m
              memory: 128Mi
            limits:
              cpu: 500m
              memory: 512Mi

      containers:
        - name: {{ .Chart.Name }}
          image: "{{ .Values.image.repository }}:{{ .Values.image.tag | default .Chart.AppVersion }}"
          imagePullPolicy: {{ .Values.image.pullPolicy }}

          ports:
            - name: http
              containerPort: 8080
              protocol: TCP
            {{- if .Values.metrics.enabled }}
            - name: metrics
              containerPort: {{ .Values.metrics.port }}
              protocol: TCP
            {{- end }}

          # Health checks
          {{- if .Values.healthChecks.liveness.enabled }}
          livenessProbe:
            httpGet:
              path: {{ .Values.healthChecks.liveness.path }}
              port: http
            initialDelaySeconds: {{ .Values.healthChecks.liveness.initialDelaySeconds }}
            periodSeconds: {{ .Values.healthChecks.liveness.periodSeconds }}
            timeoutSeconds: {{ .Values.healthChecks.liveness.timeoutSeconds }}
            failureThreshold: {{ .Values.healthChecks.liveness.failureThreshold }}
          {{- end }}

          {{- if .Values.healthChecks.readiness.enabled }}
          readinessProbe:
            httpGet:
              path: {{ .Values.healthChecks.readiness.path }}
              port: http
            initialDelaySeconds: {{ .Values.healthChecks.readiness.initialDelaySeconds }}
            periodSeconds: {{ .Values.healthChecks.readiness.periodSeconds }}
            timeoutSeconds: {{ .Values.healthChecks.readiness.timeoutSeconds }}
            failureThreshold: {{ .Values.healthChecks.readiness.failureThreshold }}
          {{- end }}

          {{- if .Values.healthChecks.startup.enabled }}
          startupProbe:
            httpGet:
              path: {{ .Values.healthChecks.startup.path }}
              port: http
            initialDelaySeconds: {{ .Values.healthChecks.startup.initialDelaySeconds }}
            periodSeconds: {{ .Values.healthChecks.startup.periodSeconds }}
            timeoutSeconds: {{ .Values.healthChecks.startup.timeoutSeconds }}
            failureThreshold: {{ .Values.healthChecks.startup.failureThreshold }}
          {{- end }}

          # Environment variables
          env:
            - name: MODEL_NAME
              value: {{ .Values.model.name | quote }}
            - name: MODEL_VERSION
              value: {{ .Values.model.version | quote }}
            - name: MODEL_FRAMEWORK
              value: {{ .Values.model.framework | quote }}
            - name: FLASK_ENV
              value: {{ .Values.flask.env | quote }}
            - name: LOG_LEVEL
              value: {{ .Values.flask.logLevel | quote }}

          envFrom:
            - configMapRef:
                name: {{ include "ml-inference.fullname" . }}
            - secretRef:
                name: {{ include "ml-inference.fullname" . }}

          # Volume mounts
          volumeMounts:
            - name: model-storage
              mountPath: /models
              readOnly: true
            {{- if .Values.persistence.enabled }}
            - name: data
              mountPath: {{ .Values.persistence.mountPath }}
            {{- end }}

          # Resources
          resources:
            {{- if .Values.gpu.enabled }}
            limits:
              cpu: {{ .Values.resources.limits.cpu }}
              memory: {{ .Values.resources.limits.memory }}
              {{ .Values.gpu.type }}: {{ .Values.gpu.count }}
            requests:
              cpu: {{ .Values.resources.requests.cpu }}
              memory: {{ .Values.resources.requests.memory }}
              {{ .Values.gpu.type }}: {{ .Values.gpu.count }}
            {{- else }}
            {{- toYaml .Values.resources | nindent 12 }}
            {{- end }}

      # Volumes
      volumes:
        - name: model-storage
          emptyDir: {}
        {{- if .Values.persistence.enabled }}
        - name: data
          persistentVolumeClaim:
            claimName: {{ include "ml-inference.fullname" . }}
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

### Environment-Specific Values

#### Development Environment

**File: `flask-app/values-dev.yaml`**

```yaml
# Development environment configuration
replicaCount: 1

image:
  tag: "dev-latest"
  pullPolicy: Always

# Enable debug mode
flask:
  env: development
  debug: true
  logLevel: DEBUG

# Minimal resources
resources:
  limits:
    cpu: 500m
    memory: 1Gi
  requests:
    cpu: 100m
    memory: 256Mi

# Disable autoscaling in dev
autoscaling:
  enabled: false

# Use local dependencies
postgresql:
  enabled: true
  primary:
    persistence:
      size: 1Gi

redis:
  enabled: true
  master:
    persistence:
      size: 1Gi

# No ingress in dev (use port-forward)
ingress:
  enabled: false

# Relaxed security for dev
securityContext:
  runAsNonRoot: false
  readOnlyRootFilesystem: false

# No GPU in dev
gpu:
  enabled: false

# Faster health checks
healthChecks:
  liveness:
    initialDelaySeconds: 10
  readiness:
    initialDelaySeconds: 5
  startup:
    failureThreshold: 10

# Development model
model:
  name: "sentiment-classifier"
  version: "dev"
  storage:
    type: "pvc"
    pvc:
      enabled: true
      size: 5Gi
```

#### Production Environment

**File: `flask-app/values-prod.yaml`**

```yaml
# Production environment configuration
replicaCount: 5

image:
  tag: "1.0.0"
  pullPolicy: IfNotPresent

# Production mode
flask:
  env: production
  debug: false
  logLevel: WARNING

# Production resources
resources:
  limits:
    cpu: 2000m
    memory: 4Gi
  requests:
    cpu: 1000m
    memory: 2Gi

# Enable GPU for production
gpu:
  enabled: true
  count: 1

# Enable autoscaling
autoscaling:
  enabled: true
  minReplicas: 5
  maxReplicas: 50
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80

# Use managed services (disable local dependencies)
postgresql:
  enabled: false

redis:
  enabled: false

# External managed database
database:
  host: "postgres.prod.company.internal"
  port: 5432
  name: "ml_inference_prod"
  username: "ml_app_user"
  sslMode: "require"

# External Redis
cache:
  host: "redis.prod.company.internal"
  port: 6379
  database: 0

# Enable ingress with TLS
ingress:
  enabled: true
  className: nginx
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "1000"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
  hosts:
    - host: ml-api.company.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: ml-api-prod-tls
      hosts:
        - ml-api.company.com

# Strict security
securityContext:
  runAsNonRoot: true
  runAsUser: 1000
  fsGroup: 1000
  capabilities:
    drop:
      - ALL
  readOnlyRootFilesystem: true
  allowPrivilegeEscalation: false

# Enable PDB
podDisruptionBudget:
  enabled: true
  minAvailable: 2

# Enable network policies
networkPolicy:
  enabled: true

# Enable monitoring
serviceMonitor:
  enabled: true
  interval: 30s

# Production model from S3
model:
  name: "sentiment-classifier"
  version: "v1.2.0"
  storage:
    type: "s3"
    s3:
      bucket: "ml-models-prod"
      region: "us-west-2"
      prefix: "production/models"

# Pod anti-affinity for HA
affinity:
  podAntiAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
      - labelSelector:
          matchExpressions:
            - key: app.kubernetes.io/name
              operator: In
              values:
                - ml-inference
        topologyKey: kubernetes.io/hostname

# Node selector for GPU nodes
nodeSelector:
  node.kubernetes.io/instance-type: "g4dn.xlarge"

# Tolerate GPU taints
tolerations:
  - key: nvidia.com/gpu
    operator: Exists
    effect: NoSchedule
```

---

## Chart Dependencies

### Adding Dependencies

Update `Chart.yaml` to declare dependencies:

```yaml
dependencies:
  - name: postgresql
    version: "12.x.x"
    repository: https://charts.bitnami.com/bitnami
    condition: postgresql.enabled

  - name: redis
    version: "17.x.x"
    repository: https://charts.bitnami.com/bitnami
    condition: redis.enabled
```

### Managing Dependencies

```bash
# Add Bitnami repository
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo update

# Download dependencies
cd flask-app
helm dependency update

# This creates:
# - charts/ directory with dependency charts
# - Chart.lock file with exact versions

# List dependencies
helm dependency list

# Build dependencies from Chart.lock
helm dependency build
```

### Configuring Dependencies

Configure dependency values in your `values.yaml`:

```yaml
postgresql:
  enabled: true
  auth:
    username: mlapp
    password: ""  # Set via helm install --set
    database: mlapp
  primary:
    persistence:
      enabled: true
      size: 20Gi
      storageClass: "fast-ssd"
    resources:
      requests:
        cpu: 500m
        memory: 1Gi
      limits:
        cpu: 2000m
        memory: 4Gi

redis:
  enabled: true
  auth:
    enabled: true
    password: ""
  master:
    persistence:
      enabled: true
      size: 10Gi
    resources:
      requests:
        cpu: 250m
        memory: 512Mi
      limits:
        cpu: 1000m
        memory: 2Gi
```

### Conditional Dependencies

Use conditions to enable/disable dependencies:

```bash
# Enable PostgreSQL
helm install my-release ./flask-app --set postgresql.enabled=true

# Disable PostgreSQL, use external database
helm install my-release ./flask-app \
  --set postgresql.enabled=false \
  --set database.host=external-db.example.com \
  --set database.password=secretpass
```

---

## Helm Hooks for ML Workflows

Helm hooks allow you to intervene at specific points in a release lifecycle.

### Hook Types

- **pre-install**: Before any resources are created
- **post-install**: After all resources are created
- **pre-upgrade**: Before upgrade
- **post-upgrade**: After upgrade
- **pre-delete**: Before deletion
- **post-delete**: After deletion
- **pre-rollback**: Before rollback
- **post-rollback**: After rollback
- **test**: When running `helm test`

### ML Model Migration Hook

**File: `flask-app/templates/hooks/model-migration-job.yaml`**

```yaml
{{- if .Values.model.migration.enabled }}
apiVersion: batch/v1
kind: Job
metadata:
  name: {{ include "ml-inference.fullname" . }}-model-migration
  labels:
    {{- include "ml-inference.labels" . | nindent 4 }}
  annotations:
    # Run before upgrade to migrate model formats
    "helm.sh/hook": pre-upgrade
    "helm.sh/hook-weight": "-5"
    "helm.sh/hook-delete-policy": before-hook-creation
spec:
  backoffLimit: 3
  template:
    metadata:
      labels:
        {{- include "ml-inference.selectorLabels" . | nindent 8 }}
    spec:
      restartPolicy: Never
      containers:
        - name: model-migration
          image: "{{ .Values.image.repository }}:{{ .Values.image.tag }}"
          command:
            - python
            - -m
            - scripts.migrate_model
          env:
            - name: MODEL_NAME
              value: {{ .Values.model.name | quote }}
            - name: OLD_VERSION
              value: {{ .Values.model.migration.fromVersion | quote }}
            - name: NEW_VERSION
              value: {{ .Values.model.version | quote }}
          envFrom:
            - configMapRef:
                name: {{ include "ml-inference.fullname" . }}
          resources:
            requests:
              cpu: 500m
              memory: 1Gi
            limits:
              cpu: 2000m
              memory: 4Gi
{{- end }}
```

### Database Migration Hook

**File: `flask-app/templates/hooks/db-migration-job.yaml`**

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: {{ include "ml-inference.fullname" . }}-db-migrate
  labels:
    {{- include "ml-inference.labels" . | nindent 4 }}
  annotations:
    "helm.sh/hook": post-install,pre-upgrade
    "helm.sh/hook-weight": "0"
    "helm.sh/hook-delete-policy": hook-succeeded
spec:
  template:
    spec:
      restartPolicy: Never
      containers:
        - name: db-migrate
          image: "{{ .Values.image.repository }}:{{ .Values.image.tag }}"
          command: ["python", "-m", "flask", "db", "upgrade"]
          env:
            - name: DATABASE_URL
              valueFrom:
                secretKeyRef:
                  name: {{ include "ml-inference.fullname" . }}
                  key: database-url
          resources:
            requests:
              cpu: 100m
              memory: 256Mi
```

### Test Hook

**File: `flask-app/templates/hooks/test-connection.yaml`**

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: {{ include "ml-inference.fullname" . }}-test
  labels:
    {{- include "ml-inference.labels" . | nindent 4 }}
  annotations:
    "helm.sh/hook": test
spec:
  restartPolicy: Never
  containers:
    - name: test
      image: curlimages/curl:latest
      command:
        - sh
        - -c
        - |
          # Test health endpoint
          curl -f http://{{ include "ml-inference.fullname" . }}:{{ .Values.service.port }}/health/ready

          # Test prediction endpoint
          curl -f -X POST http://{{ include "ml-inference.fullname" . }}:{{ .Values.service.port }}/predict \
            -H "Content-Type: application/json" \
            -d '{"text": "test input"}'
```

### Running Hooks

```bash
# Hooks run automatically during lifecycle
helm install my-release ./flask-app

# Run test hooks explicitly
helm test my-release

# View hook logs
kubectl logs -l helm.sh/hook=test
```

---

## Chart Testing and Validation

### 1. Syntax Validation

```bash
# Lint the chart
helm lint ./flask-app

# Lint with values file
helm lint ./flask-app -f values-prod.yaml

# Expected output:
# ==> Linting ./flask-app
# [INFO] Chart.yaml: icon is recommended
# 1 chart(s) linted, 0 chart(s) failed
```

### 2. Template Rendering

```bash
# Render all templates
helm template test-release ./flask-app --debug

# Render with specific values
helm template test-release ./flask-app -f values-prod.yaml

# Render specific template
helm template test-release ./flask-app -s templates/deployment.yaml

# Output to file for inspection
helm template test-release ./flask-app > rendered.yaml
```

### 3. Dry-Run Installation

```bash
# Simulate installation
helm install test-release ./flask-app --dry-run --debug

# With custom values
helm install test-release ./flask-app \
  -f values-prod.yaml \
  --set model.version=v2.0.0 \
  --dry-run --debug
```

### 4. Kubernetes Validation

```bash
# Validate manifests against Kubernetes API
helm template test-release ./flask-app | kubectl apply --dry-run=client -f -

# Validate server-side
helm template test-release ./flask-app | kubectl apply --dry-run=server -f -
```

### 5. Automated Testing Script

**File: `scripts/test-chart.sh`**

```bash
#!/bin/bash
set -e

CHART_DIR="./flask-app"
NAMESPACE="test-ml-inference"

echo "======================================"
echo "Helm Chart Testing Suite"
echo "======================================"

# 1. Lint chart
echo "\n[1/6] Linting chart..."
helm lint "$CHART_DIR"

# 2. Test template rendering
echo "\n[2/6] Testing template rendering..."
helm template test-release "$CHART_DIR" > /dev/null

# 3. Test with dev values
echo "\n[3/6] Testing with dev values..."
helm template test-release "$CHART_DIR" -f "$CHART_DIR/values-dev.yaml" > /dev/null

# 4. Test with prod values
echo "\n[4/6] Testing with prod values..."
helm template test-release "$CHART_DIR" -f "$CHART_DIR/values-prod.yaml" > /dev/null

# 5. Validate against Kubernetes
echo "\n[5/6] Validating manifests..."
helm template test-release "$CHART_DIR" | kubectl apply --dry-run=client -f -

# 6. Install and test
echo "\n[6/6] Installing and testing..."
kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -

helm install test-release "$CHART_DIR" \
  -f "$CHART_DIR/values-dev.yaml" \
  -n "$NAMESPACE" \
  --wait --timeout 5m

# Run helm tests
helm test test-release -n "$NAMESPACE"

# Cleanup
helm uninstall test-release -n "$NAMESPACE"
kubectl delete namespace "$NAMESPACE"

echo "\n======================================"
echo "All tests passed!"
echo "======================================"
```

Make it executable:

```bash
chmod +x scripts/test-chart.sh
./scripts/test-chart.sh
```

---

## Production ML Chart Deployment

### Pre-Deployment Checklist

```
[ ] Chart version bumped
[ ] Application version updated
[ ] Dependencies updated
[ ] Secrets created in target namespace
[ ] Database migrations tested
[ ] Model artifacts available in storage
[ ] Resource quotas sufficient
[ ] Monitoring configured
[ ] Rollback plan documented
[ ] Stakeholders notified
```

### Step 1: Prepare Namespace

```bash
# Create namespace
kubectl create namespace ml-production

# Create secrets
kubectl create secret generic ml-app-secrets \
  -n ml-production \
  --from-literal=flask-secret-key="$(openssl rand -base64 32)" \
  --from-literal=database-password="$(openssl rand -base64 32)" \
  --from-literal=redis-password="$(openssl rand -base64 32)"

# Create image pull secret (if using private registry)
kubectl create secret docker-registry regcred \
  -n ml-production \
  --docker-server=registry.company.com \
  --docker-username=robot \
  --docker-password=secret \
  --docker-email=devops@company.com
```

### Step 2: Update Dependencies

```bash
cd flask-app

# Update dependencies
helm dependency update

# Verify Chart.lock
cat Chart.lock
```

### Step 3: Install Release

```bash
# Install with production values
helm install ml-inference ./flask-app \
  -f values-prod.yaml \
  -n ml-production \
  --create-namespace \
  --timeout 10m \
  --wait \
  --atomic

# Flags explained:
# --timeout: Maximum time to wait
# --wait: Wait for all resources to be ready
# --atomic: Rollback on failure
```

### Step 4: Verify Deployment

```bash
# Check release status
helm status ml-inference -n ml-production

# Check pods
kubectl get pods -n ml-production -l app.kubernetes.io/instance=ml-inference

# Check logs
kubectl logs -n ml-production -l app.kubernetes.io/instance=ml-inference --tail=50

# Run tests
helm test ml-inference -n ml-production
```

### Step 5: Upgrade Release

```bash
# Upgrade with new values
helm upgrade ml-inference ./flask-app \
  -f values-prod.yaml \
  --set model.version=v1.3.0 \
  -n ml-production \
  --timeout 10m \
  --wait \
  --atomic

# View rollout status
kubectl rollout status deployment/ml-inference -n ml-production
```

### Step 6: Monitor Deployment

```bash
# Watch pods
watch kubectl get pods -n ml-production

# View events
kubectl get events -n ml-production --sort-by='.lastTimestamp'

# Check resource usage
kubectl top pods -n ml-production
```

### Step 7: Rollback (if needed)

```bash
# View release history
helm history ml-inference -n ml-production

# Rollback to previous version
helm rollback ml-inference -n ml-production

# Rollback to specific revision
helm rollback ml-inference 3 -n ml-production
```

### Blue-Green Deployment for ML Models

Deploy two versions simultaneously for testing:

```bash
# Deploy blue (current production)
helm install ml-blue ./flask-app \
  -f values-prod.yaml \
  --set model.version=v1.0.0 \
  -n ml-production

# Deploy green (new version)
helm install ml-green ./flask-app \
  -f values-prod.yaml \
  --set model.version=v2.0.0 \
  --set service.name=ml-inference-green \
  -n ml-production

# Test green version
kubectl port-forward svc/ml-inference-green 8080:80 -n ml-production

# Switch traffic (update ingress or service selector)
# ...

# Remove blue after validation
helm uninstall ml-blue -n ml-production
```

### Canary Deployment

Gradually shift traffic to new model version:

```bash
# Install stable version (90% traffic)
helm install ml-stable ./flask-app \
  -f values-prod.yaml \
  --set model.version=v1.0.0 \
  --set replicaCount=9 \
  -n ml-production

# Install canary version (10% traffic)
helm install ml-canary ./flask-app \
  -f values-prod.yaml \
  --set model.version=v2.0.0 \
  --set replicaCount=1 \
  -n ml-production

# Monitor metrics
# If successful, gradually increase canary replicas
helm upgrade ml-canary ./flask-app \
  --set replicaCount=5 \
  -n ml-production

helm upgrade ml-stable ./flask-app \
  --set replicaCount=5 \
  -n ml-production

# Eventually replace stable with canary version
```

### Production Deployment Script

**File: `scripts/deploy-production.sh`**

```bash
#!/bin/bash
set -e

CHART_DIR="./flask-app"
RELEASE_NAME="ml-inference"
NAMESPACE="ml-production"
MODEL_VERSION="${MODEL_VERSION:-v1.0.0}"

echo "======================================"
echo "Production Deployment"
echo "======================================"
echo "Release: $RELEASE_NAME"
echo "Namespace: $NAMESPACE"
echo "Model Version: $MODEL_VERSION"
echo "======================================"

# Validate chart
echo "\n[1/5] Validating chart..."
helm lint "$CHART_DIR" -f "$CHART_DIR/values-prod.yaml"

# Update dependencies
echo "\n[2/5] Updating dependencies..."
helm dependency update "$CHART_DIR"

# Dry-run
echo "\n[3/5] Running dry-run..."
helm upgrade --install "$RELEASE_NAME" "$CHART_DIR" \
  -f "$CHART_DIR/values-prod.yaml" \
  --set model.version="$MODEL_VERSION" \
  -n "$NAMESPACE" \
  --dry-run --debug

# Confirm deployment
read -p "Proceed with deployment? (yes/no): " CONFIRM
if [ "$CONFIRM" != "yes" ]; then
  echo "Deployment cancelled"
  exit 0
fi

# Deploy
echo "\n[4/5] Deploying..."
helm upgrade --install "$RELEASE_NAME" "$CHART_DIR" \
  -f "$CHART_DIR/values-prod.yaml" \
  --set model.version="$MODEL_VERSION" \
  -n "$NAMESPACE" \
  --create-namespace \
  --timeout 10m \
  --wait \
  --atomic

# Verify
echo "\n[5/5] Verifying deployment..."
helm test "$RELEASE_NAME" -n "$NAMESPACE"

echo "\n======================================"
echo "Deployment successful!"
echo "======================================"
helm status "$RELEASE_NAME" -n "$NAMESPACE"
```

Usage:

```bash
chmod +x scripts/deploy-production.sh
MODEL_VERSION=v1.2.0 ./scripts/deploy-production.sh
```

---

## Best Practices Summary

### 1. Chart Development

- Use semantic versioning for charts
- Document all values with comments
- Provide sensible defaults
- Make features optional with conditionals
- Use helper functions to keep templates DRY
- Validate required values in templates

### 2. Security

- Never commit secrets to values files
- Use external secret management (Vault, Sealed Secrets)
- Run containers as non-root
- Enable read-only filesystems
- Use network policies
- Scan container images regularly

### 3. ML-Specific

- Version models separately from code
- Use init containers for model downloads
- Implement proper health checks (model loaded)
- Configure appropriate resource limits (CPU/GPU)
- Use persistent volumes for large models
- Implement model warmup strategies
- Monitor inference latency and accuracy

### 4. Operations

- Always test charts before deploying
- Use --atomic flag for automatic rollback
- Monitor deployments closely
- Keep release history
- Document upgrade procedures
- Plan rollback strategies
- Use blue-green or canary deployments for critical changes

### 5. Performance

- Use autoscaling for variable load
- Configure pod disruption budgets
- Use pod anti-affinity for HA
- Optimize container images
- Configure resource requests accurately
- Use node selectors for GPU workloads

---

## Conclusion

You've learned how to create production-ready Helm charts for ML inference applications, including:

- Installing and using Helm effectively
- Creating custom charts with templates
- Managing dependencies
- Using hooks for ML workflows
- Testing and validating charts
- Deploying ML models in production

This foundation will enable you to package, version, and deploy complex ML applications consistently across multiple environments.

### Next Steps

1. Create your own ML inference chart
2. Implement A/B testing with multiple model versions
3. Set up automated chart testing in CI/CD
4. Explore Helmfile for managing multiple releases
5. Contribute charts to Artifact Hub
6. Learn about GitOps with ArgoCD and Helm

### Additional Resources

- [Helm Documentation](https://helm.sh/docs/)
- [Helm Best Practices Guide](https://helm.sh/docs/chart_best_practices/)
- [Artifact Hub](https://artifacthub.io/) - Discover Helm charts
- [Bitnami Charts](https://github.com/bitnami/charts) - Production-ready charts
- [Helm Template Guide](https://helm.sh/docs/chart_template_guide/)
- [Go Template Language](https://pkg.go.dev/text/template)

---

**Document Version:** 1.0
**Last Updated:** 2025
**Target Audience:** Junior AI Infrastructure Engineers
**Estimated Completion Time:** 4-6 hours
