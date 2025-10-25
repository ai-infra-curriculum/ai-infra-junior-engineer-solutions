# Deployment Guide: Prometheus Monitoring Stack

## Overview

This guide covers deploying the Prometheus monitoring stack across **Development**, **Staging**, and **Production** environments with appropriate configurations for each.

---

## Table of Contents

1. [Development Environment](#development-environment)
2. [Staging Environment](#staging-environment)
3. [Production Environment](#production-environment)
4. [Kubernetes Deployment](#kubernetes-deployment)
5. [High Availability Setup](#high-availability-setup)
6. [Upgrading Prometheus](#upgrading-prometheus)

---

## Development Environment

### Purpose
- Local testing and experimentation
- Fast iteration on recording/alerting rules
- Minimal resource usage
- Short retention (7 days)

### Quick Start

```bash
# Clone repository
cd exercise-02-prometheus-stack

# Start development stack
docker-compose up -d

# Access services
open http://localhost:9090  # Prometheus
open http://localhost:9093  # Alertmanager
```

### Development Configuration

**`docker-compose.yml`** (minimal):
```yaml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:v2.48.0
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.retention.time=7d'  # Short retention
      - '--web.enable-lifecycle'  # Enable reload via API
    volumes:
      - ./config/prometheus:/etc/prometheus
      - prometheus_data:/prometheus
    ports:
      - "9090:9090"
    networks:
      - monitoring

  alertmanager:
    image: prom/alertmanager:v0.26.0
    volumes:
      - ./config/alertmanager:/etc/alertmanager
    ports:
      - "9093:9093"
    networks:
      - monitoring

  node-exporter:
    image: prom/node-exporter:v1.7.0
    ports:
      - "9100:9100"
    networks:
      - monitoring

volumes:
  prometheus_data:

networks:
  monitoring:
```

### Development Workflow

```bash
# 1. Edit recording rules
vim config/prometheus/recording_rules.yml

# 2. Validate rules
docker run --rm -v $(pwd)/config/prometheus:/etc/prometheus \
  prom/prometheus:v2.48.0 \
  promtool check rules /etc/prometheus/recording_rules.yml

# 3. Reload Prometheus (no restart!)
curl -X POST http://localhost:9090/-/reload

# 4. Test query
curl 'http://localhost:9090/api/v1/query?query=slo:availability:ratio_rate5m'
```

### Cost
**$0/month** (runs locally)

---

## Staging Environment

### Purpose
- Pre-production testing
- Load testing with realistic data volume
- Alert testing (without paging production on-call)
- Longer retention (30 days)

### AWS ECS Deployment

**Architecture**:
```
┌─────────────────────────────────────────────┐
│           Application Load Balancer          │
│  (prometheus-staging.company.com)            │
└──────────────────┬──────────────────────────┘
                   │
    ┌──────────────┼──────────────┐
    │              │              │
┌───▼────┐   ┌─────▼──┐   ┌──────▼────┐
│Prometheus│ │Alertmgr│   │ Exporters │
│  ECS     │ │  ECS   │   │  ECS      │
│  Fargate │ │Fargate │   │  Fargate  │
└──────────┘ └────────┘   └───────────┘
    │
    └─▶ EFS Volume (metrics storage)
```

**ECS Task Definition** (`ecs-prometheus-staging.json`):
```json
{
  "family": "prometheus-staging",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "4096",
  "memory": "16384",
  "containerDefinitions": [
    {
      "name": "prometheus",
      "image": "prom/prometheus:v2.48.0",
      "essential": true,
      "command": [
        "--config.file=/etc/prometheus/prometheus.yml",
        "--storage.tsdb.path=/prometheus",
        "--storage.tsdb.retention.time=30d",
        "--web.enable-lifecycle"
      ],
      "portMappings": [
        {
          "containerPort": 9090,
          "protocol": "tcp"
        }
      ],
      "mountPoints": [
        {
          "sourceVolume": "prometheus-config",
          "containerPath": "/etc/prometheus",
          "readOnly": true
        },
        {
          "sourceVolume": "prometheus-data",
          "containerPath": "/prometheus"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/prometheus-staging",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "prometheus"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "wget -q --spider http://localhost:9090/-/healthy || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3
      }
    }
  ],
  "volumes": [
    {
      "name": "prometheus-config",
      "efsVolumeConfiguration": {
        "fileSystemId": "fs-12345678",
        "rootDirectory": "/prometheus/config"
      }
    },
    {
      "name": "prometheus-data",
      "efsVolumeConfiguration": {
        "fileSystemId": "fs-12345678",
        "rootDirectory": "/prometheus/data"
      }
    }
  ]
}
```

**Deployment Commands**:
```bash
# 1. Create EFS file system for persistent storage
aws efs create-file-system \
  --performance-mode generalPurpose \
  --throughput-mode bursting \
  --tags Key=Name,Value=prometheus-staging

# 2. Upload configuration to EFS
# (mount EFS locally and copy files)

# 3. Register task definition
aws ecs register-task-definition \
  --cli-input-json file://ecs-prometheus-staging.json

# 4. Create ECS service
aws ecs create-service \
  --cluster monitoring-staging \
  --service-name prometheus \
  --task-definition prometheus-staging \
  --desired-count 1 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx],securityGroups=[sg-xxx],assignPublicIp=ENABLED}"

# 5. Create Application Load Balancer target group
aws elbv2 create-target-group \
  --name prometheus-staging \
  --protocol HTTP \
  --port 9090 \
  --vpc-id vpc-xxx \
  --target-type ip \
  --health-check-path /-/healthy

# 6. Register targets and create listener rules
```

### Cost
**~$150-200/month**
- ECS Fargate (4 vCPU, 16GB): ~$100/month
- EFS storage (100GB): ~$30/month
- ALB: ~$20/month
- Data transfer: ~$10-20/month

---

## Production Environment

### Purpose
- Serve production monitoring needs
- High availability (2+ replicas)
- Long retention (30 days local, unlimited remote)
- Alert delivery to on-call rotation
- Security hardening

### Kubernetes Production Deployment

**Namespace and RBAC**:
```bash
kubectl create namespace monitoring

# Create service account
kubectl create serviceaccount prometheus -n monitoring

# Create cluster role for Prometheus (scrape permissions)
kubectl apply -f k8s/rbac.yaml
```

**`k8s/rbac.yaml`**:
```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: prometheus
rules:
  - apiGroups: [""]
    resources:
      - nodes
      - nodes/metrics
      - services
      - endpoints
      - pods
    verbs: ["get", "list", "watch"]
  - apiGroups: [""]
    resources:
      - configmaps
    verbs: ["get"]
  - nonResourceURLs: ["/metrics"]
    verbs: ["get"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: prometheus
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: prometheus
subjects:
  - kind: ServiceAccount
    name: prometheus
    namespace: monitoring
```

**Prometheus StatefulSet**:
```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: prometheus
  namespace: monitoring
spec:
  serviceName: prometheus
  replicas: 2  # HA setup
  selector:
    matchLabels:
      app: prometheus
  template:
    metadata:
      labels:
        app: prometheus
    spec:
      serviceAccountName: prometheus
      securityContext:
        fsGroup: 65534
        runAsNonRoot: true
        runAsUser: 65534
      containers:
      - name: prometheus
        image: prom/prometheus:v2.48.0
        args:
          - '--config.file=/etc/prometheus/prometheus.yml'
          - '--storage.tsdb.path=/prometheus'
          - '--storage.tsdb.retention.time=30d'
          - '--web.enable-lifecycle'
          - '--web.enable-admin-api'
        ports:
        - name: http
          containerPort: 9090
        volumeMounts:
        - name: config
          mountPath: /etc/prometheus
        - name: data
          mountPath: /prometheus
        resources:
          requests:
            cpu: "4000m"
            memory: "16Gi"
          limits:
            cpu: "8000m"
            memory: "32Gi"
        livenessProbe:
          httpGet:
            path: /-/healthy
            port: 9090
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /-/ready
            port: 9090
          initialDelaySeconds: 30
          periodSeconds: 5
      volumes:
      - name: config
        configMap:
          name: prometheus-config
  volumeClaimTemplates:
  - metadata:
      name: data
    spec:
      accessModes: ["ReadWriteOnce"]
      storageClassName: gp3
      resources:
        requests:
          storage: 500Gi
```

**ConfigMap for Prometheus Config**:
```bash
kubectl create configmap prometheus-config \
  --from-file=prometheus.yml=config/prometheus/prometheus.yml \
  --from-file=recording_rules.yml=config/prometheus/recording_rules.yml \
  --from-file=alerting_rules.yml=config/prometheus/alerting_rules.yml \
  -n monitoring
```

**Service (for Grafana, apps to query)**:
```yaml
apiVersion: v1
kind: Service
metadata:
  name: prometheus
  namespace: monitoring
spec:
  type: ClusterIP
  selector:
    app: prometheus
  ports:
  - name: http
    port: 9090
    targetPort: 9090
```

**Ingress (for external access)**:
```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: prometheus
  namespace: monitoring
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/auth-type: basic
    nginx.ingress.kubernetes.io/auth-secret: prometheus-basic-auth
spec:
  ingressClassName: nginx
  tls:
  - hosts:
    - prometheus.company.com
    secretName: prometheus-tls
  rules:
  - host: prometheus.company.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: prometheus
            port:
              number: 9090
```

### Deploy Alertmanager (HA)

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: alertmanager
  namespace: monitoring
spec:
  serviceName: alertmanager
  replicas: 3  # HA cluster
  selector:
    matchLabels:
      app: alertmanager
  template:
    metadata:
      labels:
        app: alertmanager
    spec:
      containers:
      - name: alertmanager
        image: prom/alertmanager:v0.26.0
        args:
          - '--config.file=/etc/alertmanager/alertmanager.yml'
          - '--storage.path=/alertmanager'
          - '--cluster.listen-address=0.0.0.0:9094'
          - '--cluster.peer=alertmanager-0.alertmanager.monitoring.svc:9094'
          - '--cluster.peer=alertmanager-1.alertmanager.monitoring.svc:9094'
          - '--cluster.peer=alertmanager-2.alertmanager.monitoring.svc:9094'
        ports:
        - name: http
          containerPort: 9093
        - name: cluster
          containerPort: 9094
        volumeMounts:
        - name: config
          mountPath: /etc/alertmanager
        - name: data
          mountPath: /alertmanager
        resources:
          requests:
            cpu: "500m"
            memory: "1Gi"
      volumes:
      - name: config
        configMap:
          name: alertmanager-config
  volumeClaimTemplates:
  - metadata:
      name: data
    spec:
      accessModes: ["ReadWriteOnce"]
      storageClassName: gp3
      resources:
        requests:
          storage: 10Gi
```

### Cost
**~$400-600/month** (AWS EKS)
- EKS cluster: ~$75/month
- EC2 instances (3 × m5.2xlarge): ~$350/month
- EBS volumes (500GB × 2): ~$100/month
- Load Balancer: ~$20/month
- Data transfer: ~$20-50/month

---

## Kubernetes Deployment

### Using Prometheus Operator (Recommended)

```bash
# Install Prometheus Operator with Helm
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# Install kube-prometheus-stack (Prometheus + Grafana + Alertmanager)
helm install kube-prometheus-stack prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace \
  --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false \
  --set prometheus.prometheusSpec.retention=30d \
  --set prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.resources.requests.storage=500Gi \
  --set alertmanager.enabled=true \
  --set grafana.enabled=true

# Access Prometheus
kubectl port-forward -n monitoring svc/kube-prometheus-stack-prometheus 9090:9090
```

**Custom PrometheusRule** (for SLO alerts):
```yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: slo-alerts
  namespace: monitoring
spec:
  groups:
    - name: slo_availability
      interval: 15s
      rules:
        - alert: SLOAvailabilityFastBurn
          expr: |
            slo:availability:burn_rate:1h > 14.4
            and
            slo:availability:burn_rate:6h > 14.4
          for: 2m
          labels:
            severity: critical
          annotations:
            summary: "Fast SLO burn detected"
            description: "Burning error budget at 14.4x rate"
```

---

## High Availability Setup

### Active-Active Prometheus with Alertmanager Deduplication

```
┌──────────────┐        ┌──────────────┐
│ Prometheus A │        │ Prometheus B │
│              │        │              │
│ Scrapes all  │        │ Scrapes all  │
│ targets      │        │ targets      │
│              │        │              │
│ Sends alerts │        │ Sends alerts │
└──────┬───────┘        └──────┬───────┘
       │                       │
       │  ┌────────────────────┘
       │  │
       ▼  ▼
┌──────────────────────────────┐
│   Alertmanager Cluster       │
│                              │
│  ┌────────┐  ┌────────┐     │
│  │  AM-1  │──│  AM-2  │     │
│  │        │  │        │     │
│  │ Dedup  │  │ Dedup  │     │
│  └────────┘  └────────┘     │
│       Gossip Protocol        │
└──────────────────────────────┘
```

**Benefits**:
- Both Prometheus instances collect identical data
- Alertmanager deduplicates alerts from both
- If one Prometheus fails, other continues
- No single point of failure

---

## Upgrading Prometheus

### Zero-Downtime Upgrade

```bash
# 1. Verify new version compatibility
promtool check config config/prometheus/prometheus.yml

# 2. Kubernetes rolling update
kubectl set image statefulset/prometheus prometheus=prom/prometheus:v2.49.0 -n monitoring

# 3. Monitor rollout
kubectl rollout status statefulset/prometheus -n monitoring

# 4. Verify targets and alerts
curl -s http://prometheus.company.com/api/v1/targets | jq '.data.activeTargets[] | select(.health!="up")'
```

### Rollback

```bash
# Rollback to previous version
kubectl rollout undo statefulset/prometheus -n monitoring

# Or specific revision
kubectl rollout undo statefulset/prometheus --to-revision=2 -n monitoring
```

---

## Summary

| Environment | Replicas | Retention | Resources | Cost/Month |
|-------------|----------|-----------|-----------|------------|
| Development | 1 | 7 days | Minimal | $0 |
| Staging | 1 | 30 days | Medium | $150-200 |
| Production | 2-3 (HA) | 30 days | High | $400-600 |

**Deployment Best Practices**:
- ✅ Use StatefulSet for Prometheus (persistent storage)
- ✅ Deploy Alertmanager in HA cluster (3 replicas)
- ✅ Use Prometheus Operator for Kubernetes
- ✅ Implement authentication (basic auth or OAuth)
- ✅ Enable TLS for external access
- ✅ Set resource limits and requests
- ✅ Monitor Prometheus itself (meta-monitoring)
- ✅ Backup configuration files to version control
- ✅ Use remote write for long-term storage
- ✅ Test alert delivery before production
