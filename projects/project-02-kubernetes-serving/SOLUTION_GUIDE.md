# Project 02: Kubernetes Model Serving - Solution Guide

## Overview

This solution extends Project 01 by deploying the model API to Kubernetes with production-grade features: auto-scaling, load balancing, rolling updates, monitoring, and high availability.

## Architecture

```
Internet → Ingress → Service → Pods (HPA: 3-10) → Model API
                ↓
         Prometheus/Grafana (Monitoring)
```

## Key Components

### 1. Deployment (`deployment.yaml`)

**Features:**
- **3 Replicas**: Ensures high availability
- **Rolling Updates**: maxSurge=1, maxUnavailable=0 for zero-downtime
- **Resource Limits**: CPU (500m-1000m), Memory (1Gi-2Gi)
- **Health Probes**: Liveness and readiness checks
- **Security**: Non-root user (1000), read-only root filesystem

**Health Checks:**
- **Liveness Probe**: Restarts unhealthy pods
  - Path: `/health`
  - Initial delay: 30s
  - Period: 10s
  - Timeout: 5s

- **Readiness Probe**: Removes unready pods from load balancer
  - Path: `/health`
  - Initial delay: 20s
  - Period: 5s
  - Timeout: 3s

### 2. Service (`service.yaml`)

**ClusterIP Service:**
- Internal load balancer
- Port 80 → 5000 (HTTP)
- Port 8000 (Metrics)

**LoadBalancer Service:**
- External access
- Cloud provider integration (ELB/GLB/Azure LB)

### 3. HPA (`hpa.yaml`)

**Auto-Scaling Configuration:**
- Min replicas: 3
- Max replicas: 10
- CPU target: 70%
- Memory target: 80%

**Scale-up Policy:**
- Add 100% of current pods (double)
- Or add 2 pods
- Whichever is larger
- Every 30 seconds

**Scale-down Policy:**
- Remove 50% of current pods
- Stabilization window: 5 minutes
- Prevents flapping

### 4. Ingress (`ingress.yaml`)

**NGINX Features:**
- Path-based routing
- Rate limiting: 100 req/sec
- Body size limit: 10MB
- Timeout: 30s
- SSL redirect (configurable)

### 5. ConfigMap (`configmap.yaml`)

**Centralized Configuration:**
- Model name
- Log level
- File size limits
- Timeouts

**Benefits:**
- No hardcoded values
- Easy updates without rebuilding
- Environment-specific configs

## Deployment Instructions

### Prerequisites

```bash
# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install kubectl /usr/local/bin/

# Install Minikube (local testing)
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube

# Install Helm
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
```

### Local Deployment (Minikube)

```bash
# Start Minikube
minikube start --cpus=4 --memory=8192

# Enable addons
minikube addons enable metrics-server
minikube addons enable ingress

# Build image in Minikube
eval $(minikube docker-env)
docker build -t model-api:v1.0 -f ../project-01-simple-model-api/docker/Dockerfile ../project-01-simple-model-api/

# Apply manifests
kubectl apply -f kubernetes/configmap.yaml
kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/service.yaml
kubectl apply -f kubernetes/hpa.yaml
kubectl apply -f kubernetes/ingress.yaml

# Verify deployment
kubectl get all -n ml-serving
kubectl describe deployment model-api -n ml-serving

# Test API
kubectl port-forward -n ml-serving svc/model-api 8080:80
curl http://localhost:8080/health
```

### Cloud Deployment (GKE Example)

```bash
# Create GKE cluster
gcloud container clusters create ml-cluster \
  --num-nodes=3 \
  --machine-type=n1-standard-4 \
  --zone=us-central1-a

# Get credentials
gcloud container clusters get-credentials ml-cluster --zone=us-central1-a

# Build and push image
docker build -t gcr.io/PROJECT_ID/model-api:v1.0 .
docker push gcr.io/PROJECT_ID/model-api:v1.0

# Update deployment.yaml with correct image
# Then apply manifests
kubectl apply -f kubernetes/

# Get external IP
kubectl get svc model-api-lb -n ml-serving
```

## Helm Chart

### Chart Structure

```
helm/model-api/
├── Chart.yaml
├── values.yaml
├── templates/
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── hpa.yaml
│   ├── configmap.yaml
│   └── ingress.yaml
```

### Installation

```bash
# Install
helm install model-api ./helm/model-api -n ml-serving --create-namespace

# Upgrade
helm upgrade model-api ./helm/model-api -n ml-serving

# Rollback
helm rollback model-api -n ml-serving

# Uninstall
helm uninstall model-api -n ml-serving
```

### Values Customization

```yaml
# values-production.yaml
replicaCount: 5
resources:
  limits:
    cpu: 2000m
    memory: 4Gi
autoscaling:
  maxReplicas: 20
```

```bash
helm install model-api ./helm/model-api -f values-production.yaml
```

## Monitoring Setup

### Prometheus Integration

```yaml
# monitoring/servicemonitor.yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: model-api
  namespace: ml-serving
spec:
  selector:
    matchLabels:
      app: model-api
  endpoints:
  - port: metrics
    interval: 30s
```

### Grafana Dashboards

Key metrics to monitor:
- Pod count and status
- CPU and memory usage
- Request rate and latency
- Error rate
- Inference time
- Auto-scaling events

## Rolling Updates

### Update Strategy

```bash
# Update image
kubectl set image deployment/model-api \
  model-api=model-api:v1.1 \
  -n ml-serving

# Monitor rollout
kubectl rollout status deployment/model-api -n ml-serving

# Check history
kubectl rollout history deployment/model-api -n ml-serving

# Rollback if needed
kubectl rollout undo deployment/model-api -n ml-serving
```

### Zero-Downtime Deployment

The configuration ensures:
1. New pod starts (maxSurge=1)
2. New pod passes readiness check
3. Traffic shifts to new pod
4. Old pod terminates (maxUnavailable=0)
5. Process repeats for each pod

## Load Testing

```bash
# Install k6
brew install k6  # or download from k6.io

# Run load test
k6 run tests/load-test.js

# Expected results:
# - All requests succeed
# - HPA scales up under load
# - Latency remains acceptable
# - No pod crashes
```

## Troubleshooting

### Pod Not Starting

```bash
# Check pod status
kubectl get pods -n ml-serving
kubectl describe pod POD_NAME -n ml-serving

# Check logs
kubectl logs POD_NAME -n ml-serving

# Common issues:
# - Image pull error: Check image name and registry
# - CrashLoopBackOff: Check application logs
# - Pending: Check resource availability
```

### HPA Not Scaling

```bash
# Check HPA status
kubectl get hpa -n ml-serving
kubectl describe hpa model-api-hpa -n ml-serving

# Check metrics-server
kubectl get deployment metrics-server -n kube-system

# Verify metrics
kubectl top pods -n ml-serving
```

### Ingress Not Working

```bash
# Check ingress
kubectl get ingress -n ml-serving
kubectl describe ingress model-api-ingress -n ml-serving

# Check ingress controller
kubectl get pods -n ingress-nginx

# Test internal service first
kubectl port-forward svc/model-api 8080:80 -n ml-serving
```

## Best Practices

### Resource Management

1. **Set appropriate requests/limits**
   - Requests: Guaranteed resources
   - Limits: Maximum allowed
   - Ratio: 1:2 is common

2. **Monitor actual usage**
   - Adjust based on metrics
   - Use VPA for recommendations

### High Availability

1. **Pod Disruption Budgets**
```yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: model-api-pdb
spec:
  minAvailable: 2
  selector:
    matchLabels:
      app: model-api
```

2. **Multiple availability zones**
```yaml
affinity:
  podAntiAffinity:
    preferredDuringSchedulingIgnoredDuringExecution:
    - weight: 100
      podAffinityTerm:
        topologyKey: topology.kubernetes.io/zone
```

### Security

1. **Network Policies**
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: model-api-netpol
spec:
  podSelector:
    matchLabels:
      app: model-api
  policyTypes:
  - Ingress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
```

2. **Pod Security Standards**
```yaml
securityContext:
  runAsNonRoot: true
  runAsUser: 1000
  fsGroup: 1000
  allowPrivilegeEscalation: false
  capabilities:
    drop:
    - ALL
  readOnlyRootFilesystem: true
```

## Performance Optimization

### Image Optimization

1. Use multi-stage builds
2. Minimize layers
3. Use .dockerignore
4. Cache dependencies

### Pod Startup Optimization

1. Reduce image size
2. Optimize model loading
3. Use init containers for prep work
4. Tune probe timings

## Testing Checklist

- [ ] Pods start successfully
- [ ] Health checks pass
- [ ] Service load balances across pods
- [ ] HPA scales up under load
- [ ] HPA scales down when idle
- [ ] Rolling update works
- [ ] Rollback works
- [ ] Ingress routes traffic
- [ ] Monitoring collects metrics
- [ ] Logs are accessible

## Conclusion

This Kubernetes deployment provides:
- **High availability** through multiple replicas
- **Auto-scaling** based on resource utilization
- **Zero-downtime updates** via rolling deployments
- **Load balancing** across pods
- **Monitoring** integration
- **Production-ready** configuration

The setup can handle production workloads and demonstrates key Kubernetes concepts essential for ML infrastructure engineering.
