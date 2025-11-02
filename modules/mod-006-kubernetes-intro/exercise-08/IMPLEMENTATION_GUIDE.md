# Implementation Guide: Kubernetes Autoscaling for ML Workloads

## Overview

This guide provides step-by-step instructions for implementing Kubernetes autoscaling for ML workloads using HPA, VPA, KEDA, and cluster autoscaling strategies.

**Estimated Time:** 3-4 hours
**Difficulty:** Intermediate to Advanced

## Prerequisites

- Kubernetes cluster (minikube, kind, or cloud cluster)
- kubectl installed and configured
- Helm 3 installed
- Basic understanding of Kubernetes concepts
- 8GB+ available cluster resources

## Phase 1: Environment Setup (30 minutes)

### Step 1.1: Verify Prerequisites

```bash
# Check kubectl
kubectl version --client

# Check Helm
helm version

# Check cluster connection
kubectl cluster-info

# Check available resources
kubectl top nodes
```

### Step 1.2: Run Automated Setup

```bash
# Clone the repository
cd modules/mod-006-kubernetes-intro/exercise-08

# Make scripts executable
chmod +x scripts/*.sh

# Run environment setup
./scripts/setup_environment.sh
```

This script will install:
- ✅ Metrics Server
- ✅ Prometheus
- ✅ Prometheus Adapter
- ✅ Vertical Pod Autoscaler
- ✅ KEDA
- ✅ Sample workloads

**Expected Output:**
```
========================================
Installation Summary
========================================
[INFO] Installed Components:
  ✅ Metrics Server
  ✅ Prometheus
  ✅ Prometheus Adapter
  ✅ Vertical Pod Autoscaler
  ✅ KEDA
```

### Step 1.3: Verify Installation

```bash
# Check all components
kubectl get pods -n kube-system | grep -E "metrics-server|vpa"
kubectl get pods -n monitoring
kubectl get pods -n keda

# Verify metrics are available
kubectl top nodes
kubectl top pods -A
```

**Troubleshooting:**

If metrics-server shows errors:
```bash
# For local clusters (minikube/kind), disable TLS verification
kubectl patch deployment metrics-server -n kube-system --type='json' \
  -p='[{"op": "add", "path": "/spec/template/spec/containers/0/args/-", "value": "--kubelet-insecure-tls"}]'

# Restart metrics-server
kubectl rollout restart deployment metrics-server -n kube-system
```

## Phase 2: Horizontal Pod Autoscaler (CPU-based) (30 minutes)

### Step 2.1: Understand HPA Basics

HPA automatically scales the number of pod replicas based on observed metrics.

**Formula:**
```
desiredReplicas = ceil(currentReplicas * (currentMetric / targetMetric))
```

**Example:**
- Current replicas: 2
- Current CPU: 80% (of requests)
- Target CPU: 50%
- Desired = ceil(2 * (80/50)) = ceil(3.2) = 4 replicas

### Step 2.2: Deploy Fraud Detection API with HPA

```bash
# Apply the manifest
kubectl apply -f manifests/10-model-serving-hpa.yaml

# Verify deployment
kubectl get deployment fraud-detector-api -n ml-platform
kubectl get hpa fraud-detector-hpa -n ml-platform
kubectl get pods -n ml-platform -l app=fraud-detector
```

**Expected Output:**
```
NAME                  REFERENCE                       TARGETS   MINPODS   MAXPODS   REPLICAS
fraud-detector-hpa    Deployment/fraud-detector-api   15%/50%   2         10        2
```

### Step 2.3: Examine HPA Configuration

```bash
kubectl describe hpa fraud-detector-hpa -n ml-platform
```

**Key Configuration:**
- **Min Replicas:** 2 (high availability)
- **Max Replicas:** 10 (cost control)
- **Target CPU:** 50% utilization
- **Target Memory:** 70% utilization
- **Scale-Down Delay:** 5 minutes (prevents thrashing)

### Step 2.4: Test Scaling Behavior

```bash
# Check initial state
kubectl get hpa fraud-detector-hpa -n ml-platform
kubectl top pods -n ml-platform -l app=fraud-detector

# Generate load
./scripts/load_generator.sh fraud-detector-api 1000 10

# Monitor scaling in real-time (in another terminal)
watch 'kubectl get hpa,pods -n ml-platform -l app=fraud-detector'
```

**Expected Behavior:**
1. **Initial State:** 2 replicas, 15-20% CPU
2. **During Load:** CPU increases to 70-90%
3. **Scale-Up:** HPA scales to 6-8 replicas within 90 seconds
4. **Load Ends:** CPU drops to 15-20%
5. **Scale-Down:** HPA scales back to 2 replicas after 5 minutes

### Step 2.5: Analyze Scaling Events

```bash
# View HPA events
kubectl describe hpa fraud-detector-hpa -n ml-platform | tail -n 30

# View pod events
kubectl get events -n ml-platform --sort-by='.lastTimestamp' | grep fraud-detector

# Check scaling metrics
kubectl get hpa fraud-detector-hpa -n ml-platform -o yaml | yq '.status'
```

## Phase 3: HPA with Custom Metrics (45 minutes)

### Step 3.1: Understand Custom Metrics

Custom metrics allow scaling based on application-specific metrics:
- HTTP requests per second
- Inference latency (P95, P99)
- Queue depth
- Active connections
- Cache hit rate

### Step 3.2: Verify Prometheus Adapter

```bash
# Check if custom metrics API is available
kubectl get apiservices | grep custom.metrics

# List available custom metrics
kubectl get --raw "/apis/custom.metrics.k8s.io/v1beta1" | jq -r '.resources[].name'
```

### Step 3.3: Deploy Recommendation API with Custom Metrics HPA

```bash
# Apply the manifest
kubectl apply -f manifests/11-model-serving-custom-metrics.yaml

# Verify deployment
kubectl get deployment recommendation-api -n ml-platform
kubectl get hpa recommendation-hpa -n ml-platform
```

### Step 3.4: Examine Multi-Metric HPA

```bash
kubectl describe hpa recommendation-hpa -n ml-platform | grep -A 30 "Metrics:"
```

**Configuration:**
```yaml
metrics:
- CPU utilization: 70%
- Memory utilization: 80%
- HTTP requests/second: 100 req/s per pod
- Inference latency P95: <200ms
- Active connections: <50 per pod
```

**Scaling Logic:**
HPA calculates desired replicas for each metric and uses the **maximum** value.

### Step 3.5: Test Custom Metrics Scaling

```bash
# Generate high request rate
./scripts/load_generator.sh recommendation-api 5000 20

# Monitor which metric drives scaling
watch 'kubectl get hpa recommendation-hpa -n ml-platform && echo && kubectl describe hpa recommendation-hpa -n ml-platform | grep -A 5 "Metrics:"'
```

**Observation:**
The HPA will scale based on whichever metric requires the most replicas.

### Step 3.6: Query Custom Metrics Directly

```bash
# Query request rate metric
kubectl get --raw "/apis/custom.metrics.k8s.io/v1beta1/namespaces/ml-platform/pods/*/http_requests_per_second" | jq .

# Query latency metric
kubectl get --raw "/apis/custom.metrics.k8s.io/v1beta1/namespaces/ml-platform/pods/*/inference_latency_p95_seconds" | jq .
```

## Phase 4: Vertical Pod Autoscaler (45 minutes)

### Step 4.1: Understand VPA

VPA automatically adjusts CPU and memory requests/limits based on actual usage.

**Components:**
1. **Recommender:** Analyzes usage and provides recommendations
2. **Updater:** Evicts pods that need resource updates
3. **Admission Controller:** Sets resources on new pods

### Step 4.2: Deploy Training Job with VPA

```bash
# Apply the manifest
kubectl apply -f manifests/20-training-job-vpa.yaml

# Verify VPA
kubectl get vpa model-training-vpa -n ml-platform
kubectl get pods -n ml-platform -l app=model-training
```

### Step 4.3: Monitor VPA Recommendations

VPA needs 5-10 minutes to gather data and generate recommendations.

```bash
# Wait for recommendations
sleep 600

# Check VPA status
kubectl describe vpa model-training-vpa -n ml-platform

# View recommendations
kubectl get vpa model-training-vpa -n ml-platform -o yaml | yq '.status.recommendation'
```

**Expected Output:**
```yaml
recommendation:
  containerRecommendations:
  - containerName: trainer
    lowerBound:      # Minimum for basic operation
      cpu: 250m
      memory: 512Mi
    target:          # Recommended requests
      cpu: 500m
      memory: 1Gi
    upperBound:      # For peak usage
      cpu: 800m
      memory: 2Gi
```

### Step 4.4: Understand VPA Update Modes

VPA has 4 update modes:

| Mode | Behavior | Use Case |
|------|----------|----------|
| **Off** | Recommendations only | Monitoring and manual tuning |
| **Initial** | Apply on pod creation only | Workloads that can't tolerate restarts |
| **Recreate** | Evict and recreate pods | Standard autoscaling |
| **Auto** | Automatic (same as Recreate) | Future: in-place updates |

### Step 4.5: Test VPA Update Modes

```bash
# Deploy workloads with different VPA modes
kubectl apply -f manifests/21-vpa-modes.yaml

# Compare recommendations across modes
kubectl get vpa -n ml-platform
kubectl describe vpa vpa-mode-off -n ml-platform
kubectl describe vpa vpa-mode-auto -n ml-platform
```

### Step 4.6: Observe VPA in Action

```bash
# Check current pod resources
kubectl get pods -n ml-platform -l app=model-training -o jsonpath='{.items[0].spec.containers[0].resources.requests}'

# Wait for VPA to evict pod (if in Auto/Recreate mode)
kubectl get events -n ml-platform --sort-by='.lastTimestamp' | grep vpa

# Check updated pod resources
kubectl get pods -n ml-platform -l app=model-training -o jsonpath='{.items[0].spec.containers[0].resources.requests}'
```

## Phase 5: KEDA Event-Driven Autoscaling (45 minutes)

### Step 5.1: Understand KEDA

KEDA extends Kubernetes autoscaling with:
- **Scale to Zero:** Save costs when idle
- **50+ Scalers:** Queues, databases, metrics, schedules
- **Event-Driven:** React to external events

### Step 5.2: Deploy Batch Inference with Queue-Based Scaling

```bash
# Apply KEDA scalers
kubectl apply -f manifests/30-keda-scalers.yaml

# Verify ScaledObjects
kubectl get scaledobject -n ml-platform
kubectl describe scaledobject batch-inference-scaler -n ml-platform
```

**Configuration:**
```yaml
minReplicaCount: 0           # Can scale to zero
maxReplicaCount: 20
pollingInterval: 30          # Check metrics every 30s
cooldownPeriod: 300          # Wait 5 min before scaling to zero
```

### Step 5.3: Test Scale-to-Zero Behavior

```bash
# Check initial state (should be 0 replicas)
kubectl get deployment batch-inference-processor -n ml-platform

# Simulate queue messages (increase metric)
# In production, this would be real messages in a queue
kubectl port-forward -n monitoring svc/prometheus-server 9090:80 &

# Open Prometheus and run query: inference_queue_length{queue="batch"}
# Or use curl to push metrics:
curl -X POST http://localhost:9090/metrics/job/test \
  --data-binary 'inference_queue_length{queue="batch"} 100'

# Watch KEDA scale up from 0
kubectl get pods -n ml-platform -l app=batch-inference --watch
```

**Expected Behavior:**
- Queue empty → 0 replicas
- Queue has 100 messages → scales to 10 replicas (100/10 per pod)
- Processing completes → scales back to 0 after 5 minutes

### Step 5.4: Deploy Schedule-Based Scaling

The `scheduled-training-scaler` demonstrates cron-based scaling:

```yaml
triggers:
- type: cron
  metadata:
    timezone: America/New_York
    start: 0 9 * * 1-5      # 9 AM Monday-Friday
    end: 0 18 * * 1-5       # 6 PM Monday-Friday
    desiredReplicas: "3"
```

**Business Value:**
- Training jobs run only during business hours
- Saves 67% compute cost (9 hours vs 24 hours daily)
- Zero replicas nights and weekends

### Step 5.5: Test Cron-Based Scaling

```bash
# Check current replicas (0 outside business hours)
kubectl get deployment scheduled-training -n ml-platform

# Edit start time to trigger scaling immediately
kubectl edit scaledobject scheduled-training-scaler -n ml-platform
# Change start time to current time + 1 minute

# Watch for scaling
kubectl get pods -n ml-platform -l app=scheduled-training --watch
```

## Phase 6: Pod Disruption Budgets and Safety (30 minutes)

### Step 6.1: Understand Pod Disruption Budgets

PDBs ensure minimum availability during voluntary disruptions:
- Node drains
- Cluster autoscaler scale-down
- VPA evictions

### Step 6.2: Deploy PDBs

```bash
# Already applied if you ran the setup script
kubectl get pdb -n ml-platform

# Examine PDB configuration
kubectl describe pdb fraud-detector-pdb -n ml-platform
```

**Configuration:**
```yaml
spec:
  minAvailable: 1  # At least 1 pod must be available
  selector:
    matchLabels:
      app: fraud-detector
```

### Step 6.3: Test PDB Protection

```bash
# Try to evict a pod
kubectl get pods -n ml-platform -l app=fraud-detector
kubectl delete pod <pod-name> -n ml-platform

# Check PDB status
kubectl get pdb fraud-detector-pdb -n ml-platform -o jsonpath='{.status}'
```

**Expected Output:**
```json
{
  "currentHealthy": 2,
  "desiredHealthy": 1,
  "disruptionsAllowed": 1,  # Can evict 1 pod safely
  "expectedPods": 2
}
```

### Step 6.4: Deploy Priority Classes

```bash
# Priority classes are already applied
kubectl get priorityclass | grep ml-

# Check pod priorities
kubectl get pods -n ml-platform -o custom-columns=\
NAME:.metadata.name,\
PRIORITY:.spec.priority,\
PRIORITY_CLASS:.spec.priorityClassName
```

## Phase 7: Testing and Validation (30 minutes)

### Step 7.1: Run Comprehensive Test Suite

```bash
# Run all tests
./scripts/test_autoscaling.sh

# Expected output:
# ✅ Prerequisites check
# ✅ HPA CPU-based scaling
# ✅ HPA custom metrics
# ✅ VPA recommendations
# ✅ KEDA scaling
# ✅ Pod Disruption Budgets
# ✅ Resource utilization
```

### Step 7.2: Validate HPA Scaling

```bash
# Test 1: Scale up
./scripts/load_generator.sh fraud-detector-api 1000 10

# Verify scale-up occurred
kubectl get hpa fraud-detector-hpa -n ml-platform

# Test 2: Scale down
# Stop load and wait 5 minutes
kubectl delete pods -n ml-platform -l load-test=true

# Verify scale-down after stabilization window
watch kubectl get hpa,pods -n ml-platform
```

### Step 7.3: Validate VPA Recommendations

```bash
# Check VPA has recommendations (needs 10+ minutes of data)
kubectl get vpa -n ml-platform

# Verify recommendations are reasonable
kubectl describe vpa model-training-vpa -n ml-platform | grep -A 15 "Recommendation:"

# Compare to actual usage
kubectl top pods -n ml-platform -l app=model-training
```

### Step 7.4: Validate KEDA Scale-to-Zero

```bash
# Verify batch processor is at 0 replicas
kubectl get deployment batch-inference-processor -n ml-platform

# Verify scheduled training follows cron schedule
kubectl get scaledobject scheduled-training-scaler -n ml-platform -o yaml | yq '.spec.triggers'
```

## Phase 8: Monitoring and Observability (20 minutes)

### Step 8.1: Monitor Autoscaling Metrics

```bash
# Watch all autoscaling resources
watch 'kubectl get hpa,vpa,scaledobject -n ml-platform'

# View resource utilization
kubectl top pods -n ml-platform --sort-by=cpu
kubectl top pods -n ml-platform --sort-by=memory
```

### Step 8.2: Check Autoscaling Events

```bash
# View all events
kubectl get events -n ml-platform --sort-by='.lastTimestamp'

# Filter HPA events
kubectl get events -n ml-platform --field-selector reason=ScalingReplicaSet

# Filter VPA events
kubectl get events -n ml-platform | grep vpa
```

### Step 8.3: Access Prometheus

```bash
# Port-forward Prometheus
kubectl port-forward -n monitoring svc/prometheus-server 9090:80

# Open browser: http://localhost:9090

# Useful queries:
# - kube_horizontalpodautoscaler_status_current_replicas
# - kube_horizontalpodautoscaler_status_desired_replicas
# - kube_pod_container_resource_requests
# - container_cpu_usage_seconds_total
```

## Phase 9: Cost Optimization (15 minutes)

### Step 9.1: Calculate Cost Savings

**Example Calculation:**

**Without Autoscaling:**
- 10 CPU nodes (always on): $1,000/month
- 5 GPU nodes (always on): $5,000/month
- **Total: $6,000/month**

**With Autoscaling:**
- HPA: Reduce serving API nodes by 40% (off-peak)
- KEDA: Scale batch jobs to zero (80% reduction)
- Cron: Training only during business hours (67% reduction)
- VPA: Right-size resources (20-30% savings)

**Result:**
- CPU nodes: 4 average (60% reduction): $400/month
- GPU nodes: 1 average (80% reduction): $1,000/month
- **Total: $1,400/month**
- **Monthly Savings: $4,600 (77%)**
- **Annual Savings: $55,200**

### Step 9.2: Monitor Resource Efficiency

```bash
# Check resource requests vs usage
kubectl get pods -n ml-platform -o custom-columns=\
NAME:.metadata.name,\
CPU_REQ:.spec.containers[0].resources.requests.cpu,\
MEM_REQ:.spec.containers[0].resources.requests.memory

kubectl top pods -n ml-platform

# Calculate efficiency
# Efficiency = (Actual Usage) / (Requested Resources) * 100%
```

## Troubleshooting Guide

### Issue 1: HPA shows `<unknown>` for metrics

**Symptoms:**
```
NAME                  REFERENCE                       TARGETS         MINPODS   MAXPODS
fraud-detector-hpa    Deployment/fraud-detector-api   <unknown>/50%   2         10
```

**Diagnosis:**
```bash
# Check metrics-server
kubectl get pods -n kube-system -l k8s-app=metrics-server
kubectl logs -n kube-system -l k8s-app=metrics-server

# Check if metrics are available
kubectl top nodes
kubectl top pods -n ml-platform
```

**Solution:**
```bash
# For local clusters, disable TLS verification
kubectl patch deployment metrics-server -n kube-system --type='json' \
  -p='[{"op": "add", "path": "/spec/template/spec/containers/0/args/-", "value": "--kubelet-insecure-tls"}]'

# Restart metrics-server
kubectl rollout restart deployment metrics-server -n kube-system

# Wait for metrics to be available
sleep 60
kubectl top nodes
```

### Issue 2: VPA not providing recommendations

**Symptoms:**
```
No recommendation available
```

**Diagnosis:**
```bash
# Check VPA components
kubectl get pods -n kube-system | grep vpa

# Check VPA logs
kubectl logs -n kube-system -l app=vpa-recommender
```

**Solution:**
- VPA needs 5-10 minutes of data
- Wait at least 10 minutes after pod creation
- Ensure metrics-server is working
- Check VPA updateMode is not "Off"

### Issue 3: KEDA not scaling

**Symptoms:**
ScaledObject exists but replicas stay at 0

**Diagnosis:**
```bash
# Check KEDA operator logs
kubectl logs -n keda -l app=keda-operator

# Check ScaledObject status
kubectl describe scaledobject batch-inference-scaler -n ml-platform
```

**Solution:**
- Verify trigger metric is available
- Check Prometheus connectivity
- Ensure metric returns valid numeric values
- Verify activationThreshold is set correctly

### Issue 4: Scaling thrashing (constant up/down)

**Symptoms:**
Replicas constantly changing every few minutes

**Diagnosis:**
```bash
# Check HPA events
kubectl describe hpa fraud-detector-hpa -n ml-platform | tail -n 50
```

**Solution:**
Increase stabilization window:
```yaml
behavior:
  scaleDown:
    stabilizationWindowSeconds: 600  # Increase to 10 minutes
```

## Best Practices Summary

### HPA
✅ Set minReplicas ≥ 2 for high availability
✅ Use conservative scale-down (5-10 min stabilization)
✅ Start with CPU/memory, add custom metrics gradually
✅ Configure PodDisruptionBudgets
✅ Set appropriate resource requests (HPA baseline)

### VPA
✅ Start with updateMode="Off" for monitoring
✅ Set min/max bounds with resourcePolicy
✅ Wait 24-48 hours for accurate recommendations
✅ Use controlledValues=RequestsOnly (don't modify limits)
✅ Don't combine with HPA on same deployment

### KEDA
✅ Use scale-to-zero for batch workloads
✅ Set minReplicas=1 for user-facing APIs (avoid cold start)
✅ Configure appropriate cooldownPeriod (5-10 minutes)
✅ Test trigger metrics before deploying
✅ Use cron scalers for predictable workloads

### PDBs and Priority Classes
✅ Set PDB for all user-facing services
✅ Use percentage-based PDBs for flexibility
✅ Assign priority to all workloads
✅ Reserve highest priority for critical services
✅ Use low priority for preemptible workloads

## Next Steps

1. **Deploy to Production Cluster**: Apply to EKS/GKE/AKS with Cluster Autoscaler
2. **Add Custom Metrics**: Instrument ML applications with Prometheus
3. **Implement Grafana Dashboards**: Visualize autoscaling metrics
4. **Cost Analysis**: Calculate actual savings for your workloads
5. **Predictive Scaling**: Implement time-series forecasting
6. **Multi-Cluster**: Extend to multi-region autoscaling

## Additional Resources

- [Kubernetes HPA Documentation](https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/)
- [VPA GitHub Repository](https://github.com/kubernetes/autoscaler/tree/master/vertical-pod-autoscaler)
- [KEDA Documentation](https://keda.sh/docs/)
- [Prometheus Adapter](https://github.com/kubernetes-sigs/prometheus-adapter)
- [Cluster Autoscaler FAQ](https://github.com/kubernetes/autoscaler/blob/master/cluster-autoscaler/FAQ.md)

## Completion Checklist

- [ ] Metrics-server installed and providing metrics
- [ ] HPA successfully scales based on CPU load
- [ ] HPA configured with custom Prometheus metrics
- [ ] VPA provides resource recommendations
- [ ] KEDA installed and scaling based on events/schedule
- [ ] PodDisruptionBudgets prevent unsafe evictions
- [ ] Load testing demonstrates scale-up and scale-down
- [ ] Documentation explains when to use each autoscaler
- [ ] Monitoring dashboard shows autoscaling metrics
- [ ] Scale-down behavior is stable (no thrashing)

**Congratulations!** You have successfully implemented Kubernetes autoscaling for ML workloads.
