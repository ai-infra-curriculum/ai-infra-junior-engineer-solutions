# Step-by-Step Implementation Guide: Debugging Kubernetes

## Overview

Master Kubernetes troubleshooting! Learn systematic debugging approaches, common failure patterns, diagnostic commands, and resolution strategies for production ML workloads.

**Time**: 2-3 hours | **Difficulty**: Intermediate to Advanced

---

## Learning Objectives

✅ Debug common Pod failures (ImagePullBackOff, CrashLoopBackOff, OOMKilled)
✅ Troubleshoot networking and service connectivity issues
✅ Diagnose resource constraints and scheduling problems
✅ Use kubectl debugging commands effectively
✅ Read and interpret Kubernetes events
✅ Debug ConfigMap and Secret issues
✅ Troubleshoot health check failures
✅ Use ephemeral debug containers
✅ Implement systematic debugging workflows

---

## Debugging Toolkit

```bash
# Essential debugging commands
kubectl get <resource>              # List resources
kubectl describe <resource> <name>  # Detailed resource info
kubectl logs <pod>                  # View container logs
kubectl logs <pod> --previous       # Logs from crashed container
kubectl exec -it <pod> -- /bin/sh   # Shell into container
kubectl port-forward <pod> <port>   # Forward port to local
kubectl debug <pod>                 # Create debug container
kubectl top <resource>              # Resource usage
kubectl events                      # View recent events
```

---

## Phase 1: ImagePullBackOff

### Scenario

```yaml
# scenarios/01-image-pull-error.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: broken-image
spec:
  replicas: 1
  selector:
    matchLabels:
      app: broken
  template:
    metadata:
      labels:
        app: broken
    spec:
      containers:
      - name: app
        image: nonexistent/fake-image:v1.0.0
        ports:
        - containerPort: 8080
```

### Symptoms

```bash
$ kubectl get pods
NAME                           READY   STATUS             RESTARTS   AGE
broken-image-5d6f8b4c7-xk9m2   0/1     ImagePullBackOff   0          2m
```

### Diagnosis

```bash
# 1. Describe pod
kubectl describe pod broken-image-5d6f8b4c7-xk9m2

# Look for:
Events:
  Warning  Failed     5m    kubelet  Failed to pull image "nonexistent/fake-image:v1.0.0": rpc error: code = Unknown desc = Error response from daemon: pull access denied
  Warning  Failed     5m    kubelet  Error: ErrImagePull
  Normal   BackOff    2m    kubelet  Back-off pulling image "nonexistent/fake-image:v1.0.0"
  Warning  Failed     2m    kubelet  Error: ImagePullBackOff

# 2. Check image name
kubectl get deployment broken-image -o jsonpath='{.spec.template.spec.containers[0].image}'

# 3. Check image pull secrets
kubectl get deployment broken-image -o jsonpath='{.spec.template.spec.imagePullSecrets}'
```

### Resolution

```bash
# Fix 1: Correct image name
kubectl set image deployment/broken-image app=nginx:1.21

# Fix 2: Add image pull secret (for private registries)
kubectl create secret docker-registry regcred \
  --docker-server=registry.example.com \
  --docker-username=user \
  --docker-password=password \
  --docker-email=user@example.com

kubectl patch deployment broken-image -p '
spec:
  template:
    spec:
      imagePullSecrets:
      - name: regcred'

# Verify fix
kubectl get pods -w
```

---

## Phase 2: CrashLoopBackOff

### Scenario

```yaml
# scenarios/02-crashloop-backoff.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: crasher
spec:
  replicas: 1
  selector:
    matchLabels:
      app: crasher
  template:
    metadata:
      labels:
        app: crasher
    spec:
      containers:
      - name: app
        image: python:3.9-slim
        command: ["python", "-c"]
        args: ["import sys; print('Starting...'); sys.exit(1)"]
        # Application exits immediately with error
```

### Symptoms

```bash
$ kubectl get pods
NAME                      READY   STATUS             RESTARTS      AGE
crasher-7d4f5b8c-mk2n3    0/1     CrashLoopBackOff   5 (30s ago)   5m
```

### Diagnosis

```bash
# 1. Check logs from current container
kubectl logs crasher-7d4f5b8c-mk2n3

# 2. Check logs from previous (crashed) container
kubectl logs crasher-7d4f5b8c-mk2n3 --previous

# Output:
Starting...
# Then exits

# 3. Describe pod for exit codes
kubectl describe pod crasher-7d4f5b8c-mk2n3

# Look for:
Last State:     Terminated
  Reason:       Error
  Exit Code:    1
  Started:      Mon, 01 Jan 2024 12:00:00 +0000
  Finished:     Mon, 01 Jan 2024 12:00:01 +0000

# 4. Check restart count
kubectl get pod crasher-7d4f5b8c-mk2n3 -o jsonpath='{.status.containerStatuses[0].restartCount}'
```

### Common Causes

**Exit Code 1**: General application error
**Exit Code 137**: OOMKilled (out of memory)
**Exit Code 143**: SIGTERM (graceful shutdown)
**Exit Code 255**: Exit status out of range

### Resolution

```bash
# Fix: Correct application code
kubectl set image deployment/crasher app=nginx:1.21

# Or edit deployment directly
kubectl edit deployment crasher

# For persistent app: Add restart policy
apiVersion: v1
kind: Pod
spec:
  restartPolicy: OnFailure  # or Never

# Verify
kubectl get pods -w
```

---

## Phase 3: Resource Constraints (OOMKilled)

### Scenario

```yaml
# scenarios/03-resource-constraints.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: memory-hog
spec:
  replicas: 1
  selector:
    matchLabels:
      app: memory-hog
  template:
    metadata:
      labels:
        app: memory-hog
    spec:
      containers:
      - name: app
        image: polinux/stress
        command: ["stress"]
        args:
        - "--vm"
        - "1"
        - "--vm-bytes"
        - "256M"  # Exceeds limit
        - "--vm-hang"
        - "1"
        resources:
          limits:
            memory: "128Mi"
          requests:
            memory: "64Mi"
```

### Symptoms

```bash
$ kubectl get pods
NAME                          READY   STATUS      RESTARTS      AGE
memory-hog-5c8d7f9b-xm4k2     0/1     OOMKilled   5 (1m ago)    3m
```

### Diagnosis

```bash
# 1. Check exit code
kubectl describe pod memory-hog-5c8d7f9b-xm4k2

# Look for:
Last State:     Terminated
  Reason:       OOMKilled
  Exit Code:    137

# 2. Check resource limits
kubectl get pod memory-hog-5c8d7f9b-xm4k2 -o jsonpath='{.spec.containers[0].resources}'

# 3. Check actual usage (if still running)
kubectl top pod memory-hog-5c8d7f9b-xm4k2

# 4. Check node resources
kubectl top nodes
kubectl describe node <node-name>
```

### Resolution

```bash
# Increase memory limit
kubectl patch deployment memory-hog -p '
spec:
  template:
    spec:
      containers:
      - name: app
        resources:
          limits:
            memory: "512Mi"
          requests:
            memory: "256Mi"'

# Or optimize application
# - Reduce batch sizes
# - Implement memory-efficient algorithms
# - Use streaming instead of loading all data

# For ML models: Use model quantization
# - Convert FP32 to FP16 or INT8
# - Use ONNX Runtime for optimization
# - Implement model sharding
```

---

## Phase 4: Service Connectivity Issues

### Scenario

```yaml
# scenarios/04-service-connectivity.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: backend
spec:
  replicas: 2
  selector:
    matchLabels:
      app: backend
      tier: api  # Mismatch!
  template:
    metadata:
      labels:
        app: backend
        # Missing tier: api label
    spec:
      containers:
      - name: api
        image: nginx:1.21
---
apiVersion: v1
kind: Service
metadata:
  name: backend-service
spec:
  selector:
    app: backend
    tier: api  # Won't match pods!
  ports:
  - port: 80
    targetPort: 80
```

### Symptoms

```bash
# Service exists but no endpoints
$ kubectl get svc backend-service
NAME              TYPE        CLUSTER-IP      EXTERNAL-IP   PORT(S)   AGE
backend-service   ClusterIP   10.96.100.50    <none>        80/TCP    5m

$ kubectl get endpoints backend-service
NAME              ENDPOINTS   AGE
backend-service   <none>      5m
```

### Diagnosis

```bash
# 1. Check service selector
kubectl get svc backend-service -o jsonpath='{.spec.selector}'

# Output: {"app":"backend","tier":"api"}

# 2. Check pod labels
kubectl get pods -l app=backend -o jsonpath='{.items[*].metadata.labels}'

# Output: {"app":"backend"} - Missing tier:api!

# 3. Debug with ephemeral container
kubectl run debug --rm -it --image=curlimages/curl --restart=Never -- \
  curl http://backend-service

# 4. Check endpoints
kubectl describe endpoints backend-service

# 5. Check network policies
kubectl get networkpolicies
```

### Resolution

```bash
# Fix 1: Update pod labels
kubectl patch deployment backend -p '
spec:
  template:
    metadata:
      labels:
        app: backend
        tier: api'

# Fix 2: Update service selector
kubectl patch service backend-service -p '
spec:
  selector:
    app: backend'

# Verify endpoints
kubectl get endpoints backend-service -w
```

---

## Phase 5: ConfigMap/Secret Issues

### Scenario

```yaml
# scenarios/05-config-issues.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: config-app
spec:
  replicas: 1
  selector:
    matchLabels:
      app: config-app
  template:
    metadata:
      labels:
        app: config-app
    spec:
      containers:
      - name: app
        image: nginx:1.21
        envFrom:
        - configMapRef:
            name: app-config  # Doesn't exist!
```

### Symptoms

```bash
$ kubectl get pods
NAME                         READY   STATUS                 RESTARTS   AGE
config-app-7c8d4f9b-k2m3n    0/1     CreateContainerError   0          1m
```

### Diagnosis

```bash
# 1. Describe pod
kubectl describe pod config-app-7c8d4f9b-k2m3n

# Look for:
Events:
  Warning  Failed  1m  kubelet  Error: configmap "app-config" not found

# 2. List existing ConfigMaps
kubectl get configmaps

# 3. Check volume mounts (if using volumeMounts)
kubectl get deployment config-app -o jsonpath='{.spec.template.spec.volumes}'
```

### Resolution

```bash
# Create missing ConfigMap
kubectl create configmap app-config \
  --from-literal=ENV=production \
  --from-literal=LOG_LEVEL=INFO

# Or from file
kubectl create configmap app-config \
  --from-file=config.yaml

# Verify
kubectl get configmap app-config -o yaml

# Pod will automatically restart
kubectl get pods -w
```

---

## Phase 6: Liveness/Readiness Probe Failures

### Scenario

```yaml
# scenarios/06-liveness-readiness.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: probe-test
spec:
  replicas: 1
  selector:
    matchLabels:
      app: probe-test
  template:
    metadata:
      labels:
        app: probe-test
    spec:
      containers:
      - name: app
        image: nginx:1.21
        livenessProbe:
          httpGet:
            path: /healthz  # Wrong path!
            port: 80
          initialDelaySeconds: 5
          periodSeconds: 5
          failureThreshold: 2
        readinessProbe:
          httpGet:
            path: /ready  # Wrong path!
            port: 80
          initialDelaySeconds: 3
          periodSeconds: 3
          failureThreshold: 2
```

### Symptoms

```bash
$ kubectl get pods
NAME                         READY   STATUS    RESTARTS      AGE
probe-test-7d4f5b8c-m9k2n    0/1     Running   5 (30s ago)   2m
```

### Diagnosis

```bash
# 1. Check events
kubectl describe pod probe-test-7d4f5b8c-m9k2n

# Look for:
Events:
  Warning  Unhealthy  1m  kubelet  Liveness probe failed: HTTP probe failed with statuscode: 404
  Warning  Unhealthy  1m  kubelet  Readiness probe failed: HTTP probe failed with statuscode: 404

# 2. Test endpoint manually
kubectl exec probe-test-7d4f5b8c-m9k2n -- curl -I http://localhost/healthz

# Output: HTTP/1.1 404 Not Found

# 3. Check what endpoints exist
kubectl exec probe-test-7d4f5b8c-m9k2n -- curl -I http://localhost/
```

### Resolution

```bash
# Fix 1: Update probe paths
kubectl patch deployment probe-test -p '
spec:
  template:
    spec:
      containers:
      - name: app
        livenessProbe:
          httpGet:
            path: /
            port: 80
        readinessProbe:
          httpGet:
            path: /
            port: 80'

# Fix 2: Increase thresholds (if app is slow to start)
kubectl patch deployment probe-test -p '
spec:
  template:
    spec:
      containers:
      - name: app
        livenessProbe:
          initialDelaySeconds: 30
          failureThreshold: 5'

# For ML models: Use startup probe
startupProbe:
  httpGet:
    path: /health
    port: 8080
  initialDelaySeconds: 0
  periodSeconds: 10
  failureThreshold: 30  # Allow 5 min for model loading
```

---

## Phase 7: Debugging Workflows

### Systematic Debugging Process

```bash
#!/bin/bash
# debug-workflow.sh

POD_NAME=$1

echo "=== DEBUGGING POD: $POD_NAME ==="

# 1. Check pod status
echo -e "\n1. Pod Status:"
kubectl get pod $POD_NAME -o wide

# 2. Check events
echo -e "\n2. Recent Events:"
kubectl get events --field-selector involvedObject.name=$POD_NAME \
  --sort-by='.lastTimestamp'

# 3. Check logs
echo -e "\n3. Container Logs:"
kubectl logs $POD_NAME --tail=50

# 4. Check previous logs (if crashed)
echo -e "\n4. Previous Container Logs:"
kubectl logs $POD_NAME --previous --tail=50 2>/dev/null || echo "No previous logs"

# 5. Describe pod
echo -e "\n5. Pod Details:"
kubectl describe pod $POD_NAME

# 6. Check resource usage
echo -e "\n6. Resource Usage:"
kubectl top pod $POD_NAME 2>/dev/null || echo "Metrics not available"

# 7. Check node
echo -e "\n7. Node Info:"
NODE=$(kubectl get pod $POD_NAME -o jsonpath='{.spec.nodeName}')
echo "Running on node: $NODE"
kubectl describe node $NODE | grep -A 5 "Allocated resources"
```

### Usage

```bash
chmod +x debug-workflow.sh
./debug-workflow.sh <pod-name>
```

---

## Phase 8: Advanced Debugging

### Ephemeral Debug Containers

```bash
# Add debug container to running pod
kubectl debug -it probe-test-7d4f5b8c-m9k2n \
  --image=busybox:1.28 \
  --target=app

# Debug with different image
kubectl debug -it probe-test-7d4f5b8c-m9k2n \
  --image=nicolaka/netshoot \
  --target=app

# Create copy of pod for debugging
kubectl debug probe-test-7d4f5b8c-m9k2n \
  --copy-to=probe-test-debug \
  --container=app
```

### Network Debugging

```bash
# Test DNS resolution
kubectl run -it --rm debug --image=busybox:1.28 --restart=Never -- \
  nslookup backend-service

# Test connectivity
kubectl run -it --rm debug --image=curlimages/curl --restart=Never -- \
  curl -v http://backend-service

# Advanced network tools
kubectl run -it --rm netshoot --image=nicolaka/netshoot --restart=Never -- bash

# Inside netshoot:
nslookup backend-service
curl http://backend-service
ping backend-service
traceroute backend-service
```

### Performance Debugging

```bash
# CPU/Memory profiling
kubectl top pods --containers

# Get resource usage over time
watch -n 5 kubectl top pod <pod-name>

# Check node pressure
kubectl describe nodes | grep -A 5 Conditions

# Check for evicted pods
kubectl get pods --all-namespaces --field-selector=status.phase=Failed
```

---

## Phase 9: Debugging Checklist

### Pod Won't Start

```
☐ Check image name and tag
☐ Verify image pull secrets
☐ Check container logs
☐ Inspect pod events
☐ Verify ConfigMaps/Secrets exist
☐ Check resource requests vs node capacity
☐ Verify node selectors/affinity rules
☐ Check security context settings
```

### Pod Crashes

```
☐ Check logs (current + previous)
☐ Verify exit code
☐ Check resource limits
☐ Review application code
☐ Verify dependencies (DB, cache, etc.)
☐ Check environment variables
☐ Review health check configuration
☐ Check for OOMKilled events
```

### Service Unreachable

```
☐ Verify service exists
☐ Check service selector matches pod labels
☐ Verify endpoints are populated
☐ Test pod connectivity directly
☐ Check network policies
☐ Verify ingress configuration
☐ Check DNS resolution
☐ Review firewall rules
```

---

## Common Error Messages

| Error | Cause | Solution |
|-------|-------|----------|
| ImagePullBackOff | Image doesn't exist | Fix image name/tag |
| CrashLoopBackOff | Container exits on start | Fix application code |
| OOMKilled | Out of memory | Increase limits |
| CreateContainerConfigError | Missing ConfigMap/Secret | Create resource |
| RunContainerError | Invalid configuration | Fix container spec |
| InvalidImageName | Malformed image name | Use valid format |
| Pending | No node can schedule | Check resources/affinity |
| Evicted | Node pressure | Increase node capacity |

---

## Best Practices

✅ Always check pod events first
✅ Review logs from both current and previous containers
✅ Use describe command for detailed diagnostics
✅ Test connectivity with ephemeral containers
✅ Monitor resource usage proactively
✅ Implement proper health checks
✅ Set appropriate resource limits
✅ Use systematic debugging workflows
✅ Keep debugging tools image handy (busybox, curl, netshoot)
✅ Document common issues and solutions

---

**Kubernetes debugging mastered!** 🔍

**Next Exercise**: StatefulSets and Storage
