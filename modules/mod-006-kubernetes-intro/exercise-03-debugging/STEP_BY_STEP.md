# Step-by-Step Debugging Guide

This guide walks you through debugging each scenario systematically, teaching you the thought process and commands to use when troubleshooting Kubernetes applications.

## Table of Contents

1. [Setup](#setup)
2. [Scenario 01: Image Pull Error](#scenario-01-image-pull-error)
3. [Scenario 02: CrashLoopBackOff](#scenario-02-crashloopbackoff)
4. [Scenario 03: Resource Constraints](#scenario-03-resource-constraints)
5. [Scenario 04: Service Connectivity](#scenario-04-service-connectivity)
6. [Scenario 05: Configuration Issues](#scenario-05-configuration-issues)
7. [Scenario 06: Liveness and Readiness Probes](#scenario-06-liveness-and-readiness-probes)
8. [General Debugging Methodology](#general-debugging-methodology)

---

## Setup

### Prerequisites Check

```bash
# Verify kubectl is installed
kubectl version --client

# Verify cluster connectivity
kubectl cluster-info

# Verify you can create resources
kubectl auth can-i create deployments --all-namespaces
```

### Deploy All Scenarios

```bash
cd scripts
./deploy-scenarios.sh all

# Verify all namespaces created
kubectl get namespaces | grep debug-scenario
```

---

## Scenario 01: Image Pull Error

### Context

**Problem**: Deployment has typo in image name (`ngnix` instead of `nginx`)

**Learning Goal**: Identify and fix image pull errors

### Step 1: Observe the Problem

```bash
# Check pod status
kubectl get pods -n debug-scenario-01

# Expected output:
# NAME                            READY   STATUS             RESTARTS   AGE
# broken-nginx-xxx-yyy            0/1     ImagePullBackOff   0          1m
```

**What you see**: Pod stuck in `ImagePullBackOff` or `ErrImagePull` status

### Step 2: Gather Information

```bash
# Describe the pod to see events
POD_NAME=$(kubectl get pods -n debug-scenario-01 -o jsonpath='{.items[0].metadata.name}')
kubectl describe pod $POD_NAME -n debug-scenario-01

# Look for these sections in output:
# - Events: Will show "Failed to pull image" or "ErrImagePull"
# - Image: Shows what image it's trying to pull
```

**Key Event Messages**:
```
Failed to pull image "ngnix:1.21-alpine": rpc error: code = Unknown desc = Error response from daemon: pull access denied for ngnix
```

### Step 3: Identify the Issue

```bash
# Check the exact image specified
kubectl get deployment broken-nginx -n debug-scenario-01 -o jsonpath='{.spec.template.spec.containers[0].image}'

# Output: ngnix:1.21-alpine
# The typo: "ngnix" should be "nginx"
```

### Step 4: Fix the Issue

**Method 1: Using kubectl set image (Recommended)**
```bash
kubectl set image deployment/broken-nginx \
  nginx=nginx:1.21-alpine \
  -n debug-scenario-01
```

**Method 2: Using kubectl edit**
```bash
kubectl edit deployment broken-nginx -n debug-scenario-01
# Change "ngnix" to "nginx" in the editor, save and exit
```

**Method 3: Using kubectl patch**
```bash
kubectl patch deployment broken-nginx -n debug-scenario-01 --type=json -p='[
  {
    "op": "replace",
    "path": "/spec/template/spec/containers/0/image",
    "value": "nginx:1.21-alpine"
  }
]'
```

### Step 5: Verify the Fix

```bash
# Watch pod status change
kubectl get pods -n debug-scenario-01 -w

# Check rollout status
kubectl rollout status deployment/broken-nginx -n debug-scenario-01

# Verify pods are running
kubectl get pods -n debug-scenario-01

# Expected output:
# NAME                            READY   STATUS    RESTARTS   AGE
# broken-nginx-xxx-yyy            1/1     Running   0          30s
```

### Step 6: Additional Verification

```bash
# Check recent events (should be clean)
kubectl get events -n debug-scenario-01 --field-selector type=Normal --sort-by='.lastTimestamp' | tail -5

# Test the deployment
kubectl run test-pod --rm -it --image=curlimages/curl -n debug-scenario-01 -- \
  curl http://broken-nginx
```

### Key Takeaways

✅ `ImagePullBackOff` means Kubernetes can't pull the image
✅ `kubectl describe pod` shows the exact error
✅ Common causes: typos, wrong tags, missing authentication
✅ Fix with `kubectl set image` for immediate update

---

## Scenario 02: CrashLoopBackOff

### Context

**Problem**: Application crashes due to malformed JSON in ConfigMap

**Learning Goal**: Debug application crashes and configuration issues

### Step 1: Observe the Problem

```bash
# Check pod status
kubectl get pods -n debug-scenario-02

# Expected output:
# NAME                              READY   STATUS             RESTARTS   AGE
# crashloop-app-xxx-yyy             0/1     CrashLoopBackOff   5          5m
```

**What you see**: Pod in `CrashLoopBackOff` with increasing restart count

### Step 2: Gather Information

```bash
# Try to view current logs
POD_NAME=$(kubectl get pods -n debug-scenario-02 -o jsonpath='{.items[0].metadata.name}')
kubectl logs $POD_NAME -n debug-scenario-02

# If pod crashed, you'll see:
# Error from server (BadRequest): previous terminated container "app" in pod "..." not found

# View PREVIOUS container logs (this is critical!)
kubectl logs $POD_NAME -n debug-scenario-02 --previous

# Expected output shows:
# Starting application...
# {
#   "database": {
#     "host": "localhost,
#     "port": 5432
#   }
# Validating JSON...
# ERROR: Invalid JSON configuration!
```

### Step 3: Identify the Issue

```bash
# Check the ConfigMap content
kubectl get configmap broken-app-config -n debug-scenario-02 -o yaml

# You'll see malformed JSON:
# config.json: |
#   {
#     "database": {
#       "host": "localhost,
#       "port": 5432
#     }
```

**The problems**:
1. Missing closing quote after "localhost"
2. Missing closing brace for database object
3. Missing closing brace for root object

### Step 4: Fix the Issue

```bash
# Edit the ConfigMap
kubectl edit configmap broken-app-config -n debug-scenario-02

# Change this:
# config.json: |
#   {
#     "database": {
#       "host": "localhost,
#       "port": 5432
#     }

# To this:
# config.json: |
#   {
#     "database": {
#       "host": "localhost",
#       "port": 5432
#     }
#   }
```

### Step 5: Restart the Deployment

```bash
# ConfigMap changes don't automatically restart pods
# Force a rollout restart
kubectl rollout restart deployment/crashloop-app -n debug-scenario-02

# Watch the pods restart
kubectl get pods -n debug-scenario-02 -w
```

### Step 6: Verify the Fix

```bash
# Check pod is now running
kubectl get pods -n debug-scenario-02

# View logs to confirm success
kubectl logs -n debug-scenario-02 $(kubectl get pods -n debug-scenario-02 -o jsonpath='{.items[0].metadata.name}')

# Expected output:
# Starting application...
# {
#   "database": {
#     "host": "localhost",
#     "port": 5432
#   }
# }
# Validating JSON...
# Application running...

# Check restart count is 0
kubectl get pods -n debug-scenario-02
```

### Key Takeaways

✅ `CrashLoopBackOff` means container keeps exiting
✅ **Always use `--previous`** to see logs from crashed container
✅ Check exit code: `kubectl get pod <name> -o jsonpath='{.status.containerStatuses[0].lastState.terminated.exitCode}'`
✅ ConfigMap/Secret changes require pod restart

---

## Scenario 03: Resource Constraints

### Context

**Problem**: Three different resource-related issues

**Learning Goal**: Debug resource requests, limits, and OOMKilled pods

### Step 1: Observe the Problems

```bash
# Check all pods in namespace
kubectl get pods -n debug-scenario-03 -o wide

# Expected output:
# NAME                                READY   STATUS    RESTARTS
# resource-hungry-app-xxx             0/1     Pending   0
# no-resources-app-xxx                1/1     Running   0
# oom-app-xxx                         0/1     OOMKilled 3
```

### Step 2: Debug Pending Pod (resource-hungry-app)

```bash
# Describe the pending pod
POD_NAME=$(kubectl get pods -n debug-scenario-03 -l app=resource-hungry -o jsonpath='{.items[0].metadata.name}')
kubectl describe pod $POD_NAME -n debug-scenario-03

# Look for Events section:
# Warning  FailedScheduling  ... 0/1 nodes are available: 1 Insufficient memory
```

**Issue**: Requesting 8Gi memory, cluster doesn't have enough

```bash
# Check node capacity
kubectl top nodes
kubectl describe nodes | grep -A 5 "Allocated resources"

# Check pod resource requests
kubectl get pod $POD_NAME -n debug-scenario-03 -o jsonpath='{.spec.containers[0].resources}'
```

**Fix**:
```bash
# Reduce resource requests to reasonable values
kubectl set resources deployment/resource-hungry-app -n debug-scenario-03 \
  --requests=cpu=100m,memory=128Mi \
  --limits=cpu=200m,memory=256Mi

# Verify pod is now scheduled
kubectl get pods -n debug-scenario-03 -l app=resource-hungry
```

### Step 3: Debug Missing Resources (no-resources-app)

```bash
# Check pod resources
POD_NAME=$(kubectl get pods -n debug-scenario-03 -l app=no-resources -o jsonpath='{.items[0].metadata.name}')
kubectl get pod $POD_NAME -n debug-scenario-03 -o jsonpath='{.spec.containers[0].resources}'

# Output: null (no resources defined!)
```

**Issue**: No resource requests or limits defined (bad practice)

**Fix**:
```bash
# Add resource limits
kubectl set resources deployment/no-resources-app -n debug-scenario-03 \
  --requests=cpu=50m,memory=64Mi \
  --limits=cpu=100m,memory=128Mi
```

### Step 4: Debug OOMKilled Pod (oom-app)

```bash
# Check pod status
kubectl get pods -n debug-scenario-03 -l app=oom-app

# Describe pod
POD_NAME=$(kubectl get pods -n debug-scenario-03 -l app=oom-app -o jsonpath='{.items[0].metadata.name}')
kubectl describe pod $POD_NAME -n debug-scenario-03

# Look for:
# Last State:     Terminated
#   Reason:       OOMKilled
#   Exit Code:    137
```

**Issue**: Application tries to allocate 256Mi but limit is 128Mi

```bash
# Check memory limit
kubectl get deployment oom-app -n debug-scenario-03 -o jsonpath='{.spec.template.spec.containers[0].resources.limits.memory}'
# Output: 128Mi

# Check what app is requesting (in command)
kubectl get deployment oom-app -n debug-scenario-03 -o yaml | grep -A 5 command
# Shows: --vm-bytes 256M
```

**Fix**:
```bash
# Increase memory limit
kubectl set resources deployment/oom-app -n debug-scenario-03 \
  --limits=cpu=200m,memory=512Mi \
  --requests=cpu=100m,memory=256Mi

# Watch pod stabilize
kubectl get pods -n debug-scenario-03 -l app=oom-app -w
```

### Step 5: Verify All Fixes

```bash
# Check all pods are running
kubectl get pods -n debug-scenario-03

# Check resource usage
kubectl top pods -n debug-scenario-03

# Verify no OOMKilled events
kubectl get events -n debug-scenario-03 --field-selector reason=OOMKilling
```

### Key Takeaways

✅ `Pending` + `FailedScheduling` = insufficient resources
✅ Always set resource requests and limits
✅ Exit code 137 = OOMKilled
✅ Use `kubectl top` to monitor actual usage
✅ Requests guarantee resources, limits prevent overuse

---

## Scenario 04: Service Connectivity

### Context

**Problem**: Services with wrong selectors and ports

**Learning Goal**: Debug service discovery and endpoint issues

### Step 1: Observe the Problem

```bash
# Check services and endpoints
kubectl get svc,endpoints -n debug-scenario-04

# Expected output shows:
# backend-service-broken        ClusterIP   ...
# backend-service-wrong-port    ClusterIP   ...
# backend-service-correct       ClusterIP   ...
#
# Endpoints:
# backend-service-broken        <none>
# backend-service-wrong-port    10.x.x.x:8080,10.x.x.x:8080
# backend-service-correct       10.x.x.x:8080,10.x.x.x:8080
```

**Issue**: `backend-service-broken` has NO endpoints!

### Step 2: Debug Service with No Endpoints

```bash
# Check service selector
kubectl get svc backend-service-broken -n debug-scenario-04 -o yaml | grep -A 3 selector

# Output:
# selector:
#   app: backend-api
#   tier: service
```

```bash
# Check pod labels
kubectl get pods -n debug-scenario-04 --show-labels

# Output shows pods have:
# app=backend,tier=api
```

**Problem**: Service selector doesn't match pod labels!
- Service expects: `app=backend-api, tier=service`
- Pods have: `app=backend, tier=api`

**Fix**:
```bash
kubectl patch svc backend-service-broken -n debug-scenario-04 -p '
{
  "spec": {
    "selector": {
      "app": "backend",
      "tier": "api"
    }
  }
}'
```

### Step 3: Debug Wrong Port

```bash
# Test connectivity to wrong-port service
kubectl exec -n debug-scenario-04 client-test-pod -- \
  curl -v http://backend-service-wrong-port

# Output:
# Connection refused
```

```bash
# Check service port configuration
kubectl get svc backend-service-wrong-port -n debug-scenario-04 -o yaml | grep -A 5 ports

# Output:
# ports:
# - port: 80
#   targetPort: 80  # Wrong! Pods listen on 8080
```

**Fix**:
```bash
kubectl patch svc backend-service-wrong-port -n debug-scenario-04 --type=json -p='[
  {
    "op": "replace",
    "path": "/spec/ports/0/targetPort",
    "value": 8080
  }
]'
```

### Step 4: Verify All Services Work

```bash
# Check all services now have endpoints
kubectl get endpoints -n debug-scenario-04

# Test connectivity
kubectl exec -n debug-scenario-04 client-test-pod -- curl http://backend-service-broken
kubectl exec -n debug-scenario-04 client-test-pod -- curl http://backend-service-wrong-port
kubectl exec -n debug-scenario-04 client-test-pod -- curl http://backend-service-correct

# All should return: "Backend API v1.0"
```

### Key Takeaways

✅ Service with no endpoints = selector mismatch
✅ Service selector must **exactly match** pod labels
✅ `targetPort` must match container port
✅ Use `kubectl get endpoints` to check service-pod connections
✅ Test connectivity from inside cluster

---

## Scenario 05: Configuration Issues

### Context

**Problem**: Missing ConfigMap, wrong Secret key, bad volume mounts

**Learning Goal**: Debug configuration injection

### Step 1: Observe the Problems

```bash
# Check pod status
kubectl get pods -n debug-scenario-05

# Expected output:
# NAME                                      READY   STATUS
# missing-configmap-app-xxx                 0/1     CreateContainerConfigError
# missing-secret-key-app-xxx                0/1     CreateContainerConfigError
# wrong-volume-mount-xxx                    0/1     CrashLoopBackOff
# fixed-config-app-xxx                      1/1     Running
```

### Step 2: Debug Missing ConfigMap

```bash
# Describe pod
POD_NAME=$(kubectl get pods -n debug-scenario-05 -l app=missing-configmap -o jsonpath='{.items[0].metadata.name}')
kubectl describe pod $POD_NAME -n debug-scenario-05

# Events show:
# Error: configmap "nonexistent-config" not found
```

```bash
# List ConfigMaps
kubectl get configmap -n debug-scenario-05

# The ConfigMap doesn't exist!
```

**Fix**:
```bash
# Create the missing ConfigMap
kubectl create configmap nonexistent-config -n debug-scenario-05 \
  --from-literal=app.env=production \
  --from-literal=log.level=info \
  --from-literal=feature.flag=enabled

# Pod will automatically recover
kubectl get pods -n debug-scenario-05 -l app=missing-configmap -w
```

### Step 3: Debug Wrong Secret Key

```bash
# Describe pod
POD_NAME=$(kubectl get pods -n debug-scenario-05 -l app=missing-secret-key -o jsonpath='{.items[0].metadata.name}')
kubectl describe pod $POD_NAME -n debug-scenario-05

# Events show:
# Error: key "apikey" not found in secret "app-secrets"
```

```bash
# Check secret keys
kubectl get secret app-secrets -n debug-scenario-05 -o jsonpath='{.data}' | jq 'keys'

# Output: ["api-key", "db-password"]
# The key is "api-key" not "apikey"!
```

**Fix**:
```bash
# Fix the environment variable reference
kubectl set env deployment/missing-secret-key-app -n debug-scenario-05 \
  API_KEY=api-key \
  --from=secret/app-secrets \
  --keys=api-key
```

### Step 4: Debug Volume Mount Issues

```bash
# Check pod logs
POD_NAME=$(kubectl get pods -n debug-scenario-05 -l app=wrong-volume -o jsonpath='{.items[0].metadata.name}')
kubectl logs $POD_NAME -n debug-scenario-05 --previous

# May show permission errors or startup failures
```

```bash
# Check volume mounts
kubectl get deployment wrong-volume-mount -n debug-scenario-05 -o yaml | grep -A 10 volumeMounts

# Shows mounting secret to /bin (system directory!)
```

**Issue**: Mounting volumes to system directories can cause issues

**Fix**:
```bash
# This requires redeployment with correct mount path
# For learning purposes, we'll just document the issue
echo "Volume should be mounted to /etc/secrets, not /bin"
```

### Step 5: Verify Fixed Configuration

```bash
# Check all pods
kubectl get pods -n debug-scenario-05

# Test environment variables in fixed pod
POD_NAME=$(kubectl get pods -n debug-scenario-05 -l app=fixed-config -o jsonpath='{.items[0].metadata.name}')
kubectl exec $POD_NAME -n debug-scenario-05 -- env | grep -E 'API_KEY|DATABASE'

# Check mounted files
kubectl exec $POD_NAME -n debug-scenario-05 -- ls -la /etc/nginx/conf.d/
```

### Key Takeaways

✅ `CreateContainerConfigError` = missing ConfigMap/Secret or wrong key
✅ Check exact key names with `kubectl get configmap/secret -o yaml`
✅ ConfigMap changes don't auto-restart pods
✅ Avoid mounting volumes to system directories
✅ Use `kubectl describe` to see exact error message

---

## Scenario 06: Liveness and Readiness Probes

### Context

**Problem**: Incorrect probe configurations causing issues

**Learning Goal**: Configure and debug health checks

### Step 1: Observe the Problems

```bash
# Check pods - watch for frequent restarts
kubectl get pods -n debug-scenario-06 -w

# You'll see:
# broken-liveness-app-xxx      0/1   Running       3   2m
# slow-startup-app-xxx          0/1   Running       5   2m
# failing-readiness-app-xxx     1/1   Running       0   2m
```

### Step 2: Debug Liveness Probe Failures

```bash
# Check restart count
kubectl get pods -n debug-scenario-06 -l app=broken-liveness

# Describe pod for probe failure events
POD_NAME=$(kubectl get pods -n debug-scenario-06 -l app=broken-liveness -o jsonpath='{.items[0].metadata.name}')
kubectl describe pod $POD_NAME -n debug-scenario-06

# Events show:
# Liveness probe failed: HTTP probe failed with statuscode: 404
```

```bash
# Check probe configuration
kubectl get deployment broken-liveness-app -n debug-scenario-06 -o yaml | grep -A 10 livenessProbe

# Shows:
# livenessProbe:
#   httpGet:
#     path: /healthz  # This endpoint doesn't exist!
#     port: 8080
```

**Fix**:
```bash
# Change probe path to root (which exists)
kubectl patch deployment broken-liveness-app -n debug-scenario-06 --type=json -p='[
  {
    "op": "replace",
    "path": "/spec/template/spec/containers/0/livenessProbe/httpGet/path",
    "value": "/"
  }
]'
```

### Step 3: Debug Slow Startup

```bash
# Check pod events
POD_NAME=$(kubectl get pods -n debug-scenario-06 -l app=slow-startup -o jsonpath='{.items[0].metadata.name}')
kubectl describe pod $POD_NAME -n debug-scenario-06

# Shows:
# Liveness probe failed: cat: can't open '/tmp/health': No such file or directory
# Container is being restarted before app finishes starting!
```

**Issue**: `initialDelaySeconds: 10` but app takes 60 seconds to start

**Fix**:
```bash
# Increase initialDelaySeconds or add startup probe
kubectl patch deployment slow-startup-app -n debug-scenario-06 --type=json -p='[
  {
    "op": "replace",
    "path": "/spec/template/spec/containers/0/livenessProbe/initialDelaySeconds",
    "value": 65
  }
]'
```

### Step 4: Debug Readiness Probe (Service Endpoints)

```bash
# Check service endpoints
kubectl get endpoints failing-readiness-svc -n debug-scenario-06

# Output: <none>
# Even though pods are running, service has no endpoints!
```

```bash
# Check readiness probe configuration
kubectl get deployment failing-readiness-app -n debug-scenario-06 -o yaml | grep -A 10 readinessProbe

# Shows:
# readinessProbe:
#   httpGet:
#     path: /
#     port: 8080  # Wrong! Nginx listens on port 80
```

**Fix**:
```bash
kubectl patch deployment failing-readiness-app -n debug-scenario-06 --type=json -p='[
  {
    "op": "replace",
    "path": "/spec/template/spec/containers/0/readinessProbe/httpGet/port",
    "value": 80
  }
]'

# Wait for pods to become ready
kubectl get pods -n debug-scenario-06 -l app=failing-readiness -w

# Check endpoints are now populated
kubectl get endpoints failing-readiness-svc -n debug-scenario-06
```

### Step 5: Verify All Probes Working

```bash
# Check pod restart counts (should stop increasing)
kubectl get pods -n debug-scenario-06

# Monitor for a few minutes
watch kubectl get pods -n debug-scenario-06

# Check service endpoints
kubectl get endpoints -n debug-scenario-06

# Test service connectivity
kubectl run test-pod --rm -it --image=curlimages/curl -n debug-scenario-06 -- \
  curl http://failing-readiness-svc
```

### Key Takeaways

✅ Liveness probe failures cause pod restarts
✅ Readiness probe failures remove pod from service endpoints
✅ Set `initialDelaySeconds` > app startup time
✅ Use startup probes for slow-starting apps
✅ Probe path and port must match application configuration

---

## General Debugging Methodology

### The 5-Step Debug Process

#### 1. Observe
```bash
kubectl get all -n <namespace>
kubectl get events -n <namespace> --sort-by='.lastTimestamp'
```

#### 2. Describe
```bash
kubectl describe pod <pod-name> -n <namespace>
kubectl describe svc <service-name> -n <namespace>
```

#### 3. Investigate
```bash
kubectl logs <pod-name> -n <namespace>
kubectl logs <pod-name> -n <namespace> --previous
kubectl exec <pod-name> -n <namespace> -- <command>
```

#### 4. Diagnose
- Identify root cause from symptoms
- Check configuration, resources, networking
- Verify labels, selectors, ports

#### 5. Fix and Verify
```bash
kubectl patch/edit/set <resource>
kubectl rollout restart deployment/<name>
kubectl get pods -w
```

### Debugging Checklist

**Pod Issues**:
- [ ] Check pod status and phase
- [ ] Read pod events
- [ ] View current logs
- [ ] View previous logs (if crashed)
- [ ] Check resource requests/limits
- [ ] Verify image name and tag
- [ ] Check probe configuration

**Service Issues**:
- [ ] Check service exists
- [ ] Verify service has endpoints
- [ ] Compare service selector with pod labels
- [ ] Verify port and targetPort
- [ ] Test connectivity from inside cluster

**Configuration Issues**:
- [ ] List ConfigMaps and Secrets
- [ ] Verify ConfigMap/Secret keys
- [ ] Check environment variables
- [ ] Verify volume mounts
- [ ] Test configuration in pod

**Network Issues**:
- [ ] Check DNS resolution
- [ ] Verify NetworkPolicies
- [ ] Test pod-to-pod connectivity
- [ ] Check service endpoints
- [ ] Verify ingress configuration

### Common Error Patterns

| Status | Common Cause | First Check |
|--------|-------------|-------------|
| ImagePullBackOff | Image name typo | `kubectl describe pod` |
| CrashLoopBackOff | App crashes | `kubectl logs --previous` |
| Pending | Resources/scheduling | `kubectl describe pod` events |
| CreateContainerConfigError | Missing config | `kubectl get configmap/secret` |
| RunContainerError | Volume/mount issues | `kubectl describe pod` |
| OOMKilled | Memory limit too low | Check exit code 137 |

---

## Practice Tips

1. **Work through scenarios in order** - They increase in complexity
2. **Try to solve without hints** - Then check the comments
3. **Use the debug-master.sh script** - Learn its output format
4. **Practice kubectl commands** - Build muscle memory
5. **Experiment with breaking things** - Then fix them
6. **Time yourself** - Track improvement in debug speed
7. **Document your process** - Write down what worked

---

## Next Steps

After mastering these scenarios:

1. Create your own debugging scenarios
2. Practice on real applications
3. Learn advanced debugging tools (stern, k9s, kubectl-debug)
4. Study production incident postmortems
5. Continue to Exercise 04: StatefulSets and Storage

---

**Remember**: Debugging is a skill that improves with practice. The more issues you encounter and fix, the faster you'll become at identifying and resolving problems!
