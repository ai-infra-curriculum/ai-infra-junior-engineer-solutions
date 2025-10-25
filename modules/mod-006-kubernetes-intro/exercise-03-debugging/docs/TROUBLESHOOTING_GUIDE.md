# Kubernetes Troubleshooting Guide

## Quick Reference

| Issue Type | Common Symptoms | First Commands to Run |
|-----------|----------------|---------------------|
| Pod not starting | Pending, ImagePullBackOff | `kubectl describe pod`, `kubectl get events` |
| Pod crashing | CrashLoopBackOff, Error | `kubectl logs`, `kubectl logs --previous` |
| Service unreachable | Connection refused | `kubectl get endpoints`, `kubectl describe svc` |
| Resource issues | Pending, OOMKilled | `kubectl top nodes`, `kubectl describe pod` |
| Configuration errors | CreateContainerConfigError | `kubectl get configmap`, `kubectl get secret` |
| Probe failures | Pod restarting frequently | `kubectl describe pod`, check events |

---

## 1. Pod Issues

### 1.1 ImagePullBackOff / ErrImagePull

**Symptoms:**
- Pod status: `ImagePullBackOff` or `ErrImagePull`
- Pod remains in pending state
- Events show image pull errors

**Common Causes:**
1. Image name is misspelled
2. Image tag doesn't exist
3. Private registry authentication missing
4. Network issues preventing image download
5. Registry is unreachable

**Debugging Steps:**

```bash
# Check pod status and events
kubectl describe pod <pod-name> -n <namespace>

# Look for ImagePullBackOff in events
kubectl get events -n <namespace> --field-selector involvedObject.name=<pod-name>

# Verify image name and tag
kubectl get pod <pod-name> -n <namespace> -o jsonpath='{.spec.containers[*].image}'

# Check if image pull secrets are configured
kubectl get pod <pod-name> -n <namespace> -o jsonpath='{.spec.imagePullSecrets}'
```

**Solutions:**

```bash
# Fix image name
kubectl set image deployment/<deployment-name> \
  <container-name>=<correct-image-name> \
  -n <namespace>

# Add image pull secret
kubectl create secret docker-registry regcred \
  --docker-server=<registry-server> \
  --docker-username=<username> \
  --docker-password=<password> \
  -n <namespace>

kubectl patch deployment <deployment-name> -n <namespace> -p '
{
  "spec": {
    "template": {
      "spec": {
        "imagePullSecrets": [{"name": "regcred"}]
      }
    }
  }
}'
```

---

### 1.2 CrashLoopBackOff

**Symptoms:**
- Pod status: `CrashLoopBackOff`
- High restart count
- Container exits shortly after starting

**Common Causes:**
1. Application error/bug causing immediate exit
2. Missing dependencies
3. Configuration error
4. Invalid command or entrypoint
5. Missing required environment variables

**Debugging Steps:**

```bash
# Check current logs
kubectl logs <pod-name> -n <namespace>

# Check previous container logs (most important!)
kubectl logs <pod-name> -n <namespace> --previous

# Check all containers in pod
kubectl logs <pod-name> -n <namespace> --all-containers

# Describe pod for restart count and reasons
kubectl describe pod <pod-name> -n <namespace>

# Check exit code
kubectl get pod <pod-name> -n <namespace> -o jsonpath='{.status.containerStatuses[0].lastState.terminated.exitCode}'
```

**Exit Code Meanings:**
- `0`: Success (but container shouldn't exit)
- `1`: General error
- `137`: OOMKilled (out of memory)
- `139`: Segmentation fault
- `143`: SIGTERM (graceful termination)

**Solutions:**

```bash
# Debug interactively (override command)
kubectl run -it --rm debug --image=<same-image> --restart=Never -- /bin/sh

# Check and fix environment variables
kubectl set env deployment/<deployment-name> KEY=VALUE -n <namespace>

# Update resource limits if OOMKilled
kubectl set resources deployment/<deployment-name> \
  --limits=memory=512Mi \
  -n <namespace>
```

---

### 1.3 Pending Pods

**Symptoms:**
- Pod status: `Pending`
- Pod not scheduled to any node
- Events show scheduling errors

**Common Causes:**
1. Insufficient cluster resources (CPU/memory)
2. No nodes match node selector/affinity
3. Taints prevent scheduling
4. Volume not available
5. Resource quotas exceeded

**Debugging Steps:**

```bash
# Describe pod for scheduling events
kubectl describe pod <pod-name> -n <namespace>
# Look for "FailedScheduling" events

# Check node resources
kubectl top nodes
kubectl describe nodes

# Check resource requests
kubectl get pod <pod-name> -n <namespace> -o jsonpath='{.spec.containers[*].resources}'

# Check node selector and affinity
kubectl get pod <pod-name> -n <namespace> -o yaml | grep -A 5 nodeSelector
kubectl get pod <pod-name> -n <namespace> -o yaml | grep -A 10 affinity

# Check resource quotas
kubectl get resourcequota -n <namespace>
kubectl describe resourcequota -n <namespace>

# Check PVC status (if using volumes)
kubectl get pvc -n <namespace>
```

**Solutions:**

```bash
# Reduce resource requests
kubectl set resources deployment/<deployment-name> \
  --requests=cpu=100m,memory=128Mi \
  -n <namespace>

# Remove node selector
kubectl patch deployment <deployment-name> -n <namespace> --type=json -p='[
  {"op": "remove", "path": "/spec/template/spec/nodeSelector"}
]'

# Scale down other workloads to free resources
kubectl scale deployment <other-deployment> --replicas=0 -n <namespace>
```

---

### 1.4 OOMKilled (Out of Memory)

**Symptoms:**
- Pod status shows `OOMKilled` in last state
- Exit code: 137
- Pod keeps restarting

**Debugging Steps:**

```bash
# Check last termination reason
kubectl get pod <pod-name> -n <namespace> -o jsonpath='{.status.containerStatuses[0].lastState.terminated.reason}'

# Check memory limits
kubectl get pod <pod-name> -n <namespace> -o jsonpath='{.spec.containers[0].resources.limits.memory}'

# Check actual memory usage
kubectl top pod <pod-name> -n <namespace>

# View OOM events
kubectl get events -n <namespace> --field-selector reason=OOMKilling
```

**Solutions:**

```bash
# Increase memory limits
kubectl set resources deployment/<deployment-name> \
  --limits=memory=1Gi \
  --requests=memory=512Mi \
  -n <namespace>

# Investigate application memory leak
kubectl logs <pod-name> -n <namespace> --previous | grep -i memory
```

---

## 2. Service Issues

### 2.1 Service Has No Endpoints

**Symptoms:**
- Service exists but has no endpoints
- `kubectl get endpoints` shows `<none>`
- Cannot connect to service

**Common Causes:**
1. Label selector doesn't match any pods
2. Pods exist but not ready (readiness probe failing)
3. Pods don't exist

**Debugging Steps:**

```bash
# Check service endpoints
kubectl get endpoints <service-name> -n <namespace>

# Get service selector
kubectl get svc <service-name> -n <namespace> -o jsonpath='{.spec.selector}'

# Find matching pods
SELECTOR=$(kubectl get svc <service-name> -n <namespace> -o jsonpath='{.spec.selector}' | jq -r 'to_entries|map("\(.key)=\(.value)")|join(",")')
kubectl get pods -n <namespace> -l "$SELECTOR" --show-labels

# Check pod readiness
kubectl get pods -n <namespace> -o wide
```

**Solutions:**

```bash
# Fix service selector to match pod labels
kubectl patch svc <service-name> -n <namespace> -p '
{
  "spec": {
    "selector": {
      "app": "correct-label"
    }
  }
}'

# Check and fix readiness probe
kubectl describe pod <pod-name> -n <namespace> | grep -A 5 Readiness
```

---

### 2.2 Wrong Target Port

**Symptoms:**
- Service has endpoints but connection refused
- Port is not listening

**Debugging Steps:**

```bash
# Check service ports
kubectl get svc <service-name> -n <namespace> -o yaml

# Check container port
kubectl get pod <pod-name> -n <namespace> -o jsonpath='{.spec.containers[*].ports[*].containerPort}'

# Test connectivity from within cluster
kubectl run test-pod --rm -it --image=curlimages/curl -- curl http://<service-name>.<namespace>:80
```

**Solutions:**

```bash
# Fix target port
kubectl patch svc <service-name> -n <namespace> -p '
{
  "spec": {
    "ports": [{
      "port": 80,
      "targetPort": 8080,
      "protocol": "TCP"
    }]
  }
}'
```

---

## 3. Configuration Issues

### 3.1 Missing ConfigMap or Secret

**Symptoms:**
- Pod status: `CreateContainerConfigError`
- Events show missing ConfigMap/Secret

**Debugging Steps:**

```bash
# Check pod status
kubectl describe pod <pod-name> -n <namespace>

# List ConfigMaps
kubectl get configmaps -n <namespace>

# List Secrets
kubectl get secrets -n <namespace>

# Check what ConfigMaps/Secrets pod is trying to use
kubectl get pod <pod-name> -n <namespace> -o yaml | grep -E 'configMap|secret'
```

**Solutions:**

```bash
# Create missing ConfigMap
kubectl create configmap <configmap-name> \
  --from-literal=key1=value1 \
  --from-literal=key2=value2 \
  -n <namespace>

# Create missing Secret
kubectl create secret generic <secret-name> \
  --from-literal=password=secretvalue \
  -n <namespace>
```

---

### 3.2 Wrong ConfigMap/Secret Key

**Symptoms:**
- Pod shows warning about missing key
- Environment variable not set

**Debugging Steps:**

```bash
# View ConfigMap keys
kubectl get configmap <configmap-name> -n <namespace> -o yaml

# View Secret keys (base64 encoded)
kubectl get secret <secret-name> -n <namespace> -o jsonpath='{.data}'

# Check pod environment configuration
kubectl get pod <pod-name> -n <namespace> -o yaml | grep -A 10 env
```

**Solutions:**

```bash
# Add missing key to ConfigMap
kubectl patch configmap <configmap-name> -n <namespace> -p '
{
  "data": {
    "missing-key": "value"
  }
}'

# Fix key reference in deployment
kubectl set env deployment/<deployment-name> \
  VAR_NAME=configmap/<configmap-name>:<correct-key> \
  -n <namespace>
```

---

## 4. Networking Issues

### 4.1 DNS Resolution Failures

**Symptoms:**
- Cannot resolve service names
- `nslookup` or `dig` fails

**Debugging Steps:**

```bash
# Check CoreDNS pods
kubectl get pods -n kube-system -l k8s-app=kube-dns

# Test DNS from pod
kubectl exec <pod-name> -n <namespace> -- nslookup kubernetes.default

# Check DNS configuration
kubectl get svc -n kube-system kube-dns
```

**Solutions:**

```bash
# Restart CoreDNS
kubectl rollout restart deployment coredns -n kube-system

# Check pod DNS config
kubectl get pod <pod-name> -n <namespace> -o jsonpath='{.spec.dnsPolicy}'
```

---

### 4.2 Network Policy Blocking Traffic

**Symptoms:**
- Connection timeout to services
- Pods can't communicate

**Debugging Steps:**

```bash
# List network policies
kubectl get networkpolicies -n <namespace>

# Describe network policy
kubectl describe networkpolicy <policy-name> -n <namespace>

# Test connectivity
kubectl exec <source-pod> -n <namespace> -- wget -O- --timeout=5 http://<dest-service>
```

**Solutions:**

```bash
# Temporarily remove network policy
kubectl delete networkpolicy <policy-name> -n <namespace>

# Update network policy to allow traffic
kubectl edit networkpolicy <policy-name> -n <namespace>
```

---

## 5. Probe Issues

### 5.1 Liveness Probe Causing Restarts

**Symptoms:**
- Pod restarting frequently
- Events show liveness probe failures

**Debugging Steps:**

```bash
# Check restart count
kubectl get pod <pod-name> -n <namespace>

# View probe configuration
kubectl get pod <pod-name> -n <namespace> -o yaml | grep -A 10 livenessProbe

# Check probe failures in events
kubectl describe pod <pod-name> -n <namespace> | grep -A 5 "Liveness probe failed"
```

**Solutions:**

```bash
# Increase initialDelaySeconds
kubectl patch deployment <deployment-name> -n <namespace> --type=json -p='[
  {
    "op": "replace",
    "path": "/spec/template/spec/containers/0/livenessProbe/initialDelaySeconds",
    "value": 60
  }
]'

# Fix probe path/port
kubectl patch deployment <deployment-name> -n <namespace> --type=json -p='[
  {
    "op": "replace",
    "path": "/spec/template/spec/containers/0/livenessProbe/httpGet/path",
    "value": "/health"
  }
]'
```

---

## 6. Resource Issues

### 6.1 Node Pressure

**Symptoms:**
- Pods being evicted
- Node in `NotReady` state

**Debugging Steps:**

```bash
# Check node status
kubectl get nodes

# Describe node
kubectl describe node <node-name>

# Check node conditions
kubectl get node <node-name> -o jsonpath='{.status.conditions[?(@.type=="Ready")]}'
```

**Solutions:**

```bash
# Identify and delete unused resources
kubectl delete pod <unused-pod> --force -n <namespace>

# Drain node for maintenance
kubectl drain <node-name> --ignore-daemonsets --delete-emptydir-data
```

---

## Debugging Command Cheatsheet

### Essential Commands

```bash
# Pod debugging
kubectl get pods -n <namespace>
kubectl describe pod <pod-name> -n <namespace>
kubectl logs <pod-name> -n <namespace>
kubectl logs <pod-name> -n <namespace> --previous
kubectl exec <pod-name> -n <namespace> -- <command>

# Service debugging
kubectl get svc -n <namespace>
kubectl get endpoints -n <namespace>
kubectl describe svc <service-name> -n <namespace>

# Events
kubectl get events -n <namespace> --sort-by='.lastTimestamp'
kubectl get events -n <namespace> --field-selector type=Warning

# Resource usage
kubectl top nodes
kubectl top pods -n <namespace>

# Configuration
kubectl get configmap -n <namespace>
kubectl get secret -n <namespace>
kubectl describe pod <pod-name> -n <namespace> | grep -A 10 "Environment"

# Network
kubectl run test-curl --rm -it --image=curlimages/curl -- sh
kubectl exec <pod-name> -n <namespace> -- nslookup <service-name>
```

---

## Best Practices

1. **Always check events first**: `kubectl get events` often reveals the issue
2. **Use `--previous` for crashed containers**: Current logs may be empty
3. **Describe resources**: `kubectl describe` shows events and status
4. **Check labels and selectors**: Mismatched labels are a common issue
5. **Verify resource limits**: Too low limits cause OOMKills
6. **Test from within the cluster**: Network issues may be cluster-internal
7. **Use verbose output**: Add `-v=8` to kubectl commands for more details
8. **Keep events history**: Events are deleted after 1 hour by default

---

## Additional Tools

- **kubectl-debug**: Add ephemeral debug containers
- **stern**: Multi-pod log tailing
- **k9s**: Terminal UI for Kubernetes
- **kubectx/kubens**: Fast context/namespace switching
- **krew**: kubectl plugin manager
