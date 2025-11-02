# Implementation Guide: Kubernetes Debugging for ML Infrastructure

## Table of Contents

1. [Introduction](#introduction)
2. [Common Kubernetes Issues](#common-kubernetes-issues)
3. [Using kubectl for Debugging](#using-kubectl-for-debugging)
4. [Pod Troubleshooting Workflow](#pod-troubleshooting-workflow)
5. [Network Debugging](#network-debugging)
6. [Resource Constraints Debugging](#resource-constraints-debugging)
7. [Events and Monitoring](#events-and-monitoring)
8. [Production Debugging Toolkit](#production-debugging-toolkit)
9. [ML-Specific Debugging](#ml-specific-debugging)
10. [Advanced Debugging Techniques](#advanced-debugging-techniques)
11. [Best Practices](#best-practices)

---

## Introduction

This implementation guide provides comprehensive coverage of Kubernetes debugging techniques with a special focus on ML infrastructure challenges. You'll learn systematic approaches to diagnose and fix common issues, as well as ML-specific problems like OOM errors during training, GPU allocation failures, and model loading issues.

### What You'll Learn

- Systematic debugging methodologies for Kubernetes
- Essential kubectl commands for troubleshooting
- How to diagnose and fix common pod issues
- Network connectivity debugging techniques
- Resource constraint analysis and resolution
- ML-specific debugging patterns
- Production-ready debugging workflows

### Prerequisites

- Working Kubernetes cluster (kind, minikube, or cloud provider)
- kubectl configured and working
- Basic understanding of Kubernetes resources
- Familiarity with Linux command line

---

## Common Kubernetes Issues

Understanding common failure patterns helps you quickly identify and resolve issues.

### 1. ImagePullBackOff / ErrImagePull

**Symptoms:**
- Pod stuck in `ImagePullBackOff` or `ErrImagePull` status
- Events show "Failed to pull image" or "image not found"
- Pods remain in Pending state

**Common Causes:**
```yaml
# Typo in image name
image: nignx:1.21  # Should be nginx:1.21

# Wrong registry or missing credentials
image: private.registry.com/model-server:latest  # Needs imagePullSecrets

# Invalid tag
image: tensorflow/tensorflow:99.99  # Tag doesn't exist

# Missing pull policy
imagePullPolicy: Always  # May cause rate limiting
```

**Debugging Steps:**

```bash
# 1. Check pod status
kubectl get pods -n <namespace>

# 2. Describe pod to see events
kubectl describe pod <pod-name> -n <namespace>
# Look for: "Failed to pull image", "manifest unknown", "unauthorized"

# 3. Verify image exists
docker pull <image-name>:<tag>

# 4. Check image pull secrets
kubectl get secrets -n <namespace>
kubectl describe pod <pod-name> -n <namespace> | grep -A 2 "Image Pull Secrets"
```

**Solutions:**

```bash
# Fix image name typo
kubectl set image deployment/<deployment> <container>=<correct-image> -n <namespace>

# Create image pull secret
kubectl create secret docker-registry regcred \
  --docker-server=<registry-url> \
  --docker-username=<username> \
  --docker-password=<password> \
  -n <namespace>

# Add secret to deployment
kubectl patch deployment <deployment> -n <namespace> -p '
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

**ML-Specific Example:**

```yaml
# Large ML model image pulling issues
apiVersion: v1
kind: Pod
metadata:
  name: llm-inference
spec:
  containers:
  - name: model-server
    image: myregistry.azurecr.io/llama-2-70b:latest
    imagePullPolicy: IfNotPresent  # Avoid pulling 50GB+ images repeatedly
  imagePullSecrets:
  - name: azure-registry-creds
  # Add tolerations for slow image pulls
  tolerations:
  - key: "node.kubernetes.io/not-ready"
    operator: "Exists"
    effect: "NoExecute"
    tolerationSeconds: 600  # Allow 10 minutes for large image pull
```

---

### 2. CrashLoopBackOff

**Symptoms:**
- Pod status shows `CrashLoopBackOff`
- Restart count continuously increases
- Container exits shortly after starting
- Increasing backoff time between restarts

**Common Causes:**

```yaml
# Application error on startup
command: ["python", "train.py"]
args: ["--config", "nonexistent.yaml"]  # File doesn't exist

# Missing dependencies
env:
- name: MODEL_PATH
  value: "/models/checkpoint.pth"  # Path doesn't exist

# Wrong entrypoint
command: ["/bin/bash"]
args: ["-c", "pythn app.py"]  # Typo in python command

# Port already in use
ports:
- containerPort: 8080
env:
- name: PORT
  value: "9000"  # Application tries to bind to 9000, conflicts
```

**Debugging Steps:**

```bash
# 1. Check restart count and status
kubectl get pods -n <namespace>

# 2. View current logs
kubectl logs <pod-name> -n <namespace>

# 3. View previous container logs (critical!)
kubectl logs <pod-name> -n <namespace> --previous

# 4. Check exit code
kubectl describe pod <pod-name> -n <namespace> | grep -A 10 "Last State"
# Look for exit code and reason

# 5. Get detailed container status
kubectl get pod <pod-name> -n <namespace> -o jsonpath='{.status.containerStatuses[0]}'

# 6. Check for liveness probe issues
kubectl describe pod <pod-name> -n <namespace> | grep -A 5 "Liveness"
```

**Solutions:**

```bash
# Fix ConfigMap issue
kubectl edit configmap <configmap-name> -n <namespace>

# Update environment variable
kubectl set env deployment/<deployment> MODEL_PATH=/correct/path -n <namespace>

# Disable liveness probe temporarily to debug
kubectl patch deployment <deployment> -n <namespace> --type=json -p='[
  {"op": "remove", "path": "/spec/template/spec/containers/0/livenessProbe"}
]'

# Add debug command to keep container running
kubectl set env deployment/<deployment> DEBUG=true -n <namespace>
```

**ML-Specific Example:**

```yaml
# Training job crashing due to invalid hyperparameters
apiVersion: batch/v1
kind: Job
metadata:
  name: model-training
spec:
  backoffLimit: 3  # Retry 3 times
  template:
    spec:
      restartPolicy: Never  # Don't restart on failure
      containers:
      - name: trainer
        image: pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
        command: ["python", "train.py"]
        env:
        - name: LEARNING_RATE
          value: "0.001"  # Validate before running
        - name: BATCH_SIZE
          value: "32"
        # Add validation script
        lifecycle:
          postStart:
            exec:
              command: ["/bin/sh", "-c", "python validate_config.py || exit 1"]
```

---

### 3. Pending Pods

**Symptoms:**
- Pod stuck in `Pending` state
- Pod never gets scheduled to a node
- No container activity

**Common Causes:**

```yaml
# Insufficient resources
resources:
  requests:
    memory: "64Gi"  # No node has this much memory
    cpu: "32"

# Node selector mismatch
nodeSelector:
  gpu: "nvidia-a100"  # No nodes with this label

# Taints and tolerations
# Node has taint, pod missing toleration

# Persistent Volume not available
volumes:
- name: data
  persistentVolumeClaim:
    claimName: missing-pvc  # PVC doesn't exist
```

**Debugging Steps:**

```bash
# 1. Check why pod is pending
kubectl describe pod <pod-name> -n <namespace>
# Look for: "FailedScheduling", "Insufficient cpu/memory"

# 2. Check node resources
kubectl top nodes
kubectl describe nodes

# 3. Check node selectors and labels
kubectl get pod <pod-name> -n <namespace> -o yaml | grep -A 5 nodeSelector
kubectl get nodes --show-labels

# 4. Check taints and tolerations
kubectl describe nodes | grep -A 3 Taints

# 5. Check PVC status
kubectl get pvc -n <namespace>

# 6. Check events
kubectl get events -n <namespace> --field-selector involvedObject.name=<pod-name>
```

**Solutions:**

```bash
# Reduce resource requests
kubectl set resources deployment/<deployment> \
  --requests=cpu=1,memory=2Gi \
  -n <namespace>

# Add node label
kubectl label nodes <node-name> gpu=nvidia-a100

# Add toleration
kubectl patch deployment <deployment> -n <namespace> -p '
{
  "spec": {
    "template": {
      "spec": {
        "tolerations": [{
          "key": "nvidia.com/gpu",
          "operator": "Exists",
          "effect": "NoSchedule"
        }]
      }
    }
  }
}'
```

---

### 4. OOMKilled (Out of Memory)

**Symptoms:**
- Pod shows `OOMKilled` in status
- Container restarts with exit code 137
- Events show "Container exceeded memory limit"

**Common Causes:**

```yaml
# Memory limit too low
resources:
  limits:
    memory: "128Mi"  # Application needs 512Mi

# Memory leak
# Application not releasing memory properly

# Large dataset loading
# Loading entire dataset into memory

# Model too large
# Model parameters exceed available memory
```

**Debugging Steps:**

```bash
# 1. Check for OOMKilled status
kubectl get pods -n <namespace>
kubectl describe pod <pod-name> -n <namespace> | grep -i oom

# 2. Check memory usage before crash
kubectl top pod <pod-name> -n <namespace>

# 3. Check memory limits
kubectl get pod <pod-name> -n <namespace> -o jsonpath='{.spec.containers[0].resources}'

# 4. Check events for memory-related issues
kubectl get events -n <namespace> --field-selector reason=OOMKilled

# 5. Analyze logs before crash
kubectl logs <pod-name> -n <namespace> --previous | tail -50
```

**Solutions:**

```bash
# Increase memory limit
kubectl set resources deployment/<deployment> \
  --limits=memory=2Gi \
  -n <namespace>

# Add memory request to prevent overcommit
kubectl set resources deployment/<deployment> \
  --requests=memory=1Gi \
  --limits=memory=2Gi \
  -n <namespace>
```

---

## Using kubectl for Debugging

Master these essential kubectl commands for effective debugging.

### Pod Inspection Commands

```bash
# Get pod status with additional details
kubectl get pods -n <namespace> -o wide

# Get pod in YAML format (full details)
kubectl get pod <pod-name> -n <namespace> -o yaml

# Get specific field using jsonpath
kubectl get pod <pod-name> -n <namespace> \
  -o jsonpath='{.status.containerStatuses[0].state}'

# Watch pod status in real-time
kubectl get pods -n <namespace> -w

# Get pods with specific labels
kubectl get pods -n <namespace> -l app=training

# Get pods on specific node
kubectl get pods -n <namespace> --field-selector spec.nodeName=<node-name>
```

### Describe Command (Most Important!)

```bash
# Describe pod (shows events, status, conditions)
kubectl describe pod <pod-name> -n <namespace>

# Describe specific sections
kubectl describe pod <pod-name> -n <namespace> | grep -A 10 "Events:"
kubectl describe pod <pod-name> -n <namespace> | grep -A 5 "Conditions:"
kubectl describe pod <pod-name> -n <namespace> | grep -A 10 "Containers:"

# Describe deployment
kubectl describe deployment <deployment-name> -n <namespace>

# Describe node
kubectl describe node <node-name>
```

### Log Analysis Commands

```bash
# View current logs
kubectl logs <pod-name> -n <namespace>

# View logs from previous container (crashed)
kubectl logs <pod-name> -n <namespace> --previous

# Follow logs in real-time
kubectl logs <pod-name> -n <namespace> -f

# Get logs from specific container (multi-container pod)
kubectl logs <pod-name> -n <namespace> -c <container-name>

# Get last N lines
kubectl logs <pod-name> -n <namespace> --tail=100

# Get logs since specific time
kubectl logs <pod-name> -n <namespace> --since=1h

# Get logs with timestamps
kubectl logs <pod-name> -n <namespace> --timestamps

# Save logs to file
kubectl logs <pod-name> -n <namespace> > pod-logs.txt

# Get logs from all pods in deployment
kubectl logs -n <namespace> -l app=<app-label> --all-containers=true
```

### Interactive Debugging

```bash
# Execute single command in pod
kubectl exec <pod-name> -n <namespace> -- ls -la /app

# Get interactive shell
kubectl exec -it <pod-name> -n <namespace> -- /bin/bash
kubectl exec -it <pod-name> -n <namespace> -- /bin/sh

# Execute in specific container
kubectl exec -it <pod-name> -n <namespace> -c <container-name> -- /bin/bash

# Common debugging commands inside pod
kubectl exec <pod-name> -n <namespace> -- ps aux
kubectl exec <pod-name> -n <namespace> -- netstat -tulpn
kubectl exec <pod-name> -n <namespace> -- df -h
kubectl exec <pod-name> -n <namespace> -- cat /etc/resolv.conf
kubectl exec <pod-name> -n <namespace> -- env
```

### File Operations

```bash
# Copy file from pod to local
kubectl cp <namespace>/<pod-name>:/path/to/file ./local-file

# Copy file from local to pod
kubectl cp ./local-file <namespace>/<pod-name>:/path/to/file

# Copy from specific container
kubectl cp <namespace>/<pod-name>:/path/to/file ./local-file -c <container-name>

# Copy entire directory
kubectl cp <namespace>/<pod-name>:/app/logs ./logs
```

### Port Forwarding for Testing

```bash
# Forward pod port to local
kubectl port-forward <pod-name> -n <namespace> 8080:80

# Forward service port
kubectl port-forward svc/<service-name> -n <namespace> 8080:80

# Forward to specific local address
kubectl port-forward --address 0.0.0.0 <pod-name> -n <namespace> 8080:80

# Background port forward
kubectl port-forward <pod-name> -n <namespace> 8080:80 &
```

### Resource Usage Monitoring

```bash
# Get node resource usage
kubectl top nodes

# Get pod resource usage
kubectl top pods -n <namespace>

# Get specific pod usage
kubectl top pod <pod-name> -n <namespace>

# Get pods sorted by memory
kubectl top pods -n <namespace> --sort-by=memory

# Get pods sorted by CPU
kubectl top pods -n <namespace> --sort-by=cpu

# Get all containers in pod
kubectl top pod <pod-name> -n <namespace> --containers
```

---

## Pod Troubleshooting Workflow

Follow this systematic workflow for debugging pod issues.

### Step 1: Identify the Problem

```bash
# Start with high-level overview
kubectl get all -n <namespace>

# Check pod status
kubectl get pods -n <namespace>

# Common problematic states:
# - Pending: Not scheduled to node
# - ImagePullBackOff: Can't pull container image
# - CrashLoopBackOff: Container keeps crashing
# - Error: Container exited with error
# - OOMKilled: Out of memory
# - CreateContainerConfigError: Configuration issue
# - Unknown: Node communication issue
```

### Step 2: Gather Initial Information

```bash
# Get detailed pod information
kubectl describe pod <pod-name> -n <namespace>

# Key sections to check:
# - Status: Current pod phase
# - Conditions: Ready, Initialized, ContainersReady
# - Containers: Container states and ready status
# - Events: Recent events (most important!)
# - QoS Class: Quality of Service tier
# - Node: Which node it's scheduled on
```

### Step 3: Analyze Events

```bash
# Get events for namespace
kubectl get events -n <namespace> --sort-by='.lastTimestamp'

# Get events for specific pod
kubectl get events -n <namespace> \
  --field-selector involvedObject.name=<pod-name>

# Get warning events only
kubectl get events -n <namespace> --field-selector type=Warning

# Event types to look for:
# - FailedScheduling: Can't schedule to node
# - FailedMount: Volume mount issues
# - FailedCreate: Pod creation failed
# - Unhealthy: Probe failures
# - BackOff: CrashLoopBackOff
# - Killing: Container being killed
# - OOMKilling: Out of memory kill
```

### Step 4: Check Logs

```bash
# Current logs
kubectl logs <pod-name> -n <namespace>

# Previous container logs (if crashed)
kubectl logs <pod-name> -n <namespace> --previous

# Look for:
# - Error messages
# - Stack traces
# - Exit codes
# - Startup messages
# - Configuration errors
# - Connection errors
```

### Step 5: Inspect Configuration

```bash
# Get full pod YAML
kubectl get pod <pod-name> -n <namespace> -o yaml

# Check specific configuration:

# Container image
kubectl get pod <pod-name> -n <namespace> \
  -o jsonpath='{.spec.containers[*].image}'

# Environment variables
kubectl get pod <pod-name> -n <namespace> \
  -o jsonpath='{.spec.containers[*].env}'

# Volume mounts
kubectl get pod <pod-name> -n <namespace> \
  -o jsonpath='{.spec.containers[*].volumeMounts}'

# Resource limits
kubectl get pod <pod-name> -n <namespace> \
  -o jsonpath='{.spec.containers[*].resources}'

# Node selector
kubectl get pod <pod-name> -n <namespace> \
  -o jsonpath='{.spec.nodeSelector}'

# Tolerations
kubectl get pod <pod-name> -n <namespace> \
  -o jsonpath='{.spec.tolerations}'
```

### Step 6: Interactive Debugging (if pod is running)

```bash
# Get shell access
kubectl exec -it <pod-name> -n <namespace> -- /bin/sh

# Inside the pod, check:

# Process list
ps aux

# Disk usage
df -h

# Network interfaces
ip addr
ifconfig

# DNS resolution
nslookup kubernetes.default
cat /etc/resolv.conf

# Environment variables
env

# Application files
ls -la /app
cat /app/config.yaml

# Test network connectivity
ping <service-name>
curl http://<service-name>:8080

# Check listening ports
netstat -tulpn
ss -tulpn
```

### Step 7: Check Dependencies

```bash
# Check ConfigMaps
kubectl get configmap -n <namespace>
kubectl describe configmap <name> -n <namespace>

# Check Secrets
kubectl get secrets -n <namespace>
kubectl describe secret <name> -n <namespace>

# Check Services
kubectl get svc -n <namespace>
kubectl describe svc <name> -n <namespace>

# Check Endpoints
kubectl get endpoints -n <namespace>

# Check PVCs
kubectl get pvc -n <namespace>
kubectl describe pvc <name> -n <namespace>
```

### Step 8: Apply Fix

```bash
# Option 1: Edit resource directly
kubectl edit pod <pod-name> -n <namespace>
kubectl edit deployment <deployment> -n <namespace>

# Option 2: Patch resource
kubectl patch deployment <deployment> -n <namespace> -p '<json-patch>'

# Option 3: Use kubectl set commands
kubectl set image deployment/<deployment> <container>=<new-image> -n <namespace>
kubectl set env deployment/<deployment> KEY=VALUE -n <namespace>
kubectl set resources deployment/<deployment> --limits=cpu=1,memory=1Gi -n <namespace>

# Option 4: Apply updated YAML
kubectl apply -f updated-deployment.yaml

# Option 5: Replace resource
kubectl replace -f deployment.yaml --force
```

### Step 9: Verify Fix

```bash
# Watch pod status
kubectl get pods -n <namespace> -w

# Check if pods are running and ready
kubectl get pods -n <namespace>

# Verify no new errors in events
kubectl get events -n <namespace> --field-selector type=Warning

# Check logs for successful startup
kubectl logs <pod-name> -n <namespace> | tail -20

# Test application functionality
kubectl exec <pod-name> -n <namespace> -- curl http://localhost:8080/health
```

---

## Network Debugging

Network issues are common in Kubernetes. Here's how to debug them systematically.

### Service Connectivity Issues

```bash
# 1. Check if service exists
kubectl get svc -n <namespace>

# 2. Describe service
kubectl describe svc <service-name> -n <namespace>

# 3. Check service endpoints
kubectl get endpoints <service-name> -n <namespace>

# If endpoints are empty:
# - Service selector doesn't match any pods
# - Pods exist but aren't ready
# - Pods have wrong labels

# 4. Compare service selector with pod labels
kubectl get svc <service-name> -n <namespace> -o jsonpath='{.spec.selector}'
kubectl get pods -n <namespace> --show-labels

# 5. Check if pods are ready
kubectl get pods -n <namespace> -l app=<label>

# 6. Test service connectivity from another pod
kubectl run debug-pod --rm -it --image=nicolaka/netshoot -n <namespace> -- /bin/bash

# Inside debug pod:
nslookup <service-name>
nslookup <service-name>.<namespace>.svc.cluster.local
curl http://<service-name>:<port>
wget -O- http://<service-name>:<port>
```

### DNS Resolution Issues

```bash
# 1. Check DNS pods are running
kubectl get pods -n kube-system -l k8s-app=kube-dns

# 2. Check CoreDNS logs
kubectl logs -n kube-system -l k8s-app=kube-dns

# 3. Test DNS from pod
kubectl exec <pod-name> -n <namespace> -- nslookup kubernetes.default

# 4. Check DNS configuration
kubectl exec <pod-name> -n <namespace> -- cat /etc/resolv.conf

# Should show:
# nameserver 10.96.0.10 (or cluster DNS IP)
# search <namespace>.svc.cluster.local svc.cluster.local cluster.local
# options ndots:5

# 5. Test DNS resolution patterns
kubectl exec <pod-name> -n <namespace> -- nslookup <service-name>
kubectl exec <pod-name> -n <namespace> -- nslookup <service-name>.<namespace>
kubectl exec <pod-name> -n <namespace> -- nslookup <service-name>.<namespace>.svc.cluster.local
```

### Network Policy Debugging

```bash
# 1. Check if NetworkPolicies exist
kubectl get networkpolicy -n <namespace>

# 2. Describe NetworkPolicy
kubectl describe networkpolicy <policy-name> -n <namespace>

# 3. Check policy selectors
kubectl get networkpolicy <policy-name> -n <namespace> -o yaml

# 4. Verify pod labels match policy
kubectl get pods -n <namespace> --show-labels

# 5. Test connectivity with and without policy
# Temporarily delete policy to test
kubectl delete networkpolicy <policy-name> -n <namespace>

# Test connectivity
kubectl exec <pod-name> -n <namespace> -- curl http://<target-service>

# Recreate policy
kubectl apply -f networkpolicy.yaml
```

### Port and Protocol Issues

```bash
# 1. Check service ports configuration
kubectl get svc <service-name> -n <namespace> -o yaml

# Verify:
# - spec.ports[].port: Service port
# - spec.ports[].targetPort: Container port
# - spec.ports[].protocol: TCP/UDP

# 2. Check container ports
kubectl get pod <pod-name> -n <namespace> \
  -o jsonpath='{.spec.containers[*].ports}'

# 3. Test if application is listening
kubectl exec <pod-name> -n <namespace> -- netstat -tulpn
kubectl exec <pod-name> -n <namespace> -- ss -tulpn

# 4. Test port connectivity
kubectl exec debug-pod -n <namespace> -- nc -zv <service-name> <port>
kubectl exec debug-pod -n <namespace> -- telnet <service-name> <port>
```

### Ingress Debugging

```bash
# 1. Check ingress resource
kubectl get ingress -n <namespace>
kubectl describe ingress <ingress-name> -n <namespace>

# 2. Check ingress controller logs
kubectl logs -n ingress-nginx -l app.kubernetes.io/component=controller

# 3. Verify backend service
kubectl get svc <backend-service> -n <namespace>
kubectl get endpoints <backend-service> -n <namespace>

# 4. Test ingress path
curl -H "Host: <hostname>" http://<ingress-ip>/<path>

# 5. Check TLS configuration
kubectl get secret <tls-secret> -n <namespace>
openssl s_client -connect <hostname>:443 -servername <hostname>
```

---

## Resource Constraints Debugging

Resource issues are critical in ML workloads where training can require significant CPU, memory, and GPU.

### Understanding Resource Requests and Limits

```yaml
resources:
  requests:
    # Guaranteed resources (used for scheduling)
    cpu: "1"          # 1 CPU core
    memory: "2Gi"     # 2 Gibibytes
    nvidia.com/gpu: 1 # 1 GPU
  limits:
    # Maximum allowed resources (enforced by kernel)
    cpu: "2"          # Can burst to 2 cores
    memory: "4Gi"     # Hard limit, OOMKilled if exceeded
    nvidia.com/gpu: 1 # GPU limits must equal requests
```

### Debugging Insufficient Resources

```bash
# 1. Check node resources
kubectl top nodes

# 2. Get detailed node information
kubectl describe nodes

# Look for:
# - Capacity: Total resources
# - Allocatable: Available for pods
# - Allocated resources: Current usage
# - Non-terminated Pods: Pods using resources

# 3. Check pod resource requests
kubectl describe pod <pod-name> -n <namespace> | grep -A 5 "Requests:"

# 4. Find why pod is pending
kubectl describe pod <pod-name> -n <namespace> | grep -A 10 "Events:"

# Common messages:
# - "Insufficient cpu"
# - "Insufficient memory"
# - "Insufficient nvidia.com/gpu"

# 5. List all pod resource usage
kubectl get pods -n <namespace> -o custom-columns=\
NAME:.metadata.name,\
CPU_REQ:.spec.containers[*].resources.requests.cpu,\
MEM_REQ:.spec.containers[*].resources.requests.memory,\
CPU_LIM:.spec.containers[*].resources.limits.cpu,\
MEM_LIM:.spec.containers[*].resources.limits.memory
```

### Debugging OOM (Out of Memory) Issues

```bash
# 1. Check for OOMKilled status
kubectl get pods -n <namespace>
kubectl describe pod <pod-name> -n <namespace> | grep -i "oom"

# 2. Check memory usage history
kubectl top pod <pod-name> -n <namespace>

# 3. Get exit code (137 = OOMKilled)
kubectl get pod <pod-name> -n <namespace> \
  -o jsonpath='{.status.containerStatuses[0].lastState.terminated.exitCode}'

# 4. Check memory limit
kubectl get pod <pod-name> -n <namespace> \
  -o jsonpath='{.spec.containers[0].resources.limits.memory}'

# 5. Check events for OOM
kubectl get events -n <namespace> \
  --field-selector reason=OOMKilled,involvedObject.name=<pod-name>
```

### ML-Specific: GPU Resource Debugging

```bash
# 1. Check GPU availability on nodes
kubectl get nodes -o json | \
  jq '.items[] | {name:.metadata.name, gpu:.status.allocatable."nvidia.com/gpu"}'

# 2. Check GPU requests in pod
kubectl get pod <pod-name> -n <namespace> \
  -o jsonpath='{.spec.containers[*].resources.requests.nvidia\.com/gpu}'

# 3. Check if pod has GPU allocated
kubectl describe pod <pod-name> -n <namespace> | grep -i gpu

# 4. Check NVIDIA device plugin
kubectl get pods -n kube-system -l name=nvidia-device-plugin-ds

# 5. Check GPU utilization
kubectl exec <pod-name> -n <namespace> -- nvidia-smi

# 6. Test GPU access
kubectl exec <pod-name> -n <namespace> -- python -c "import torch; print(torch.cuda.is_available())"
```

### Setting Appropriate Resource Limits

```bash
# Update deployment with proper resources
kubectl set resources deployment/<deployment> -n <namespace> \
  --requests=cpu=2,memory=8Gi \
  --limits=cpu=4,memory=16Gi

# For GPU workloads
kubectl patch deployment <deployment> -n <namespace> -p '
{
  "spec": {
    "template": {
      "spec": {
        "containers": [{
          "name": "<container-name>",
          "resources": {
            "requests": {
              "cpu": "4",
              "memory": "16Gi",
              "nvidia.com/gpu": "1"
            },
            "limits": {
              "cpu": "8",
              "memory": "32Gi",
              "nvidia.com/gpu": "1"
            }
          }
        }]
      }
    }
  }
}'
```

---

## Events and Monitoring

Events provide critical insights into what's happening in your cluster.

### Understanding Kubernetes Events

```bash
# Get all events in namespace
kubectl get events -n <namespace>

# Sort events by timestamp
kubectl get events -n <namespace> --sort-by='.lastTimestamp'

# Get recent events
kubectl get events -n <namespace> --sort-by='.lastTimestamp' | tail -20

# Filter by type
kubectl get events -n <namespace> --field-selector type=Warning
kubectl get events -n <namespace> --field-selector type=Normal

# Filter by reason
kubectl get events -n <namespace> --field-selector reason=FailedScheduling
kubectl get events -n <namespace> --field-selector reason=OOMKilling

# Filter by involved object
kubectl get events -n <namespace> \
  --field-selector involvedObject.name=<pod-name>

# Custom output format
kubectl get events -n <namespace> -o custom-columns=\
LAST_SEEN:.lastTimestamp,\
TYPE:.type,\
REASON:.reason,\
OBJECT:.involvedObject.name,\
MESSAGE:.message
```

### Common Event Reasons

```bash
# Scheduling Events
FailedScheduling      # Can't schedule pod to node
Scheduled             # Pod scheduled successfully

# Image Events
Pulling               # Pulling container image
Pulled                # Image pulled successfully
Failed                # Failed to pull image
BackOff               # Back-off pulling image

# Container Events
Created               # Container created
Started               # Container started
Killing               # Container being killed
Killed                # Container killed
OOMKilling            # Out of memory kill

# Probe Events
Unhealthy             # Probe failed
ProbeWarning          # Probe configuration issue

# Volume Events
FailedMount           # Failed to mount volume
SuccessfulAttachVolume # Volume attached
FailedAttachVolume    # Failed to attach volume
```

### Monitoring Pod Health

```bash
# Check pod conditions
kubectl get pod <pod-name> -n <namespace> \
  -o jsonpath='{.status.conditions}' | jq

# Conditions to monitor:
# - PodScheduled: Pod assigned to node
# - Initialized: Init containers completed
# - ContainersReady: All containers ready
# - Ready: Pod ready to serve traffic

# Check probe status
kubectl describe pod <pod-name> -n <namespace> | grep -A 5 "Liveness:"
kubectl describe pod <pod-name> -n <namespace> | grep -A 5 "Readiness:"

# Monitor pod phase changes
kubectl get pods -n <namespace> -w
```

### Setting Up Monitoring for ML Workloads

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: training-job
  annotations:
    prometheus.io/scrape: "true"
    prometheus.io/port: "8080"
    prometheus.io/path: "/metrics"
spec:
  containers:
  - name: trainer
    image: pytorch-training:latest
    ports:
    - containerPort: 8080
      name: metrics
    livenessProbe:
      httpGet:
        path: /health
        port: 8080
      initialDelaySeconds: 60
      periodSeconds: 30
      timeoutSeconds: 5
      failureThreshold: 3
    readinessProbe:
      httpGet:
        path: /ready
        port: 8080
      initialDelaySeconds: 30
      periodSeconds: 10
    resources:
      requests:
        cpu: "4"
        memory: "16Gi"
        nvidia.com/gpu: "1"
      limits:
        cpu: "8"
        memory: "32Gi"
        nvidia.com/gpu: "1"
```

---

## Production Debugging Toolkit

Essential tools and practices for production debugging.

### Creating Debug Pods

```bash
# Quick debug pod with networking tools
kubectl run debug-pod --rm -it --image=nicolaka/netshoot -n <namespace> -- /bin/bash

# Debug pod with curl
kubectl run debug-curl --rm -it --image=curlimages/curl -n <namespace> -- sh

# Debug pod with Python
kubectl run debug-python --rm -it --image=python:3.11-slim -n <namespace> -- python

# Debug pod on specific node
kubectl run debug-node --rm -it \
  --image=nicolaka/netshoot \
  --overrides='{"spec": {"nodeName": "<node-name>"}}' \
  -- /bin/bash

# Persistent debug pod (doesn't delete on exit)
kubectl run debug-persistent --image=nicolaka/netshoot -n <namespace> -- sleep infinity
```

### Ephemeral Debug Containers (Kubernetes 1.23+)

```bash
# Add debug container to running pod
kubectl debug <pod-name> -n <namespace> \
  -it \
  --image=nicolaka/netshoot \
  --target=<container-name>

# Debug with different image
kubectl debug <pod-name> -n <namespace> \
  -it \
  --image=busybox:1.35 \
  --copy-to=<pod-name>-debug

# Debug node by creating pod on node
kubectl debug node/<node-name> -it --image=ubuntu
```

### Advanced kubectl Techniques

```bash
# Get all resources with specific label
kubectl get all -n <namespace> -l app=<label>

# Diff before applying changes
kubectl diff -f deployment.yaml

# Dry run to see what would be created
kubectl apply -f deployment.yaml --dry-run=client -o yaml

# Server-side dry run
kubectl apply -f deployment.yaml --dry-run=server

# Get resource with custom columns
kubectl get pods -n <namespace> -o custom-columns=\
NAME:.metadata.name,\
STATUS:.status.phase,\
NODE:.spec.nodeName,\
IP:.status.podIP

# Watch specific field
kubectl get pod <pod-name> -n <namespace> \
  -o jsonpath='{.status.phase}' \
  --watch

# Use alternative kubeconfig
kubectl get pods --kubeconfig=/path/to/config

# Use specific context
kubectl get pods --context=<context-name>
```

### Debugging Scripts

```bash
#!/bin/bash
# comprehensive-debug.sh - Complete pod debugging

POD_NAME=$1
NAMESPACE=$2

echo "=== Pod Status ==="
kubectl get pod $POD_NAME -n $NAMESPACE -o wide

echo -e "\n=== Pod Description ==="
kubectl describe pod $POD_NAME -n $NAMESPACE

echo -e "\n=== Recent Events ==="
kubectl get events -n $NAMESPACE \
  --field-selector involvedObject.name=$POD_NAME \
  --sort-by='.lastTimestamp' | tail -10

echo -e "\n=== Container Logs ==="
kubectl logs $POD_NAME -n $NAMESPACE --tail=50

echo -e "\n=== Previous Container Logs (if crashed) ==="
kubectl logs $POD_NAME -n $NAMESPACE --previous --tail=50 2>/dev/null || echo "No previous logs"

echo -e "\n=== Resource Usage ==="
kubectl top pod $POD_NAME -n $NAMESPACE 2>/dev/null || echo "Metrics not available"

echo -e "\n=== Container Status ==="
kubectl get pod $POD_NAME -n $NAMESPACE \
  -o jsonpath='{.status.containerStatuses}' | jq

echo -e "\n=== Node Information ==="
NODE=$(kubectl get pod $POD_NAME -n $NAMESPACE -o jsonpath='{.spec.nodeName}')
kubectl describe node $NODE | grep -A 10 "Allocated resources"
```

---

## ML-Specific Debugging

Special considerations for ML infrastructure debugging.

### OOM Errors in Training

**Problem:** Model training crashes due to out-of-memory errors.

```bash
# Symptoms
kubectl get pods -n ml-training
# NAME                     READY   STATUS      RESTARTS   AGE
# training-job-xyz123      0/1     OOMKilled   5          10m

# Debug steps
# 1. Check memory limit
kubectl get pod training-job-xyz123 -n ml-training \
  -o jsonpath='{.spec.containers[0].resources.limits.memory}'

# 2. Check logs before crash
kubectl logs training-job-xyz123 -n ml-training --previous | tail -100

# 3. Look for memory usage patterns
# Common causes:
# - Batch size too large
# - Model too large for available memory
# - Data loader using too much memory
# - Memory leak in training loop

# 4. Get last known memory usage
kubectl describe pod training-job-xyz123 -n ml-training | grep -A 5 "Last State"
```

**Solutions:**

```yaml
# Solution 1: Increase memory limit
apiVersion: batch/v1
kind: Job
metadata:
  name: training-job
spec:
  template:
    spec:
      containers:
      - name: trainer
        image: pytorch-training:latest
        resources:
          requests:
            memory: "32Gi"  # Increased from 16Gi
          limits:
            memory: "64Gi"  # Increased from 32Gi
        env:
        # Solution 2: Reduce batch size
        - name: BATCH_SIZE
          value: "16"  # Reduced from 32
        # Solution 3: Enable gradient checkpointing
        - name: GRADIENT_CHECKPOINTING
          value: "true"
        # Solution 4: Use mixed precision
        - name: USE_AMP
          value: "true"
```

### GPU Allocation Issues

**Problem:** Pods can't access GPUs or wrong GPU type allocated.

```bash
# Symptoms
kubectl describe pod training-pod -n ml-training
# Events:
#   Warning  FailedScheduling  10s   No nodes are available with requested GPU

# Debug steps
# 1. Check GPU availability
kubectl get nodes -o json | \
  jq -r '.items[] | "\(.metadata.name): \(.status.allocatable["nvidia.com/gpu"] // "0") GPUs"'

# 2. Check GPU node labels
kubectl get nodes -l nvidia.com/gpu.present=true --show-labels

# 3. Check GPU device plugin
kubectl get pods -n kube-system -l name=nvidia-device-plugin-ds

# 4. Check NVIDIA runtime
kubectl get nodes -o json | \
  jq -r '.items[] | "\(.metadata.name): \(.status.nodeInfo.containerRuntimeVersion)"'

# 5. Test GPU in running pod
kubectl exec training-pod -n ml-training -- nvidia-smi

# 6. Check CUDA availability
kubectl exec training-pod -n ml-training -- python -c \
  "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); \
   print(f'GPU count: {torch.cuda.device_count()}'); \
   print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

**Solutions:**

```yaml
# Solution: Proper GPU configuration
apiVersion: v1
kind: Pod
metadata:
  name: gpu-training
spec:
  # Schedule on GPU nodes
  nodeSelector:
    nvidia.com/gpu.present: "true"
    # Optional: specific GPU type
    nvidia.com/gpu.product: "Tesla-V100-SXM2-16GB"

  # Tolerate GPU taints
  tolerations:
  - key: nvidia.com/gpu
    operator: Exists
    effect: NoSchedule

  containers:
  - name: trainer
    image: pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
    resources:
      limits:
        nvidia.com/gpu: 1  # Request 1 GPU

    # Verify GPU access on startup
    command:
    - /bin/bash
    - -c
    - |
      nvidia-smi
      python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"
      python train.py
```

### Model Loading Failures

**Problem:** Model fails to load due to missing files, corrupted checkpoints, or version mismatches.

```bash
# Symptoms
kubectl logs model-server-xyz -n ml-inference
# Error: FileNotFoundError: [Errno 2] No such file or directory: '/models/checkpoint.pth'

# Debug steps
# 1. Check if volume is mounted
kubectl describe pod model-server-xyz -n ml-inference | grep -A 10 "Mounts:"

# 2. Check volume contents
kubectl exec model-server-xyz -n ml-inference -- ls -la /models/

# 3. Check PVC status
kubectl get pvc -n ml-inference
kubectl describe pvc model-storage -n ml-inference

# 4. Check PV status
kubectl get pv
kubectl describe pv <pv-name>

# 5. Test model loading
kubectl exec -it model-server-xyz -n ml-inference -- python -c \
  "import torch; model = torch.load('/models/checkpoint.pth'); print('Model loaded successfully')"

# 6. Check model file permissions
kubectl exec model-server-xyz -n ml-inference -- stat /models/checkpoint.pth
```

**Solutions:**

```yaml
# Solution: Proper volume configuration and validation
apiVersion: v1
kind: Pod
metadata:
  name: model-server
spec:
  # Init container to validate model
  initContainers:
  - name: model-validator
    image: pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
    command:
    - /bin/bash
    - -c
    - |
      set -e
      echo "Checking model files..."
      ls -lh /models/

      if [ ! -f "/models/model.pth" ]; then
        echo "ERROR: model.pth not found"
        exit 1
      fi

      echo "Validating model..."
      python -c "
      import torch
      try:
          model = torch.load('/models/model.pth')
          print(f'Model loaded successfully')
          print(f'Model type: {type(model)}')
      except Exception as e:
          print(f'ERROR: Failed to load model: {e}')
          exit(1)
      "

      echo "Model validation complete"
    volumeMounts:
    - name: model-storage
      mountPath: /models

  containers:
  - name: server
    image: model-server:latest
    env:
    - name: MODEL_PATH
      value: "/models/model.pth"
    volumeMounts:
    - name: model-storage
      mountPath: /models
      readOnly: true

    # Readiness probe to ensure model loaded
    readinessProbe:
      httpGet:
        path: /health
        port: 8080
      initialDelaySeconds: 30
      periodSeconds: 10

  volumes:
  - name: model-storage
    persistentVolumeClaim:
      claimName: model-pvc
```

### Training Job Stuck or Slow

**Problem:** Training job appears to be stuck or running much slower than expected.

```bash
# Debug steps
# 1. Check if pod is actually running
kubectl get pod training-job-xyz -n ml-training -o wide

# 2. Check resource usage
kubectl top pod training-job-xyz -n ml-training --containers

# 3. Check GPU utilization
kubectl exec training-job-xyz -n ml-training -- nvidia-smi

# 4. Get training logs
kubectl logs training-job-xyz -n ml-training -f | tail -100

# 5. Check if waiting on I/O
kubectl exec training-job-xyz -n ml-training -- iostat -x 1 5

# 6. Check network usage (for distributed training)
kubectl exec training-job-xyz -n ml-training -- iftop -i eth0

# 7. Profile the process
kubectl exec training-job-xyz -n ml-training -- ps aux | grep python
kubectl exec training-job-xyz -n ml-training -- top -b -n 1

# 8. Check for deadlocks
kubectl exec -it training-job-xyz -n ml-training -- python -c \
  "import sys; import traceback; import threading; \
   print('\\n'.join([str(x) for x in threading.enumerate()]))"
```

**Solutions:**

```yaml
# Add monitoring and debugging capabilities
apiVersion: batch/v1
kind: Job
metadata:
  name: training-job
spec:
  template:
    spec:
      containers:
      - name: trainer
        image: pytorch-training:latest
        env:
        # Enable PyTorch profiling
        - name: TORCH_PROFILER_ENABLE
          value: "true"
        # Enable NCCL debugging for distributed training
        - name: NCCL_DEBUG
          value: "INFO"
        # Set reasonable timeout
        - name: NCCL_SOCKET_TIMEOUT
          value: "600"
        # Enable progress logging
        - name: LOG_INTERVAL
          value: "10"

        # Add sidecar for monitoring
      - name: monitor
        image: nicolaka/netshoot
        command:
        - /bin/bash
        - -c
        - |
          while true; do
            echo "=== $(date) ==="
            nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader
            sleep 30
          done
```

---

## Advanced Debugging Techniques

### Using kubectl patch for Quick Fixes

```bash
# Patch deployment to change image
kubectl patch deployment training-job -n ml-training -p \
  '{"spec":{"template":{"spec":{"containers":[{"name":"trainer","image":"new-image:latest"}]}}}}'

# Patch to add environment variable
kubectl patch deployment training-job -n ml-training --type=json -p='[
  {"op": "add", "path": "/spec/template/spec/containers/0/env/-", "value": {"name": "DEBUG", "value": "true"}}
]'

# Patch to remove liveness probe
kubectl patch deployment training-job -n ml-training --type=json -p='[
  {"op": "remove", "path": "/spec/template/spec/containers/0/livenessProbe"}
]'

# Patch to scale replicas
kubectl patch deployment training-job -n ml-training -p '{"spec":{"replicas":3}}'
```

### Debugging with Labels and Selectors

```bash
# Get all resources with label
kubectl get all -n ml-training -l experiment=bert-large

# Add label to running pod
kubectl label pod training-xyz -n ml-training debug=true

# Remove label
kubectl label pod training-xyz -n ml-training debug-

# Update label
kubectl label pod training-xyz -n ml-training experiment=bert-large-v2 --overwrite

# Get pods not matching label
kubectl get pods -n ml-training -l '!debug'
```

### Advanced Log Analysis

```bash
# Get logs from all pods with label
kubectl logs -n ml-training -l app=training --all-containers=true

# Get logs from multiple pods
for pod in $(kubectl get pods -n ml-training -l app=training -o name); do
  echo "=== $pod ==="
  kubectl logs $pod -n ml-training --tail=20
done

# Search logs for error
kubectl logs <pod-name> -n ml-training | grep -i error

# Get logs with context
kubectl logs <pod-name> -n ml-training | grep -B 5 -A 5 "OOM"

# Monitor logs from multiple pods
kubectl logs -f -n ml-training -l app=training --max-log-requests=10
```

---

## Best Practices

### 1. Always Set Resource Limits

```yaml
# Good: Proper resource configuration
resources:
  requests:
    cpu: "2"
    memory: "8Gi"
  limits:
    cpu: "4"
    memory: "16Gi"

# Bad: No limits (can cause node instability)
# resources: {}
```

### 2. Use Meaningful Labels

```yaml
metadata:
  labels:
    app: bert-training
    version: v2.0
    environment: production
    experiment: fine-tuning-squad
    owner: ml-team
```

### 3. Implement Health Checks

```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8080
  initialDelaySeconds: 60
  periodSeconds: 30
  timeoutSeconds: 5
  failureThreshold: 3

readinessProbe:
  httpGet:
    path: /ready
    port: 8080
  initialDelaySeconds: 30
  periodSeconds: 10
```

### 4. Use Init Containers for Validation

```yaml
initContainers:
- name: validate-environment
  image: busybox
  command:
  - sh
  - -c
  - |
    echo "Validating environment..."
    test -f /data/dataset.tar.gz || exit 1
    echo "Validation complete"
  volumeMounts:
  - name: data
    mountPath: /data
```

### 5. Enable Logging and Monitoring

```yaml
containers:
- name: trainer
  env:
  - name: LOG_LEVEL
    value: "INFO"
  - name: METRICS_PORT
    value: "8080"
```

### 6. Use Namespaces for Organization

```bash
# Development
kubectl create namespace ml-dev

# Staging
kubectl create namespace ml-staging

# Production
kubectl create namespace ml-prod
```

### 7. Document Issues with Annotations

```yaml
metadata:
  annotations:
    debug.issue: "OOM on large batch sizes"
    debug.resolution: "Increased memory limit to 32Gi"
    debug.date: "2024-01-15"
```

---

## Conclusion

This implementation guide has covered:

1. **Common Kubernetes Issues**: ImagePullBackOff, CrashLoopBackOff, Pending, OOMKilled
2. **kubectl Debugging**: Essential commands for inspection, logs, and interactive debugging
3. **Pod Troubleshooting**: Systematic workflow from identification to resolution
4. **Network Debugging**: Service connectivity, DNS, NetworkPolicies, and Ingress
5. **Resource Constraints**: Understanding and debugging CPU, memory, and GPU issues
6. **Events and Monitoring**: Using events and setting up monitoring
7. **Production Toolkit**: Debug pods, ephemeral containers, and advanced techniques
8. **ML-Specific Debugging**: OOM in training, GPU allocation, model loading, stuck jobs

### Next Steps

1. Practice with the 6 debugging scenarios in this exercise
2. Create your own debug pod with your preferred tools
3. Set up monitoring for your ML workloads
4. Document common issues and solutions for your team
5. Automate debugging workflows with scripts

### Additional Resources

- [Kubernetes Debugging Documentation](https://kubernetes.io/docs/tasks/debug/)
- [kubectl Cheat Sheet](https://kubernetes.io/docs/reference/kubectl/cheatsheet/)
- [NVIDIA GPU Operator Documentation](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/)
- [Troubleshooting Guide](./docs/TROUBLESHOOTING_GUIDE.md)
- [Step-by-Step Guide](./STEP_BY_STEP.md)

---

**Remember:** Systematic debugging is a skill that improves with practice. Start with the basics, follow the workflows, and build your debugging toolkit over time.
