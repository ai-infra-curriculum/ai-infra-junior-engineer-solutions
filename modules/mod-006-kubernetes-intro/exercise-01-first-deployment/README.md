# Exercise 01: First Kubernetes Deployment - Solution

**Module**: MOD-006 Kubernetes Introduction
**Exercise**: 01 of 07
**Difficulty**: Beginner
**Estimated Time**: 2-3 hours

---

## Overview

This solution provides a complete, production-ready implementation of a first Kubernetes deployment exercise. You'll learn to deploy applications, expose them with Services, scale, perform rolling updates, and troubleshoot common issues.

### Learning Objectives

By completing this exercise, you will master:

✅ Creating and managing Kubernetes Deployments
✅ Exposing applications using ClusterIP and NodePort Services
✅ Scaling applications horizontally
✅ Performing zero-downtime rolling updates
✅ Rolling back failed deployments
✅ Using ConfigMaps for configuration
✅ Setting resource requests and limits
✅ Debugging common Kubernetes issues
✅ Using kubectl effectively

---

## Quick Start

### Automated Deployment (Recommended)

```bash
# 1. Setup environment
./scripts/setup.sh

# 2. Deploy all resources
./scripts/deploy.sh

# 3. Test deployments
./scripts/test.sh

# 4. Access the applications
curl http://localhost:30081  # Custom nginx with HTML

# 5. Cleanup when done
./scripts/cleanup.sh
```

### Manual Step-by-Step

See `STEP_BY_STEP.md` for detailed manual instructions.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Kubernetes Cluster                        │
│                                                               │
│  ┌────────────────────────────────────────────────────────┐ │
│  │           Namespace: exercise-01                        │ │
│  │                                                          │ │
│  │  ┌─────────────────────────────────────────────────┐   │ │
│  │  │   Deployment: nginx-web (replicas: 2)          │   │ │
│  │  │   ├─ Pod: nginx-web-xxxxx-yyyyy                 │   │ │
│  │  │   └─ Pod: nginx-web-xxxxx-zzzzz                 │   │ │
│  │  └─────────────────────────────────────────────────┘   │ │
│  │                        ↑                                 │ │
│  │  ┌─────────────────────┴───────────────────────────┐   │ │
│  │  │   Service: nginx-service (ClusterIP)            │   │ │
│  │  │   IP: 10.96.x.x  Port: 80                       │   │ │
│  │  └─────────────────────────────────────────────────┘   │ │
│  │                                                          │ │
│  │  ┌─────────────────────────────────────────────────┐   │ │
│  │  │   Deployment: nginx-custom (replicas: 2)        │   │ │
│  │  │   ├─ Pod: nginx-custom-xxxxx-yyyyy              │   │ │
│  │  │   │   └─ Volume: nginx-html (ConfigMap)         │   │ │
│  │  │   └─ Pod: nginx-custom-xxxxx-zzzzz              │   │ │
│  │  │       └─ Volume: nginx-html (ConfigMap)         │   │ │
│  │  └─────────────────────────────────────────────────┘   │ │
│  │                        ↑                                 │ │
│  │  ┌─────────────────────┴───────────────────────────┐   │ │
│  │  │   Service: nginx-custom-service (NodePort)      │   │ │
│  │  │   NodePort: 30081  Port: 80                     │   │ │
│  │  └─────────────────────────────────────────────────┘   │ │
│  │                                                          │ │
│  │  ┌─────────────────────────────────────────────────┐   │ │
│  │  │   ConfigMap: nginx-html                         │   │ │
│  │  │   Data: index.html (custom HTML)                │   │ │
│  │  └─────────────────────────────────────────────────┘   │ │
│  └──────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────┘

External Access:
  └─> http://localhost:30081 → nginx-custom-service → Pods
```

---

## Directory Structure

```
exercise-01-first-deployment/
├── README.md                              # This file
├── STEP_BY_STEP.md                        # Detailed manual guide
├── manifests/                             # Kubernetes YAML files
│   ├── 01-nginx-deployment.yaml          # Basic nginx deployment
│   ├── 02-nginx-service-clusterip.yaml   # ClusterIP service
│   ├── 03-nginx-service-nodeport.yaml    # NodePort service
│   ├── 04-nginx-custom-html-configmap.yaml # ConfigMap with custom HTML
│   ├── 05-nginx-with-configmap.yaml      # Deployment using ConfigMap
│   ├── 06-broken-deployment.yaml         # For troubleshooting practice
│   └── 07-resource-test-pod.yaml         # Resource limits testing
├── scripts/                               # Automation scripts
│   ├── setup.sh                           # Environment setup
│   ├── deploy.sh                          # Deploy all resources
│   ├── test.sh                            # Validate deployments
│   └── cleanup.sh                         # Remove all resources
└── docs/                                  # Additional documentation
    ├── kubectl-cheatsheet.md              # kubectl command reference
    └── troubleshooting.md                 # Common issues and solutions
```

---

## Prerequisites

### Required Software

- **kubectl** (v1.24+) - Kubernetes command-line tool
- **Kubernetes cluster** - One of:
  - Docker Desktop with Kubernetes enabled (macOS/Windows)
  - Minikube (all platforms)
  - Kind (Kubernetes in Docker)
  - Cloud cluster (GKE, EKS, AKS)

### Verify Installation

```bash
# Check kubectl version
kubectl version --short

# Check cluster connection
kubectl cluster-info

# Check nodes are ready
kubectl get nodes
```

Expected output:
```
NAME             STATUS   ROLES           AGE   VERSION
docker-desktop   Ready    control-plane   10d   v1.25.2
```

---

## Deployment Guide

### 1. Environment Setup

Run the setup script to create the namespace and verify prerequisites:

```bash
./scripts/setup.sh
```

This script will:
- ✅ Verify kubectl is installed and configured
- ✅ Check cluster connectivity
- ✅ Create `exercise-01` namespace
- ✅ Set namespace as default for current context
- ✅ Check metrics-server availability (optional)

---

### 2. Deploy Resources

Deploy all Kubernetes resources:

```bash
./scripts/deploy.sh
```

This deploys:
1. **nginx-web deployment** (2 replicas)
2. **nginx-service** (ClusterIP)
3. **nginx-html ConfigMap** (custom HTML)
4. **nginx-custom deployment** (2 replicas with ConfigMap)
5. **nginx-custom-service** (NodePort on 30081)

---

### 3. Verify Deployments

Run the test suite to validate everything is working:

```bash
./scripts/test.sh
```

Tests validate:
- ✅ Deployments exist and are ready
- ✅ Pods are running
- ✅ Services exist with correct types
- ✅ Endpoints are populated
- ✅ ConfigMap is created
- ✅ HTTP connectivity works
- ✅ Health probes are configured
- ✅ Resource limits are set

---

### 4. Access the Applications

#### nginx-web (ClusterIP Service)

Access via port-forward:

```bash
# Terminal 1: Start port-forward
kubectl port-forward deployment/nginx-web 8080:80

# Terminal 2: Test
curl http://localhost:8080
```

Or create a test pod inside the cluster:

```bash
kubectl run test-pod --rm -it --image=busybox -- sh
# Inside pod:
wget -qO- http://nginx-service
exit
```

#### nginx-custom (NodePort Service)

Access directly via NodePort:

```bash
# Using curl
curl http://localhost:30081

# Or open in browser
open http://localhost:30081  # macOS
xdg-open http://localhost:30081  # Linux
start http://localhost:30081  # Windows
```

You should see a custom HTML page with styling!

---

## Key Operations

### Scaling

```bash
# Scale up to 5 replicas
kubectl scale deployment nginx-web --replicas=5

# Watch pods being created
kubectl get pods -w

# Verify
kubectl get deployment nginx-web

# Scale down to 1
kubectl scale deployment nginx-web --replicas=1
```

---

### Rolling Updates

```bash
# Update image
kubectl set image deployment/nginx-web nginx=nginx:1.22

# Watch the rollout
kubectl rollout status deployment/nginx-web

# Check rollout history
kubectl rollout history deployment/nginx-web

# Rollback if needed
kubectl rollout undo deployment/nginx-web
```

---

### Troubleshooting

Deploy the broken deployment for practice:

```bash
# Deploy broken app
kubectl apply -f manifests/06-broken-deployment.yaml

# Watch it fail
kubectl get pods -l app=broken

# Expected: ImagePullBackOff

# Debug
kubectl describe pod <pod-name>
# Look for: "Failed to pull image"

# Fix by editing the manifest
# Change: image: nginx:invalid-tag
# To: image: nginx:1.21
kubectl apply -f manifests/06-broken-deployment.yaml

# Cleanup
kubectl delete deployment broken-app
```

---

### Resource Testing

Test memory limits:

```bash
# Deploy stress test pod
kubectl apply -f manifests/07-resource-test-pod.yaml

# Watch (may get OOMKilled)
kubectl get pod resource-test -w

# Check events
kubectl describe pod resource-test

# If OOMKilled, you'll see: "Memory cgroup out of memory"

# Cleanup
kubectl delete pod resource-test
```

---

## Common kubectl Commands

### Viewing Resources

```bash
# Get all resources
kubectl get all

# Get deployments
kubectl get deployments
kubectl get deploy  # Short form

# Get pods with more details
kubectl get pods -o wide

# Get services
kubectl get services
kubectl get svc  # Short form

# Watch resources (auto-refresh)
kubectl get pods -w
```

### Describing Resources

```bash
# Deployment details
kubectl describe deployment nginx-web

# Pod details (includes events)
kubectl describe pod <pod-name>

# Service details (shows endpoints)
kubectl describe service nginx-service
```

### Logs and Debugging

```bash
# View pod logs
kubectl logs <pod-name>

# Follow logs (like tail -f)
kubectl logs <pod-name> -f

# Previous container logs (if crashed)
kubectl logs <pod-name> --previous

# Logs from all pods with label
kubectl logs -l app=nginx

# Execute command in pod
kubectl exec <pod-name> -- ls /usr/share/nginx/html
kubectl exec -it <pod-name> -- /bin/bash  # Interactive shell
```

### Editing Resources

```bash
# Edit deployment (opens editor)
kubectl edit deployment nginx-web

# Edit service
kubectl edit service nginx-service

# Patch deployment (quick update)
kubectl patch deployment nginx-web -p '{"spec":{"replicas":5}}'
```

---

## Cleanup

### Remove All Resources

```bash
./scripts/cleanup.sh
```

This will:
1. Show current resources
2. Ask for confirmation
3. Delete deployments, services, config maps
4. Optionally delete the namespace
5. Switch back to default namespace

### Manual Cleanup

```bash
# Delete specific resource
kubectl delete deployment nginx-web

# Delete all resources in namespace
kubectl delete all --all -n exercise-01

# Delete namespace (removes everything)
kubectl delete namespace exercise-01
```

---

## Troubleshooting Guide

### Issue: "error: error validating..."

**Cause**: YAML syntax error

**Solution**:
```bash
# Validate YAML before applying
kubectl apply -f manifest.yaml --dry-run=client
```

### Issue: "error: the server doesn't have a resource type 'deployments'"

**Cause**: Wrong API version or typo

**Solution**: Check `apiVersion` in YAML. For Deployments, use `apps/v1`.

### Issue: Pods stuck in "Pending"

**Cause**: Not enough cluster resources

**Solution**:
```bash
# Check node resources
kubectl describe nodes

# Check pod events
kubectl describe pod <pod-name>

# Look for: "Insufficient cpu" or "Insufficient memory"
```

### Issue: "ImagePullBackOff"

**Cause**: Invalid image name or tag

**Solution**:
```bash
# Check image name in deployment
kubectl get deployment nginx-web -o jsonpath='{.spec.template.spec.containers[0].image}'

# Describe pod to see exact error
kubectl describe pod <pod-name>

# Fix image name and reapply
kubectl set image deployment/nginx-web nginx=nginx:1.21
```

### Issue: Service has no endpoints

**Cause**: Label selector mismatch

**Solution**:
```bash
# Check service selector
kubectl get service nginx-service -o jsonpath='{.spec.selector}'

# Check pod labels
kubectl get pods --show-labels

# Ensure they match!
```

### Issue: "Connection refused" when accessing NodePort

**Cause**: May need to use node IP instead of localhost

**Solution**:
```bash
# For Docker Desktop/Minikube, use localhost
curl http://localhost:30081

# For cloud clusters, get node external IP
NODE_IP=$(kubectl get nodes -o jsonpath='{.items[0].status.addresses[?(@.type=="ExternalIP")].address}')
curl http://$NODE_IP:30081
```

---

## Learning Checkpoints

After completing this exercise, you should be able to answer:

### Architecture
- [ ] What is the relationship between Deployment, ReplicaSet, and Pods?
- [ ] How does a Service discover and load balance to Pods?
- [ ] What's the difference between ClusterIP and NodePort?

### Operations
- [ ] How do you scale a Deployment?
- [ ] What happens during a rolling update?
- [ ] How do you rollback a failed deployment?

### Debugging
- [ ] How do you find out why a Pod is failing?
- [ ] Where can you see events for a resource?
- [ ] How do you view logs from a crashed container?

### Configuration
- [ ] How do you inject configuration into Pods?
- [ ] What are resource requests vs limits?
- [ ] How do you set environment variables?

---

## Next Steps

After mastering this exercise:

1. **Exercise 02**: Create a Helm Chart for the nginx deployment
2. **Exercise 03**: Learn advanced debugging techniques
3. **Exercise 04**: Deploy stateful applications with StatefulSets
4. **Exercise 05**: Manage configuration with ConfigMaps and Secrets
5. **Exercise 06**: Set up Ingress for HTTP routing
6. **Exercise 07**: Deploy ML workloads on Kubernetes

---

## Additional Resources

### Official Documentation
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [kubectl Cheat Sheet](https://kubernetes.io/docs/reference/kubectl/cheatsheet/)
- [Kubernetes Concepts](https://kubernetes.io/docs/concepts/)

### Interactive Learning
- [Play with Kubernetes](https://labs.play-with-k8s.com/)
- [Kubernetes Tutorial](https://kubernetes.io/docs/tutorials/)

### Books
- *Kubernetes Up & Running* by Kelsey Hightower
- *The Kubernetes Book* by Nigel Poulton

---

## Success Criteria

✅ All automated tests pass (`./scripts/test.sh`)
✅ Can access both ClusterIP and NodePort services
✅ Successfully scaled deployment up and down
✅ Performed rolling update and rollback
✅ Debugged the broken deployment
✅ Understood Pod, Deployment, Service relationship
✅ Comfortable with kubectl commands

---

## Feedback

If you encounter issues or have suggestions for improving this exercise:

1. Check the troubleshooting guide above
2. Review common issues in `docs/troubleshooting.md`
3. Ensure you're using the correct Kubernetes version
4. Try the automated scripts if manual steps fail

---

**Congratulations!** 🎉

You've completed your first Kubernetes deployment. You now understand the core concepts and workflows for deploying, scaling, updating, and troubleshooting applications on Kubernetes.

These skills form the foundation for all future Kubernetes work in ML infrastructure!
