# Implementation Guide: First Kubernetes Deployment

**Module**: MOD-006 Kubernetes Introduction
**Exercise**: 01 - First Kubernetes Deployment
**Focus**: ML Inference Service Deployment
**Difficulty**: Beginner to Intermediate
**Estimated Time**: 3-4 hours

---

## Table of Contents

1. [Overview](#overview)
2. [Learning Objectives](#learning-objectives)
3. [Prerequisites](#prerequisites)
4. [Part 1: Kubernetes Cluster Setup](#part-1-kubernetes-cluster-setup)
5. [Part 2: First Deployment Creation](#part-2-first-deployment-creation)
6. [Part 3: Service Exposure](#part-3-service-exposure)
7. [Part 4: Scaling Deployments](#part-4-scaling-deployments)
8. [Part 5: Rolling Updates and Rollbacks](#part-5-rolling-updates-and-rollbacks)
9. [Part 6: Resource Management](#part-6-resource-management)
10. [Part 7: Production ML Model Deployment](#part-7-production-ml-model-deployment)
11. [Troubleshooting](#troubleshooting)
12. [Best Practices](#best-practices)
13. [Summary](#summary)

---

## Overview

This implementation guide provides a comprehensive, step-by-step walkthrough for deploying your first application on Kubernetes with a focus on ML inference services. You'll learn the fundamentals of Kubernetes deployments, services, scaling, and production-ready configurations.

### What You'll Build

By the end of this guide, you'll have deployed:

1. A basic web application to understand Kubernetes fundamentals
2. An ML inference service with proper resource management
3. Auto-scaling configurations for production workloads
4. Zero-downtime update mechanisms for ML models

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      Kubernetes Cluster                          │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Namespace: exercise-01                                   │  │
│  │                                                            │  │
│  │  ┌────────────────────────────────────────────────────┐  │  │
│  │  │  Deployment: nginx-web (Learning)                  │  │  │
│  │  │  ├─ ReplicaSet                                      │  │  │
│  │  │  │  ├─ Pod 1 (nginx:1.21)                          │  │  │
│  │  │  │  └─ Pod 2 (nginx:1.21)                          │  │  │
│  │  └────────────────────────────────────────────────────┘  │  │
│  │                          ↑                                 │  │
│  │  ┌───────────────────────┴─────────────────────────────┐ │  │
│  │  │  Service: nginx-service (ClusterIP/NodePort)        │ │  │
│  │  └─────────────────────────────────────────────────────┘ │  │
│  │                                                            │  │
│  │  ┌────────────────────────────────────────────────────┐  │  │
│  │  │  Deployment: ml-inference (Production)             │  │  │
│  │  │  ├─ ReplicaSet                                      │  │  │
│  │  │  │  ├─ Pod 1 (sklearn-model:v1)                    │  │  │
│  │  │  │  ├─ Pod 2 (sklearn-model:v1)                    │  │  │
│  │  │  │  └─ Pod 3 (sklearn-model:v1)                    │  │  │
│  │  │  └─ HorizontalPodAutoscaler (2-10 replicas)        │  │  │
│  │  └────────────────────────────────────────────────────┘  │  │
│  │                          ↑                                 │  │
│  │  ┌───────────────────────┴─────────────────────────────┐ │  │
│  │  │  Service: ml-service (LoadBalancer)                 │ │  │
│  │  └─────────────────────────────────────────────────────┘ │  │
│  └──────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────┘
```

---

## Learning Objectives

After completing this guide, you will be able to:

- Set up a local Kubernetes cluster using minikube or kind
- Create and manage Kubernetes Deployments with YAML manifests
- Expose applications using ClusterIP, NodePort, and LoadBalancer services
- Scale applications horizontally using manual and automatic scaling
- Perform zero-downtime rolling updates and rollbacks
- Configure resource requests and limits for predictable performance
- Deploy production ML inference services with proper configurations
- Implement health checks and readiness probes for ML models
- Monitor and troubleshoot Kubernetes deployments

---

## Prerequisites

### Required Knowledge

- Basic understanding of Docker and containers
- Familiarity with YAML syntax
- Basic command-line experience
- Understanding of HTTP/REST APIs

### Required Software

- **kubectl** (v1.24+) - Kubernetes CLI tool
- **Docker Desktop** OR **minikube** OR **kind** - Local Kubernetes cluster
- **curl** - For testing HTTP endpoints
- **Git** - For accessing exercise files

### Verification Commands

```bash
# Check kubectl installation
kubectl version --client

# Expected output:
# Client Version: v1.28.x

# Check Docker (if using Docker Desktop)
docker version

# Check minikube (if using minikube)
minikube version

# Check kind (if using kind)
kind version
```

---

## Part 1: Kubernetes Cluster Setup

### Option A: Docker Desktop (Recommended for macOS/Windows)

Docker Desktop includes a single-node Kubernetes cluster that's perfect for learning.

#### Step 1: Enable Kubernetes in Docker Desktop

1. Open Docker Desktop
2. Navigate to Settings → Kubernetes
3. Check "Enable Kubernetes"
4. Click "Apply & Restart"
5. Wait for the Kubernetes icon to show green

#### Step 2: Verify Installation

```bash
# Check cluster info
kubectl cluster-info

# Expected output:
# Kubernetes control plane is running at https://kubernetes.docker.internal:6443
# CoreDNS is running at https://kubernetes.docker.internal:6443/api/v1/namespaces/kube-system/services/kube-dns:dns/proxy

# Check nodes
kubectl get nodes

# Expected output:
# NAME             STATUS   ROLES           AGE   VERSION
# docker-desktop   Ready    control-plane   10d   v1.28.2
```

### Option B: Minikube (Cross-Platform)

Minikube creates a local Kubernetes cluster in a virtual machine.

#### Step 1: Install Minikube

```bash
# macOS
brew install minikube

# Linux
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube

# Windows (PowerShell as Admin)
choco install minikube
```

#### Step 2: Start Minikube Cluster

```bash
# Start cluster with adequate resources for ML workloads
minikube start \
  --cpus=4 \
  --memory=8192 \
  --disk-size=20g \
  --driver=docker

# Enable metrics-server for autoscaling
minikube addons enable metrics-server

# Verify installation
kubectl get nodes
```

### Option C: kind (Kubernetes in Docker)

kind runs Kubernetes clusters using Docker containers as nodes.

#### Step 1: Install kind

```bash
# macOS
brew install kind

# Linux
curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.20.0/kind-linux-amd64
chmod +x ./kind
sudo mv ./kind /usr/local/bin/kind

# Windows (PowerShell)
choco install kind
```

#### Step 2: Create Cluster Configuration

Create a file called `kind-config.yaml`:

```yaml
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
- role: control-plane
  extraPortMappings:
  - containerPort: 30080
    hostPort: 30080
    protocol: TCP
  - containerPort: 30081
    hostPort: 30081
    protocol: TCP
- role: worker
- role: worker
```

#### Step 3: Create Cluster

```bash
# Create cluster
kind create cluster --name ml-cluster --config kind-config.yaml

# Verify
kubectl cluster-info --context kind-ml-cluster
kubectl get nodes
```

### Post-Setup: Create Namespace

Regardless of which option you chose, create a dedicated namespace for this exercise:

```bash
# Create namespace
kubectl create namespace exercise-01

# Set as default for current context
kubectl config set-context --current --namespace=exercise-01

# Verify current namespace
kubectl config view --minify | grep namespace:

# Expected output:
# namespace: exercise-01
```

---

## Part 2: First Deployment Creation

### Understanding Kubernetes Deployments

A Deployment provides declarative updates for Pods and ReplicaSets. It manages:
- Desired state for your application
- Rolling updates and rollbacks
- Scaling replicas
- Self-healing of failed pods

### Step 1: Create Basic YAML Manifest

Create a file called `01-nginx-deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx-web
  labels:
    app: nginx
    tier: frontend
    exercise: first-deployment
spec:
  replicas: 2
  selector:
    matchLabels:
      app: nginx
  template:
    metadata:
      labels:
        app: nginx
        tier: frontend
        version: v1
    spec:
      containers:
      - name: nginx
        image: nginx:1.21
        ports:
        - containerPort: 80
          name: http
          protocol: TCP
        resources:
          requests:
            memory: "64Mi"
            cpu: "100m"
          limits:
            memory: "128Mi"
            cpu: "200m"
        livenessProbe:
          httpGet:
            path: /
            port: 80
          initialDelaySeconds: 10
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /
            port: 80
          initialDelaySeconds: 5
          periodSeconds: 3
          timeoutSeconds: 2
          failureThreshold: 3
        env:
        - name: POD_NAME
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
        - name: POD_NAMESPACE
          valueFrom:
            fieldRef:
              fieldPath: metadata.namespace
        - name: POD_IP
          valueFrom:
            fieldRef:
              fieldPath: status.podIP
```

### Step 2: Understanding the Manifest

Let's break down each section:

#### Metadata Section
```yaml
metadata:
  name: nginx-web          # Deployment name
  labels:
    app: nginx             # Used for grouping and selection
    tier: frontend         # Organizational label
```

#### Spec Section
```yaml
spec:
  replicas: 2              # Number of pod copies
  selector:
    matchLabels:
      app: nginx           # Pods to manage (must match template labels)
```

#### Template Section
```yaml
template:
  spec:
    containers:
    - name: nginx
      image: nginx:1.21    # Container image
      ports:
      - containerPort: 80  # Port the container exposes
```

#### Resource Management
```yaml
resources:
  requests:                # Guaranteed resources
    memory: "64Mi"
    cpu: "100m"            # 0.1 CPU cores
  limits:                  # Maximum allowed
    memory: "128Mi"
    cpu: "200m"
```

#### Health Checks
```yaml
livenessProbe:             # Restart if fails
  httpGet:
    path: /
    port: 80
  initialDelaySeconds: 10  # Wait before first check
  periodSeconds: 5         # Check every 5 seconds

readinessProbe:            # Remove from service if fails
  httpGet:
    path: /
    port: 80
  initialDelaySeconds: 5
  periodSeconds: 3
```

### Step 3: Deploy the Application

```bash
# Apply the deployment
kubectl apply -f 01-nginx-deployment.yaml

# Expected output:
# deployment.apps/nginx-web created

# Watch the deployment progress
kubectl get deployments -w

# Press Ctrl+C after seeing READY 2/2

# Check pods
kubectl get pods

# Expected output:
# NAME                         READY   STATUS    RESTARTS   AGE
# nginx-web-xxxxxxxxx-xxxxx    1/1     Running   0          30s
# nginx-web-xxxxxxxxx-xxxxx    1/1     Running   0          30s

# Check replica sets
kubectl get replicasets

# Expected output:
# NAME                   DESIRED   CURRENT   READY   AGE
# nginx-web-xxxxxxxxx    2         2         2       1m
```

### Step 4: Inspect the Deployment

```bash
# View detailed deployment information
kubectl describe deployment nginx-web

# Key sections to observe:
# - Replicas: 2 desired | 2 updated | 2 total | 2 available
# - Conditions: Available (True), Progressing (True)
# - Events: ScalingReplicaSet, ReplicaSetUpdate

# View deployment YAML (includes runtime status)
kubectl get deployment nginx-web -o yaml | less

# Get specific information using JSONPath
kubectl get deployment nginx-web -o jsonpath='{.spec.replicas}'
# Output: 2

kubectl get deployment nginx-web -o jsonpath='{.spec.template.spec.containers[0].image}'
# Output: nginx:1.21
```

### Step 5: Inspect Pods

```bash
# Get pod details with IP addresses
kubectl get pods -o wide

# Describe a specific pod (replace <pod-name> with actual name)
POD_NAME=$(kubectl get pods -l app=nginx -o jsonpath='{.items[0].metadata.name}')
kubectl describe pod $POD_NAME

# Key sections to observe:
# - Status: Running
# - IP: 10.x.x.x
# - Containers: nginx (Running)
# - Events: Scheduled, Pulling, Pulled, Created, Started

# View pod logs
kubectl logs $POD_NAME

# Expected output: nginx access logs (may be empty initially)
```

### Step 6: Test Pod Connectivity

```bash
# Option 1: Port-forward to access the pod
kubectl port-forward deployment/nginx-web 8080:80

# In another terminal:
curl http://localhost:8080
# Expected: nginx welcome page HTML

# Stop port-forward with Ctrl+C

# Option 2: Create a temporary pod to test from inside the cluster
kubectl run test-pod --rm -it --image=busybox --restart=Never -- sh

# Inside the test pod:
wget -qO- http://<pod-ip>:80
# (Replace <pod-ip> with actual pod IP from kubectl get pods -o wide)
exit
```

---

## Part 3: Service Exposure

Services provide stable network endpoints to access pods. Kubernetes offers several service types.

### Understanding Service Types

1. **ClusterIP** (default): Internal cluster access only
2. **NodePort**: Exposes service on each node's IP at a static port
3. **LoadBalancer**: Cloud provider's load balancer (external IP)
4. **ExternalName**: Maps service to DNS name

### Step 1: Create ClusterIP Service

Create a file called `02-nginx-service-clusterip.yaml`:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: nginx-service
  labels:
    app: nginx
spec:
  type: ClusterIP
  selector:
    app: nginx
  ports:
  - name: http
    port: 80
    targetPort: 80
    protocol: TCP
```

Deploy the service:

```bash
# Apply the service
kubectl apply -f 02-nginx-service-clusterip.yaml

# View the service
kubectl get services

# Expected output:
# NAME            TYPE        CLUSTER-IP      EXTERNAL-IP   PORT(S)   AGE
# nginx-service   ClusterIP   10.96.xxx.xxx   <none>        80/TCP    10s

# Describe the service
kubectl describe service nginx-service

# Key information:
# Type: ClusterIP
# IP: 10.96.xxx.xxx
# Endpoints: <pod-ip>:80, <pod-ip>:80

# View endpoints
kubectl get endpoints nginx-service

# Expected output:
# NAME            ENDPOINTS                     AGE
# nginx-service   10.244.0.5:80,10.244.0.6:80   1m
```

### Step 2: Test ClusterIP Service

```bash
# Create test pod
kubectl run test-pod --rm -it --image=busybox --restart=Never -- sh

# Inside the test pod:
# Test with service name (DNS)
wget -qO- http://nginx-service
# Expected: nginx welcome page

# Test with fully qualified domain name
wget -qO- http://nginx-service.exercise-01.svc.cluster.local
# Expected: nginx welcome page

# Test DNS resolution
nslookup nginx-service
# Expected: ClusterIP address

exit
```

### Step 3: Create NodePort Service

Create a file called `03-nginx-service-nodeport.yaml`:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: nginx-service-nodeport
  labels:
    app: nginx
spec:
  type: NodePort
  selector:
    app: nginx
  ports:
  - name: http
    port: 80
    targetPort: 80
    nodePort: 30080
    protocol: TCP
```

Deploy and test:

```bash
# Apply the service
kubectl apply -f 03-nginx-service-nodeport.yaml

# View the service
kubectl get service nginx-service-nodeport

# Expected output:
# NAME                    TYPE       CLUSTER-IP      EXTERNAL-IP   PORT(S)        AGE
# nginx-service-nodeport  NodePort   10.96.xxx.xxx   <none>        80:30080/TCP   10s

# Test access (varies by setup)

# Docker Desktop / kind:
curl http://localhost:30080
# Expected: nginx welcome page

# Minikube:
minikube service nginx-service-nodeport --url
# Use the returned URL in browser or curl

# Cloud provider:
# Get node external IP and use: http://<node-external-ip>:30080
```

### Step 4: Understanding Service Discovery

Services enable pod-to-pod communication through DNS:

```bash
# Format: <service-name>.<namespace>.svc.cluster.local

# Same namespace (shortest form)
http://nginx-service

# With namespace (portable across namespaces)
http://nginx-service.exercise-01

# Fully qualified domain name (FQDN)
http://nginx-service.exercise-01.svc.cluster.local
```

### Step 5: Load Balancing Verification

```bash
# Create a script to test load balancing
for i in {1..10}; do
  kubectl run test-$i --rm -it --image=busybox --restart=Never -- \
    wget -qO- http://nginx-service | grep -i server
done

# Observe: Different pod hostnames indicate load balancing

# Or use a single pod with multiple requests:
kubectl run test-pod --rm -it --image=busybox --restart=Never -- sh

# Inside pod:
for i in 1 2 3 4 5; do
  echo "Request $i:"
  wget -qO- http://nginx-service 2>&1 | head -5
  echo "---"
done
exit
```

---

## Part 4: Scaling Deployments

Kubernetes makes it easy to scale applications horizontally by adding or removing pod replicas.

### Step 1: Manual Scaling - kubectl scale

```bash
# Scale up to 5 replicas
kubectl scale deployment nginx-web --replicas=5

# Expected output:
# deployment.apps/nginx-web scaled

# Watch pods being created
kubectl get pods -w
# Press Ctrl+C after all pods are Running

# Verify the deployment
kubectl get deployment nginx-web

# Expected output:
# NAME        READY   UP-TO-DATE   AVAILABLE   AGE
# nginx-web   5/5     5            5           10m

# Check that service endpoints updated
kubectl get endpoints nginx-service

# Expected: 5 pod IPs listed
```

### Step 2: Scale Down

```bash
# Scale down to 1 replica
kubectl scale deployment nginx-web --replicas=1

# Watch pods being terminated
kubectl get pods -w

# Verify
kubectl get pods
kubectl get endpoints nginx-service
# Expected: 1 pod IP
```

### Step 3: Declarative Scaling - Update YAML

Edit `01-nginx-deployment.yaml`:

```yaml
spec:
  replicas: 3  # Changed from 2
```

Apply the change:

```bash
# Apply updated manifest
kubectl apply -f 01-nginx-deployment.yaml

# Expected output:
# deployment.apps/nginx-web configured

# Verify
kubectl get deployment nginx-web
# READY should show 3/3
```

### Step 4: Understanding Scaling Behavior

```bash
# Watch scaling in real-time
kubectl get pods -w &
WATCH_PID=$!

# Scale up
kubectl scale deployment nginx-web --replicas=8

# Wait 30 seconds, then scale down
sleep 30
kubectl scale deployment nginx-web --replicas=3

# Stop watching
kill $WATCH_PID

# Observations:
# - New pods start in ContainerCreating → Running
# - Terminated pods go to Terminating → removed
# - Service automatically adds/removes endpoints
# - No downtime during scaling operations
```

### Step 5: Test Service During Scaling

```bash
# Terminal 1: Continuous requests
while true; do
  curl -s http://localhost:30080 | grep -o "nginx" && echo " - $(date)"
  sleep 1
done

# Terminal 2: Scale operations
kubectl scale deployment nginx-web --replicas=10
sleep 20
kubectl scale deployment nginx-web --replicas=2

# Observation: No failed requests during scaling
# Ctrl+C to stop in both terminals
```

---

## Part 5: Rolling Updates and Rollbacks

Kubernetes performs zero-downtime deployments through rolling updates.

### Step 1: Perform Rolling Update

```bash
# Current image version
kubectl get deployment nginx-web -o jsonpath='{.spec.template.spec.containers[0].image}'
# Output: nginx:1.21

# Update to nginx:1.22
kubectl set image deployment/nginx-web nginx=nginx:1.22

# Expected output:
# deployment.apps/nginx-web image updated

# Watch the rollout
kubectl rollout status deployment/nginx-web

# Expected output:
# Waiting for deployment "nginx-web" rollout to finish: 1 out of 3 new replicas have been updated...
# Waiting for deployment "nginx-web" rollout to finish: 2 out of 3 new replicas have been updated...
# Waiting for deployment "nginx-web" rollout to finish: 1 old replicas are pending termination...
# deployment "nginx-web" successfully rolled out
```

### Step 2: Monitor Rolling Update Process

```bash
# In another terminal, watch pods during update
kubectl get pods -w

# Observations:
# 1. New pods created with updated image
# 2. New pods become Ready
# 3. Old pods are terminated
# 4. Process repeats until all pods updated
# 5. Never more than 1 pod unavailable (default strategy)

# View replica sets
kubectl get replicasets

# Expected: Old and new ReplicaSets
# NAME                   DESIRED   CURRENT   READY   AGE
# nginx-web-xxxxxxxxx    3         3         3       2m    (new)
# nginx-web-yyyyyyyyy    0         0         0       10m   (old)
```

### Step 3: Understanding Deployment Strategy

View the deployment strategy:

```bash
kubectl describe deployment nginx-web | grep -A 5 "StrategyType"

# Default RollingUpdate strategy:
# StrategyType: RollingUpdate
# RollingUpdateStrategy:
#   Max Unavailable: 25%
#   Max Surge:       25%
```

Create a deployment with custom strategy:

```yaml
spec:
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 1  # At most 1 pod unavailable during update
      maxSurge: 1        # At most 1 extra pod during update
```

### Step 4: View Rollout History

```bash
# View rollout history
kubectl rollout history deployment/nginx-web

# Expected output:
# REVISION  CHANGE-CAUSE
# 1         <none>
# 2         <none>

# View specific revision details
kubectl rollout history deployment/nginx-web --revision=2

# Add annotations for better tracking (going forward)
kubectl annotate deployment/nginx-web \
  kubernetes.io/change-cause="Update to nginx:1.22"

# Future rollouts will show the change cause
```

### Step 5: Rollback Deployment

```bash
# Rollback to previous version
kubectl rollout undo deployment/nginx-web

# Expected output:
# deployment.apps/nginx-web rolled back

# Watch the rollback
kubectl rollout status deployment/nginx-web

# Verify image version
kubectl get deployment nginx-web -o jsonpath='{.spec.template.spec.containers[0].image}'
# Output: nginx:1.21 (back to previous)

# Rollback to specific revision
kubectl rollout undo deployment/nginx-web --to-revision=1
```

### Step 6: Pause and Resume Rollouts

```bash
# Update image
kubectl set image deployment/nginx-web nginx=nginx:1.23

# Immediately pause the rollout
kubectl rollout pause deployment/nginx-web

# Check status (some pods updated, some not)
kubectl get pods

# Make additional changes while paused
kubectl set env deployment/nginx-web APP_VERSION=v2

# Resume rollout
kubectl rollout resume deployment/nginx-web

# Watch it complete
kubectl rollout status deployment/nginx-web
```

---

## Part 6: Resource Management

Proper resource management ensures predictable performance and prevents resource contention.

### Understanding Requests and Limits

- **Requests**: Guaranteed resources (used for scheduling)
- **Limits**: Maximum allowed (enforced by container runtime)

```yaml
resources:
  requests:     # Scheduler uses this to place pods
    memory: "64Mi"
    cpu: "100m"
  limits:       # Container killed if exceeded
    memory: "128Mi"
    cpu: "200m"
```

### Step 1: CPU Resource Management

Create `cpu-stress-deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cpu-stress
spec:
  replicas: 1
  selector:
    matchLabels:
      app: cpu-stress
  template:
    metadata:
      labels:
        app: cpu-stress
    spec:
      containers:
      - name: stress
        image: containerstack/cpustress
        resources:
          requests:
            cpu: "500m"
            memory: "128Mi"
          limits:
            cpu: "1000m"
            memory: "256Mi"
        args:
        - --cpu
        - "2"
        - --timeout
        - "300s"
```

Deploy and observe:

```bash
# Deploy
kubectl apply -f cpu-stress-deployment.yaml

# Watch CPU usage (requires metrics-server)
kubectl top pod -l app=cpu-stress

# Expected output:
# NAME                         CPU(cores)   MEMORY(bytes)
# cpu-stress-xxxxxxxxx-xxxxx   500m-1000m   50Mi

# View resource usage over time
watch -n 2 kubectl top pod -l app=cpu-stress

# Cleanup
kubectl delete deployment cpu-stress
```

### Step 2: Memory Resource Management

Create `memory-stress-deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: memory-stress
spec:
  replicas: 1
  selector:
    matchLabels:
      app: memory-stress
  template:
    metadata:
      labels:
        app: memory-stress
    spec:
      containers:
      - name: stress
        image: polinux/stress
        command: ["stress"]
        args:
        - "--vm"
        - "1"
        - "--vm-bytes"
        - "150M"
        - "--vm-hang"
        - "1"
        resources:
          requests:
            memory: "128Mi"
            cpu: "100m"
          limits:
            memory: "200Mi"
            cpu: "200m"
```

Deploy and test:

```bash
# Deploy
kubectl apply -f memory-stress-deployment.yaml

# Watch the pod
kubectl get pod -l app=memory-stress -w

# Check memory usage
kubectl top pod -l app=memory-stress

# Describe pod to see resource allocation
kubectl describe pod -l app=memory-stress | grep -A 5 "Limits\|Requests"

# Cleanup
kubectl delete deployment memory-stress
```

### Step 3: Testing OOMKilled

Create a pod that exceeds memory limits:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: oom-test
spec:
  containers:
  - name: stress
    image: polinux/stress
    command: ["stress"]
    args:
    - "--vm"
    - "1"
    - "--vm-bytes"
    - "300M"
    - "--vm-hang"
    - "1"
    resources:
      requests:
        memory: "128Mi"
      limits:
        memory: "200Mi"
```

Test:

```bash
# Apply
kubectl apply -f oom-test.yaml

# Watch the pod get OOMKilled
kubectl get pod oom-test -w

# Expected: CrashLoopBackOff or OOMKilled

# Check events
kubectl describe pod oom-test | grep -A 10 Events

# Expected event:
# Reason: OOMKilled
# Message: Memory cgroup out of memory

# Cleanup
kubectl delete pod oom-test
```

### Step 4: Quality of Service (QoS) Classes

Kubernetes assigns QoS classes based on resources:

1. **Guaranteed**: requests == limits for all containers
2. **Burstable**: requests < limits
3. **BestEffort**: no requests or limits

```bash
# Create pods with different QoS classes

# Guaranteed QoS
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: qos-guaranteed
spec:
  containers:
  - name: nginx
    image: nginx:1.21
    resources:
      requests:
        memory: "128Mi"
        cpu: "100m"
      limits:
        memory: "128Mi"
        cpu: "100m"
EOF

# Burstable QoS
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: qos-burstable
spec:
  containers:
  - name: nginx
    image: nginx:1.21
    resources:
      requests:
        memory: "64Mi"
        cpu: "100m"
      limits:
        memory: "128Mi"
        cpu: "200m"
EOF

# BestEffort QoS
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: qos-besteffort
spec:
  containers:
  - name: nginx
    image: nginx:1.21
EOF

# Check QoS classes
kubectl get pod qos-guaranteed -o jsonpath='{.status.qosClass}'
# Output: Guaranteed

kubectl get pod qos-burstable -o jsonpath='{.status.qosClass}'
# Output: Burstable

kubectl get pod qos-besteffort -o jsonpath='{.status.qosClass}'
# Output: BestEffort

# Cleanup
kubectl delete pod qos-guaranteed qos-burstable qos-besteffort
```

---

## Part 7: Production ML Model Deployment

Now let's apply everything we've learned to deploy a production ML inference service.

### Step 1: Create ML Model Container

For this example, we'll use a simple scikit-learn model served with Flask. Create `ml-inference-deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-inference
  labels:
    app: ml-inference
    tier: ml-service
    version: v1
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ml-inference
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 1
      maxSurge: 1
  template:
    metadata:
      labels:
        app: ml-inference
        tier: ml-service
        version: v1
    spec:
      containers:
      - name: ml-model
        image: docker.io/seldonio/sklearn-iris:0.3
        ports:
        - containerPort: 9000
          name: http
          protocol: TCP
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health/alive
            port: 9000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 9000
          initialDelaySeconds: 20
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3
        env:
        - name: MODEL_NAME
          value: "iris-classifier"
        - name: SERVICE_TYPE
          value: "MODEL"
        - name: POD_NAME
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
        - name: POD_NAMESPACE
          valueFrom:
            fieldRef:
              fieldPath: metadata.namespace
```

### Step 2: Create ML Service

Create `ml-service.yaml`:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: ml-service
  labels:
    app: ml-inference
spec:
  type: LoadBalancer
  selector:
    app: ml-inference
  ports:
  - name: http
    port: 80
    targetPort: 9000
    protocol: TCP
  sessionAffinity: ClientIP
  sessionAffinityConfig:
    clientIP:
      timeoutSeconds: 3600
```

### Step 3: Deploy ML Service

```bash
# Deploy the ML model
kubectl apply -f ml-inference-deployment.yaml

# Deploy the service
kubectl apply -f ml-service.yaml

# Watch deployment
kubectl rollout status deployment/ml-inference

# Check pods
kubectl get pods -l app=ml-inference

# Expected output:
# NAME                            READY   STATUS    RESTARTS   AGE
# ml-inference-xxxxxxxxx-xxxxx    1/1     Running   0          1m
# ml-inference-xxxxxxxxx-xxxxx    1/1     Running   0          1m
# ml-inference-xxxxxxxxx-xxxxx    1/1     Running   0          1m

# Check service
kubectl get service ml-service

# For cloud providers, wait for EXTERNAL-IP
# For local clusters, use NodePort or port-forward
```

### Step 4: Test ML Inference

```bash
# Option 1: Port-forward (local development)
kubectl port-forward service/ml-service 9000:80

# In another terminal, send prediction request
curl -X POST http://localhost:9000/api/v1.0/predictions \
  -H "Content-Type: application/json" \
  -d '{
    "data": {
      "ndarray": [[5.1, 3.5, 1.4, 0.2]]
    }
  }'

# Expected response:
# {
#   "data": {
#     "names": ["t:0", "t:1", "t:2"],
#     "ndarray": [[0.9, 0.05, 0.05]]
#   },
#   "meta": {}
# }

# Option 2: Using service IP (if LoadBalancer)
SERVICE_IP=$(kubectl get service ml-service -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
curl -X POST http://$SERVICE_IP/api/v1.0/predictions \
  -H "Content-Type: application/json" \
  -d '{
    "data": {
      "ndarray": [[6.2, 3.4, 5.4, 2.3]]
    }
  }'
```

### Step 5: Configure Horizontal Pod Autoscaler (HPA)

Create `ml-hpa.yaml`:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ml-inference-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ml-inference
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100
        periodSeconds: 30
      - type: Pods
        value: 2
        periodSeconds: 30
      selectPolicy: Max
```

Deploy HPA:

```bash
# First, ensure metrics-server is installed
kubectl get deployment metrics-server -n kube-system

# If not installed (minikube):
minikube addons enable metrics-server

# Apply HPA
kubectl apply -f ml-hpa.yaml

# Check HPA status
kubectl get hpa

# Expected output:
# NAME                REFERENCE                 TARGETS         MINPODS   MAXPODS   REPLICAS   AGE
# ml-inference-hpa    Deployment/ml-inference   15%/70%, 20%/80%   2        10        3          1m

# Watch HPA
kubectl get hpa -w
```

### Step 6: Load Testing and Auto-Scaling

```bash
# Create a load generator
kubectl run load-generator \
  --image=busybox \
  --restart=Never \
  --rm -it -- /bin/sh

# Inside the load generator pod:
while true; do
  wget -q -O- http://ml-service/api/v1.0/predictions \
    --post-data='{"data": {"ndarray": [[5.1, 3.5, 1.4, 0.2]]}}' \
    --header='Content-Type: application/json'
  sleep 0.01
done

# In another terminal, watch scaling
watch -n 2 'kubectl get hpa; echo "---"; kubectl get pods -l app=ml-inference'

# Observations:
# 1. CPU usage increases
# 2. HPA triggers scale-up
# 3. New pods created
# 4. Load distributed across pods
# 5. After stopping load, HPA scales down (after stabilization window)

# Stop load generator with Ctrl+C
```

### Step 7: Zero-Downtime Model Updates

```bash
# Simulate model update by changing image tag
kubectl set image deployment/ml-inference \
  ml-model=docker.io/seldonio/sklearn-iris:0.4

# Watch rolling update
kubectl rollout status deployment/ml-inference

# In another terminal, continuous predictions
while true; do
  curl -X POST http://localhost:9000/api/v1.0/predictions \
    -H "Content-Type: application/json" \
    -d '{"data": {"ndarray": [[5.1, 3.5, 1.4, 0.2]]}}' \
    -w "\n" -s | jq '.data.ndarray[0][0]'
  sleep 0.5
done

# Observations:
# - No failed requests during update
# - Gradual transition from old to new pods
# - Readiness probes ensure traffic only goes to ready pods

# If update has issues, rollback immediately
kubectl rollout undo deployment/ml-inference
```

---

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: Pods Stuck in Pending

**Symptoms:**
```bash
kubectl get pods
# NAME                         READY   STATUS    RESTARTS   AGE
# ml-inference-xxx-xxx         0/1     Pending   0          2m
```

**Diagnosis:**
```bash
kubectl describe pod <pod-name> | grep -A 10 Events

# Look for:
# - "Insufficient cpu"
# - "Insufficient memory"
# - "No nodes available"
```

**Solutions:**
```bash
# Check node resources
kubectl top nodes

# Check pod resource requests
kubectl describe pod <pod-name> | grep -A 5 "Requests"

# Solutions:
# 1. Reduce resource requests
# 2. Add more nodes to cluster
# 3. Delete unused pods/deployments
```

#### Issue 2: ImagePullBackOff

**Symptoms:**
```bash
kubectl get pods
# NAME                         READY   STATUS             RESTARTS   AGE
# ml-inference-xxx-xxx         0/1     ImagePullBackOff   0          1m
```

**Diagnosis:**
```bash
kubectl describe pod <pod-name> | grep -A 5 "Events"

# Look for:
# - "Failed to pull image"
# - "manifest unknown"
# - "unauthorized"
```

**Solutions:**
```bash
# Verify image name and tag
kubectl get deployment ml-inference -o jsonpath='{.spec.template.spec.containers[0].image}'

# Test image pull manually
docker pull <image-name>

# Fix: Update deployment with correct image
kubectl set image deployment/ml-inference ml-model=<correct-image>
```

#### Issue 3: CrashLoopBackOff

**Symptoms:**
```bash
kubectl get pods
# NAME                         READY   STATUS              RESTARTS   AGE
# ml-inference-xxx-xxx         0/1     CrashLoopBackOff    5          3m
```

**Diagnosis:**
```bash
# Check current logs
kubectl logs <pod-name>

# Check previous container logs
kubectl logs <pod-name> --previous

# Check events
kubectl describe pod <pod-name>
```

**Common Causes:**
- Application errors
- Missing environment variables
- Misconfigured liveness probe
- OOMKilled (memory limit exceeded)

**Solutions:**
```bash
# Check exit code
kubectl get pod <pod-name> -o jsonpath='{.status.containerStatuses[0].lastState.terminated.exitCode}'

# Exit code 137 = OOMKilled (increase memory limit)
# Exit code 1 = Application error (check logs)

# Debug interactively
kubectl run debug --rm -it --image=<same-image> -- /bin/sh
```

#### Issue 4: Service Has No Endpoints

**Symptoms:**
```bash
kubectl get endpoints ml-service
# NAME         ENDPOINTS   AGE
# ml-service   <none>      5m
```

**Diagnosis:**
```bash
# Check service selector
kubectl get service ml-service -o jsonpath='{.spec.selector}'

# Check pod labels
kubectl get pods --show-labels

# Labels must match!
```

**Solution:**
```bash
# Fix selector in service
kubectl edit service ml-service

# Or fix labels in deployment
kubectl edit deployment ml-inference
```

#### Issue 5: Readiness Probe Failing

**Symptoms:**
```bash
kubectl get pods
# NAME                         READY   STATUS    RESTARTS   AGE
# ml-inference-xxx-xxx         0/1     Running   0          2m
```

**Diagnosis:**
```bash
kubectl describe pod <pod-name> | grep -A 5 "Readiness"

# Look for:
# "Readiness probe failed: HTTP probe failed"
```

**Solutions:**
```bash
# Check if health endpoint exists
kubectl exec <pod-name> -- curl -f http://localhost:9000/health/ready

# Adjust probe timing
kubectl edit deployment ml-inference

# Increase initialDelaySeconds or periodSeconds
```

---

## Best Practices

### 1. Resource Management

```yaml
# Always set requests and limits
resources:
  requests:
    memory: "256Mi"  # Guaranteed
    cpu: "250m"
  limits:
    memory: "512Mi"  # Maximum
    cpu: "500m"

# ML workloads: Set memory limits 2x requests for flexibility
# CPU: Set limits 1.5-2x requests for burst capacity
```

### 2. Health Checks

```yaml
# Liveness: Detect and restart dead containers
livenessProbe:
  httpGet:
    path: /health/alive
    port: 9000
  initialDelaySeconds: 30  # Wait for app to start
  periodSeconds: 10
  failureThreshold: 3      # Restart after 3 failures

# Readiness: Remove from service if not ready
readinessProbe:
  httpGet:
    path: /health/ready
    port: 9000
  initialDelaySeconds: 20
  periodSeconds: 5
  failureThreshold: 3

# ML models: Use separate endpoints for model loading vs. serving
```

### 3. Deployment Strategy

```yaml
strategy:
  type: RollingUpdate
  rollingUpdate:
    maxUnavailable: 1  # At most 1 pod down during update
    maxSurge: 1        # At most 1 extra pod during update

# For critical services: maxUnavailable: 0 (zero downtime)
# For ML models: Allow time for model loading
```

### 4. Labels and Annotations

```yaml
metadata:
  labels:
    app: ml-inference          # Application name
    tier: ml-service           # Architecture tier
    version: v1                # Version for canary deployments
    model-version: iris-1.0    # ML model version
  annotations:
    description: "Iris classification model"
    model-trained: "2024-01-15"
    kubernetes.io/change-cause: "Update to model v1.1"
```

### 5. Auto-Scaling Configuration

```yaml
# Conservative auto-scaling for ML workloads
spec:
  minReplicas: 2              # Always at least 2 for HA
  maxReplicas: 10             # Limit max to control costs
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        averageUtilization: 70  # Scale before saturation
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300  # Wait 5min before scale-down
```

### 6. Production Checklist

- [ ] Resource requests and limits set
- [ ] Liveness and readiness probes configured
- [ ] Multiple replicas for high availability
- [ ] HPA configured for auto-scaling
- [ ] Rolling update strategy defined
- [ ] Labels and annotations documented
- [ ] Service type appropriate for use case
- [ ] Health endpoints implemented
- [ ] Logging configured
- [ ] Monitoring in place

---

## Summary

Congratulations! You've completed a comprehensive implementation of Kubernetes deployments with a focus on ML inference services.

### Key Takeaways

1. **Cluster Setup**: You can run Kubernetes locally using Docker Desktop, minikube, or kind
2. **Deployments**: Manage application lifecycle with declarative YAML manifests
3. **Services**: Expose applications using ClusterIP, NodePort, or LoadBalancer
4. **Scaling**: Scale manually or automatically based on resource utilization
5. **Updates**: Perform zero-downtime rolling updates with automatic rollback
6. **Resources**: Set requests/limits for predictable performance
7. **ML Workloads**: Deploy production ML models with health checks and auto-scaling

### Skills Acquired

- Creating and managing Kubernetes resources
- Writing production-ready YAML manifests
- Implementing health checks and readiness probes
- Configuring auto-scaling for variable workloads
- Performing zero-downtime deployments
- Troubleshooting common Kubernetes issues
- Managing resources for ML inference services

### Next Steps

1. **Exercise 02**: Package applications with Helm charts
2. **Exercise 03**: Advanced debugging techniques
3. **Exercise 04**: StatefulSets for databases and ML training
4. **Exercise 05**: ConfigMaps and Secrets for configuration management
5. **Exercise 06**: Ingress controllers for HTTP routing
6. **Exercise 07**: Advanced ML workloads with GPU support

### Additional Resources

- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [kubectl Cheat Sheet](https://kubernetes.io/docs/reference/kubectl/cheatsheet/)
- [Production Best Practices](https://kubernetes.io/docs/concepts/configuration/overview/)
- [Autoscaling Guide](https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/)

---

**Exercise Complete!** You now have hands-on experience deploying, scaling, and managing applications on Kubernetes with a focus on ML inference services. These foundational skills are essential for AI infrastructure engineering.
