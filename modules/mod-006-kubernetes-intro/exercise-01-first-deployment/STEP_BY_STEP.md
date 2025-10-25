# Step-by-Step Implementation Guide: First Kubernetes Deployment

## Overview

Deploy your first application on Kubernetes, learning core concepts: Pods, Deployments, Services, scaling, rolling updates, and troubleshooting.

**Time**: 2-3 hours | **Difficulty**: Beginner

---

## Phase 1: Verify Kubernetes Cluster (15 minutes)

### Step 1: Check Cluster Status

```bash
# Verify kubectl is configured
kubectl version --short

# Check cluster information
kubectl cluster-info

# Expected output:
# Kubernetes control plane is running at https://127.0.0.1:6443
# CoreDNS is running at https://127.0.0.1:6443/api/v1/namespaces/kube-system/services/kube-dns:dns/proxy

# List nodes
kubectl get nodes

# Expected output:
# NAME       STATUS   ROLES           AGE   VERSION
# minikube   Ready    control-plane   10d   v1.28.3

# Check node details
kubectl describe node <node-name>
```

**Troubleshooting**:
- If cluster not running: `minikube start` or enable Kubernetes in Docker Desktop
- If kubectl not found: Install via `brew install kubectl` (Mac) or download from kubernetes.io

---

## Phase 2: Create Your First Pod (20 minutes)

### Step 2: Create a Simple Pod

**`manifests/01-pod.yaml`**:
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: nginx-pod
  labels:
    app: nginx
    env: demo
spec:
  containers:
  - name: nginx
    image: nginx:1.25
    ports:
    - containerPort: 80
    resources:
      requests:
        memory: "64Mi"
        cpu: "250m"
      limits:
        memory: "128Mi"
        cpu: "500m"
```

**Apply the Pod**:
```bash
# Create the Pod
kubectl apply -f manifests/01-pod.yaml

# Watch Pod startup
kubectl get pods -w

# Wait for Pod to be Running
kubectl wait --for=condition=Ready pod/nginx-pod --timeout=60s

# Check Pod details
kubectl describe pod nginx-pod

# View Pod logs
kubectl logs nginx-pod

# Access Pod (port-forward)
kubectl port-forward pod/nginx-pod 8080:80

# In another terminal, test
curl http://localhost:8080
# Expected: Nginx welcome page HTML
```

**Why Pods Alone Are Not Enough**:
- Pods are ephemeral (die on node failure)
- No automatic restart
- No load balancing
- No rolling updates
- **Solution**: Use Deployments instead

---

## Phase 3: Create a Deployment (30 minutes)

### Step 3: Deploy Nginx with Deployment

**`manifests/02-deployment.yaml`**:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx-deployment
  labels:
    app: nginx
spec:
  replicas: 3  # Run 3 Pods for high availability
  selector:
    matchLabels:
      app: nginx
  template:
    metadata:
      labels:
        app: nginx
    spec:
      containers:
      - name: nginx
        image: nginx:1.25
        ports:
        - containerPort: 80
        resources:
          requests:
            memory: "64Mi"
            cpu: "250m"
          limits:
            memory: "128Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /
            port: 80
          initialDelaySeconds: 10
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /
            port: 80
          initialDelaySeconds: 5
          periodSeconds: 5
```

**Apply the Deployment**:
```bash
# Create Deployment
kubectl apply -f manifests/02-deployment.yaml

# Watch Pods being created
kubectl get pods -l app=nginx -w

# Check Deployment status
kubectl get deployment nginx-deployment

# Expected output:
# NAME               READY   UP-TO-DATE   AVAILABLE   AGE
# nginx-deployment   3/3     3            3           30s

# View Deployment details
kubectl describe deployment nginx-deployment

# View ReplicaSet (created by Deployment)
kubectl get replicaset

# Check rollout status
kubectl rollout status deployment/nginx-deployment
```

**Key Concepts**:
- **Deployment**: Manages ReplicaSets and Pods
- **ReplicaSet**: Ensures desired number of Pods are running
- **Pod**: Actual running container(s)

---

## Phase 4: Expose with Service (25 minutes)

### Step 4: Create ClusterIP Service

**`manifests/03-service-clusterip.yaml`**:
```yaml
apiVersion: v1
kind: Service
metadata:
  name: nginx-service
spec:
  type: ClusterIP  # Internal cluster access only
  selector:
    app: nginx
  ports:
  - protocol: TCP
    port: 80  # Service port
    targetPort: 80  # Container port
```

**Apply and Test**:
```bash
# Create Service
kubectl apply -f manifests/03-service-clusterip.yaml

# Get Service details
kubectl get service nginx-service

# Expected output:
# NAME            TYPE        CLUSTER-IP      EXTERNAL-IP   PORT(S)   AGE
# nginx-service   ClusterIP   10.96.100.200   <none>        80/TCP    10s

# Test from within cluster (create test Pod)
kubectl run test-pod --image=busybox --rm -it --restart=Never -- wget -qO- nginx-service

# Expected: Nginx HTML output
```

### Step 5: Create NodePort Service (External Access)

**`manifests/04-service-nodeport.yaml`**:
```yaml
apiVersion: v1
kind: Service
metadata:
  name: nginx-nodeport
spec:
  type: NodePort
  selector:
    app: nginx
  ports:
  - protocol: TCP
    port: 80
    targetPort: 80
    nodePort: 30081  # External port (30000-32767 range)
```

**Apply and Access**:
```bash
# Create NodePort Service
kubectl apply -f manifests/04-service-nodeport.yaml

# Get Service
kubectl get service nginx-nodeport

# Access via NodePort
# If using Minikube
minikube service nginx-nodeport --url

# Or get node IP
kubectl get nodes -o wide
# Access via http://<NODE_IP>:30081

# If using Docker Desktop (localhost)
curl http://localhost:30081
```

---

## Phase 5: Scaling (15 minutes)

### Step 6: Scale the Deployment

```bash
# Scale to 5 replicas
kubectl scale deployment nginx-deployment --replicas=5

# Watch Pods being created
kubectl get pods -l app=nginx -w

# Scale down to 2 replicas
kubectl scale deployment nginx-deployment --replicas=2

# Verify
kubectl get deployment nginx-deployment

# Autoscaling (Horizontal Pod Autoscaler)
kubectl autoscale deployment nginx-deployment --min=2 --max=10 --cpu-percent=80

# Check HPA
kubectl get hpa
```

---

## Phase 6: Rolling Updates (30 minutes)

### Step 7: Update Application (Zero-Downtime)

```bash
# Current image
kubectl get deployment nginx-deployment -o jsonpath='{.spec.template.spec.containers[0].image}'
# Output: nginx:1.25

# Update to new version
kubectl set image deployment/nginx-deployment nginx=nginx:1.26

# Watch rollout
kubectl rollout status deployment/nginx-deployment

# Check rollout history
kubectl rollout history deployment/nginx-deployment

# Verify new Pods
kubectl get pods -l app=nginx

# Check image version
kubectl get pods -l app=nginx -o jsonpath='{.items[*].spec.containers[*].image}'
```

**How Rolling Updates Work**:
```
Initial State: 3 Pods (nginx:1.25)

Step 1: Create 1 new Pod (nginx:1.26)
  Old: 3, New: 1 (Total: 4)

Step 2: Terminate 1 old Pod
  Old: 2, New: 1 (Total: 3)

Step 3: Create 1 new Pod
  Old: 2, New: 2 (Total: 4)

Step 4: Terminate 1 old Pod
  Old: 1, New: 2 (Total: 3)

Step 5: Create 1 new Pod
  Old: 1, New: 3 (Total: 4)

Step 6: Terminate last old Pod
  Old: 0, New: 3 (Total: 3) ✓ Complete
```

### Step 8: Rollback Failed Update

```bash
# Simulate bad update
kubectl set image deployment/nginx-deployment nginx=nginx:bad-version

# Watch it fail
kubectl rollout status deployment/nginx-deployment
# Output: Waiting for deployment "nginx-deployment" rollout to finish: 1 old replicas are pending termination...

# Check Pods
kubectl get pods -l app=nginx
# Some Pods will be in ImagePullBackOff state

# Rollback to previous version
kubectl rollout undo deployment/nginx-deployment

# Verify rollback
kubectl rollout status deployment/nginx-deployment

# Rollback to specific revision
kubectl rollout history deployment/nginx-deployment
kubectl rollout undo deployment/nginx-deployment --to-revision=1
```

---

## Phase 7: ConfigMaps (25 minutes)

### Step 9: Add Configuration

**`manifests/05-configmap.yaml`**:
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: nginx-config
data:
  nginx.conf: |
    server {
        listen 80;
        server_name localhost;

        location / {
            root /usr/share/nginx/html;
            index index.html;
        }

        location /health {
            access_log off;
            return 200 "healthy\n";
            add_header Content-Type text/plain;
        }
    }
  index.html: |
    <!DOCTYPE html>
    <html>
    <head><title>Kubernetes Nginx</title></head>
    <body>
        <h1>Hello from Kubernetes!</h1>
        <p>This is served from a ConfigMap</p>
    </body>
    </html>
```

**Update Deployment to Use ConfigMap**:
```yaml
# Add to Deployment spec.template.spec
volumes:
- name: nginx-config
  configMap:
    name: nginx-config
- name: html
  configMap:
    name: nginx-config
    items:
    - key: index.html
      path: index.html

# Add to container spec
volumeMounts:
- name: nginx-config
  mountPath: /etc/nginx/conf.d/default.conf
  subPath: nginx.conf
- name: html
  mountPath: /usr/share/nginx/html
```

**Apply**:
```bash
kubectl apply -f manifests/05-configmap.yaml
kubectl apply -f manifests/02-deployment.yaml  # Updated version

# Test custom HTML
curl http://localhost:30081
# Expected: "Hello from Kubernetes!"

# Test health endpoint
curl http://localhost:30081/health
# Expected: "healthy"
```

---

## Phase 8: Troubleshooting (20 minutes)

### Step 10: Debug Common Issues

**Issue 1: Pod Not Starting**

```bash
# Check Pod status
kubectl get pods

# Describe Pod (most useful!)
kubectl describe pod <pod-name>

# Common issues in Events section:
# - ImagePullBackOff: Image doesn't exist
# - CrashLoopBackOff: Container keeps crashing
# - Pending: Can't be scheduled (resource constraints)

# Check logs
kubectl logs <pod-name>

# If container keeps restarting, get logs from previous instance
kubectl logs <pod-name> --previous
```

**Issue 2: Service Not Accessible**

```bash
# Verify Service exists
kubectl get service nginx-service

# Check Service endpoints (should show Pod IPs)
kubectl get endpoints nginx-service

# If no endpoints, check selector matches Pod labels
kubectl get pods --show-labels

# Test from within cluster
kubectl run test --image=busybox --rm -it --restart=Never -- wget -qO- nginx-service
```

**Issue 3: Rolling Update Stuck**

```bash
# Check rollout status
kubectl rollout status deployment/nginx-deployment

# Check Deployment events
kubectl describe deployment nginx-deployment

# Check ReplicaSets
kubectl get rs

# Rollback if needed
kubectl rollout undo deployment/nginx-deployment
```

---

## Phase 9: Cleanup (5 minutes)

### Step 11: Remove Resources

```bash
# Delete specific resources
kubectl delete deployment nginx-deployment
kubectl delete service nginx-service nginx-nodeport
kubectl delete configmap nginx-config

# Or delete everything with label
kubectl delete all -l app=nginx

# Verify cleanup
kubectl get all
```

---

## Summary

**What You Built**:
- ✅ Created and managed Pods
- ✅ Deployed applications with Deployments
- ✅ Exposed services with ClusterIP and NodePort
- ✅ Scaled applications horizontally
- ✅ Performed zero-downtime rolling updates
- ✅ Rolled back failed deployments
- ✅ Used ConfigMaps for configuration
- ✅ Set resource requests and limits
- ✅ Debugged common Kubernetes issues

**Key kubectl Commands**:
```bash
# Deployments
kubectl create deployment <name> --image=<image>
kubectl get deployments
kubectl describe deployment <name>
kubectl scale deployment <name> --replicas=<n>
kubectl set image deployment/<name> <container>=<image>
kubectl rollout status deployment/<name>
kubectl rollout undo deployment/<name>

# Pods
kubectl get pods
kubectl describe pod <name>
kubectl logs <name>
kubectl exec -it <name> -- /bin/bash
kubectl port-forward pod/<name> <local-port>:<pod-port>

# Services
kubectl expose deployment <name> --port=80 --type=NodePort
kubectl get services
kubectl get endpoints

# General
kubectl apply -f <file.yaml>
kubectl delete -f <file.yaml>
kubectl get all
```

**Next Steps**:
- Exercise 02: Create custom Helm charts
- Exercise 03: Advanced debugging techniques
- Exercise 04: StatefulSets and persistent storage
