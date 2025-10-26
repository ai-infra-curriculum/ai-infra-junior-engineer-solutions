# Step-by-Step Implementation Guide: First Kubernetes Deployment

## Overview

Deploy your first application to Kubernetes! Learn essential concepts: Pods, Deployments, Services, scaling, rolling updates, and basic troubleshooting. This exercise builds foundation skills for running ML workloads in production.

**Time**: 2-3 hours | **Difficulty**: Beginner

---

## Learning Objectives

✅ Understand Kubernetes core concepts (Pods, Deployments, Services)
✅ Create and manage Deployments
✅ Expose applications with Services
✅ Scale applications horizontally
✅ Perform rolling updates
✅ Configure health checks (liveness/readiness probes)
✅ Set resource requests and limits
✅ Troubleshoot basic deployment issues

---

## Prerequisites

```bash
# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# Verify installation
kubectl version --client

# Install minikube (local Kubernetes)
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube

# Start cluster
minikube start --cpus 4 --memory 8192

# Verify cluster
kubectl cluster-info
kubectl get nodes
```

---

## Phase 1: Understanding Kubernetes Architecture

### Key Concepts

```
┌─────────────────────────────────────────┐
│           Kubernetes Cluster            │
│                                         │
│  ┌──────────────────────────────────┐  │
│  │        Control Plane             │  │
│  │  - API Server                    │  │
│  │  - Scheduler                     │  │
│  │  - Controller Manager            │  │
│  │  - etcd (state storage)          │  │
│  └──────────────────────────────────┘  │
│                                         │
│  ┌──────────────────────────────────┐  │
│  │         Worker Nodes             │  │
│  │  ┌────────────┐  ┌────────────┐  │  │
│  │  │   Pod 1    │  │   Pod 2    │  │  │
│  │  │ Container  │  │ Container  │  │  │
│  │  └────────────┘  └────────────┘  │  │
│  │         kubelet + kube-proxy      │  │
│  └──────────────────────────────────┘  │
└─────────────────────────────────────────┘
```

**Pod**: Smallest deployable unit (one or more containers)
**Deployment**: Manages replica Pods, handles updates
**Service**: Stable networking endpoint for Pods
**ReplicaSet**: Ensures desired number of Pod replicas

---

## Phase 2: Create Your First Deployment

### Basic Nginx Deployment

```yaml
# manifests/01-nginx-deployment.yaml
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
            scheme: HTTP
          initialDelaySeconds: 10
          periodSeconds: 5
          timeoutSeconds: 3
          successThreshold: 1
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /
            port: 80
            scheme: HTTP
          initialDelaySeconds: 5
          periodSeconds: 3
          timeoutSeconds: 2
          successThreshold: 1
          failureThreshold: 3
        env:
        - name: NGINX_HOST
          value: "kubernetes.local"
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

### Deploy to Kubernetes

```bash
# Create namespace
kubectl create namespace exercise-01
kubectl config set-context --current --namespace=exercise-01

# Apply deployment
kubectl apply -f manifests/01-nginx-deployment.yaml

# Watch deployment progress
kubectl get deployments -w

# Check pods
kubectl get pods -o wide

# Describe deployment (detailed info)
kubectl describe deployment nginx-web

# Check events
kubectl get events --sort-by=.metadata.creationTimestamp
```

---

## Phase 3: Expose with Services

### ClusterIP Service (Internal)

```yaml
# manifests/02-nginx-service-clusterip.yaml
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
  - protocol: TCP
    port: 80
    targetPort: 80
    name: http
  sessionAffinity: None
```

```bash
# Create service
kubectl apply -f manifests/02-nginx-service-clusterip.yaml

# Get service details
kubectl get svc nginx-service

# Test connectivity (from within cluster)
kubectl run -it --rm debug --image=curlimages/curl --restart=Never -- \
  curl http://nginx-service

# Port forward to local machine
kubectl port-forward svc/nginx-service 8080:80

# Access from browser: http://localhost:8080
```

### NodePort Service (External Access)

```yaml
# manifests/03-nginx-service-nodeport.yaml
apiVersion: v1
kind: Service
metadata:
  name: nginx-nodeport
  labels:
    app: nginx
spec:
  type: NodePort
  selector:
    app: nginx
  ports:
  - protocol: TCP
    port: 80
    targetPort: 80
    nodePort: 30080
    name: http
```

```bash
# Create NodePort service
kubectl apply -f manifests/03-nginx-service-nodeport.yaml

# Get node IP
minikube ip

# Access service
curl http://$(minikube ip):30080

# Or use minikube service
minikube service nginx-nodeport
```

---

## Phase 4: ConfigMaps for Configuration

### Custom HTML ConfigMap

```yaml
# manifests/04-nginx-custom-html-configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: nginx-html
data:
  index.html: |
    <!DOCTYPE html>
    <html>
    <head>
        <title>Kubernetes ML Infrastructure</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                margin: 0;
            }
            .container {
                text-align: center;
                background: rgba(255,255,255,0.1);
                padding: 50px;
                border-radius: 20px;
                backdrop-filter: blur(10px);
            }
            h1 { font-size: 3em; margin-bottom: 20px; }
            p { font-size: 1.2em; }
            .info { margin-top: 30px; font-family: monospace; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 Kubernetes Deployment</h1>
            <p>Your first ML infrastructure deployment!</p>
            <div class="info">
                <p>Hostname: <span id="hostname"></span></p>
                <p>Pod IP: <span id="ip"></span></p>
            </div>
        </div>
        <script>
            fetch('/api/info')
                .then(r => r.json())
                .then(data => {
                    document.getElementById('hostname').textContent = data.hostname;
                    document.getElementById('ip').textContent = data.ip;
                });
        </script>
    </body>
    </html>
```

### Deployment with ConfigMap

```yaml
# manifests/05-nginx-with-configmap.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx-custom
spec:
  replicas: 3
  selector:
    matchLabels:
      app: nginx-custom
  template:
    metadata:
      labels:
        app: nginx-custom
    spec:
      containers:
      - name: nginx
        image: nginx:1.21
        ports:
        - containerPort: 80
        volumeMounts:
        - name: html
          mountPath: /usr/share/nginx/html
          readOnly: true
      volumes:
      - name: html
        configMap:
          name: nginx-html
```

```bash
# Apply ConfigMap and deployment
kubectl apply -f manifests/04-nginx-custom-html-configmap.yaml
kubectl apply -f manifests/05-nginx-with-configmap.yaml

# Verify
kubectl get pods -l app=nginx-custom
kubectl port-forward deployment/nginx-custom 8081:80
```

---

## Phase 5: Scaling and Updates

### Horizontal Scaling

```bash
# Scale up
kubectl scale deployment nginx-web --replicas=5

# Watch scaling
kubectl get pods -l app=nginx -w

# Verify
kubectl get deployment nginx-web

# Autoscaling (Horizontal Pod Autoscaler)
kubectl autoscale deployment nginx-web \
  --cpu-percent=50 \
  --min=2 \
  --max=10

# Check HPA status
kubectl get hpa
```

### Rolling Updates

```bash
# Update image (rolling update)
kubectl set image deployment/nginx-web nginx=nginx:1.22

# Watch rollout
kubectl rollout status deployment/nginx-web

# Check rollout history
kubectl rollout history deployment/nginx-web

# Rollback if needed
kubectl rollout undo deployment/nginx-web

# Rollback to specific revision
kubectl rollout undo deployment/nginx-web --to-revision=1
```

### Update Strategy

```yaml
# Add to deployment spec
spec:
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1        # Max additional pods during update
      maxUnavailable: 0  # Max unavailable pods during update
```

---

## Phase 6: Resource Management

### Resource Requests and Limits

```yaml
resources:
  requests:
    memory: "64Mi"   # Guaranteed minimum
    cpu: "100m"      # 100 millicores = 0.1 CPU
  limits:
    memory: "128Mi"  # Maximum allowed
    cpu: "200m"      # Maximum CPU
```

**Requests**: Used for scheduling decisions
**Limits**: Hard caps enforced by kubelet

### Test Resource Limits

```yaml
# manifests/07-resource-test-pod.yaml
apiVersion: v1
kind: Pod
metadata:
  name: resource-test
spec:
  containers:
  - name: stress
    image: polinux/stress
    command: ["stress"]
    args:
    - "--vm"
    - "1"
    - "--vm-bytes"
    - "150M"  # Exceeds limit
    - "--vm-hang"
    - "1"
    resources:
      requests:
        memory: "64Mi"
      limits:
        memory: "128Mi"
```

```bash
# Deploy and observe OOMKilled
kubectl apply -f manifests/07-resource-test-pod.yaml
kubectl get pod resource-test -w
kubectl describe pod resource-test
```

---

## Phase 7: Health Checks

### Liveness Probe

Restarts container if probe fails.

```yaml
livenessProbe:
  httpGet:
    path: /healthz
    port: 8080
  initialDelaySeconds: 15
  periodSeconds: 10
  timeoutSeconds: 5
  failureThreshold: 3
```

### Readiness Probe

Removes Pod from Service endpoints if not ready.

```yaml
readinessProbe:
  httpGet:
    path: /ready
    port: 8080
  initialDelaySeconds: 5
  periodSeconds: 3
  timeoutSeconds: 2
  failureThreshold: 3
```

### Startup Probe

Protects slow-starting containers.

```yaml
startupProbe:
  httpGet:
    path: /startup
    port: 8080
  initialDelaySeconds: 0
  periodSeconds: 10
  failureThreshold: 30  # 30*10 = 5 min max startup
```

---

## Phase 8: Troubleshooting

### Common Commands

```bash
# Get pod logs
kubectl logs <pod-name>

# Follow logs
kubectl logs -f <pod-name>

# Previous container logs (after crash)
kubectl logs <pod-name> --previous

# Logs from all pods in deployment
kubectl logs -l app=nginx --tail=50

# Execute command in pod
kubectl exec -it <pod-name> -- /bin/bash

# Copy files from pod
kubectl cp <pod-name>:/path/to/file ./local-file

# Check pod events
kubectl describe pod <pod-name>

# Debug with ephemeral container
kubectl debug <pod-name> -it --image=busybox
```

### Common Issues

**ImagePullBackOff**
```bash
# Check image name and tag
kubectl describe pod <pod-name> | grep Image

# Check image pull secret
kubectl get secrets
```

**CrashLoopBackOff**
```bash
# Check logs
kubectl logs <pod-name> --previous

# Check liveness probe configuration
kubectl describe pod <pod-name> | grep -A 5 Liveness
```

**Pending**
```bash
# Check resource availability
kubectl describe pod <pod-name> | grep -A 10 Events

# Check node resources
kubectl top nodes
```

---

## Phase 9: Complete Deployment Script

```bash
#!/bin/bash
# scripts/deploy.sh
set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

NAMESPACE="exercise-01"

echo -e "${GREEN}========================================"
echo "Deploying Exercise 01 Resources"
echo -e "========================================${NC}"

# Switch namespace
kubectl config set-context --current --namespace="$NAMESPACE"

# Deploy resources
echo -e "\n${YELLOW}Step 1: Deploying nginx Deployment...${NC}"
kubectl apply -f manifests/01-nginx-deployment.yaml
kubectl wait --for=condition=available --timeout=60s deployment/nginx-web
echo -e "${GREEN}✓ Deployment ready${NC}"

echo -e "\n${YELLOW}Step 2: Creating ClusterIP Service...${NC}"
kubectl apply -f manifests/02-nginx-service-clusterip.yaml
echo -e "${GREEN}✓ Service created${NC}"

echo -e "\n${YELLOW}Step 3: Creating ConfigMap...${NC}"
kubectl apply -f manifests/04-nginx-custom-html-configmap.yaml
echo -e "${GREEN}✓ ConfigMap created${NC}"

echo -e "\n${YELLOW}Step 4: Deploying custom nginx...${NC}"
kubectl apply -f manifests/05-nginx-with-configmap.yaml
kubectl wait --for=condition=available --timeout=60s deployment/nginx-custom
echo -e "${GREEN}✓ Custom nginx ready${NC}"

# Display status
echo -e "\n${YELLOW}Current Status:${NC}"
kubectl get deployments,services,pods

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Deployment Complete!${NC}"
echo -e "${GREEN}========================================${NC}"

echo -e "\n${YELLOW}Next steps:${NC}"
echo "1. Port forward: kubectl port-forward deployment/nginx-web 8080:80"
echo "2. View logs: kubectl logs -l app=nginx"
echo "3. Scale: kubectl scale deployment nginx-web --replicas=5"
echo "4. Update: kubectl set image deployment/nginx-web nginx=nginx:1.22"
echo "5. Cleanup: ./scripts/cleanup.sh"
```

---

## Best Practices

✅ Always set resource requests and limits
✅ Use health checks (liveness + readiness)
✅ Use namespaces to organize resources
✅ Label resources consistently
✅ Use ConfigMaps for configuration
✅ Implement graceful shutdown
✅ Monitor resource usage
✅ Use rolling updates for zero-downtime
✅ Test rollback procedures
✅ Document deployment procedures

---

## Cleanup

```bash
#!/bin/bash
# scripts/cleanup.sh

echo "Cleaning up Exercise 01 resources..."

# Delete deployments
kubectl delete deployment nginx-web nginx-custom

# Delete services
kubectl delete service nginx-service nginx-nodeport

# Delete configmaps
kubectl delete configmap nginx-html

# Verify cleanup
kubectl get all

echo "✓ Cleanup complete"
```

---

## Verification Checklist

✅ Deployment created successfully
✅ Pods running and healthy
✅ Services accessible
✅ ConfigMaps mounted correctly
✅ Scaling works properly
✅ Rolling updates succeed
✅ Health checks functioning
✅ Resource limits enforced
✅ Logs accessible
✅ Cleanup successful

---

**First Kubernetes deployment complete!** 🎉

**Next Exercise**: Helm Charts for package management
