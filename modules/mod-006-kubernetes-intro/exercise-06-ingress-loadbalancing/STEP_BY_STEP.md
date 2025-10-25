# Step-by-Step Implementation Guide: Ingress & Load Balancing

## Overview

Master advanced Kubernetes networking by implementing Ingress controllers, path-based and host-based routing, TLS termination, canary deployments, and network security policies. Learn to expose applications efficiently and securely.

**Time**: 3-4 hours | **Difficulty**: Intermediate to Advanced

---

## Phase 1: Install Ingress Controller (30 minutes)

### Step 1: Understand the Ingress Architecture

**Key Concept**: Unlike Services (Layer 4 - TCP/UDP), Ingress operates at Layer 7 (HTTP/HTTPS), providing advanced routing, SSL termination, and cost-effective load balancing.

```
┌─────────────────────────────────────────────────────┐
│              External Traffic (Internet)             │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
        ┌────────────────────────┐
        │   Cloud Load Balancer   │  (Single external IP)
        │   (or NodePort)         │
        └────────────┬───────────┘
                     │
                     ▼
        ┌────────────────────────┐
        │  Ingress Controller    │  (NGINX, Traefik, HAProxy)
        │  - Watches Ingress     │
        │  - Configures routing  │
        │  - Terminates TLS      │
        └────────────┬───────────┘
                     │
        ┌────────────┴──────────────────┐
        ▼                               ▼
┌───────────────┐              ┌──────────────┐
│ ClusterIP Svc │              │ ClusterIP Svc│
│  (api-svc)    │              │  (web-svc)   │
└───────┬───────┘              └──────┬───────┘
        │                             │
        ▼                             ▼
  ┌─────────┐                   ┌─────────┐
  │ Pods    │                   │ Pods    │
  │ (API)   │                   │ (Web)   │
  └─────────┘                   └─────────┘
```

### Step 2: Install NGINX Ingress Controller

**For Minikube** (local testing):
```bash
# Enable Ingress addon
minikube addons enable ingress

# Verify installation
kubectl get pods -n ingress-nginx

# Expected output:
# NAME                                        READY   STATUS      RESTARTS   AGE
# ingress-nginx-admission-create-xxxxx        0/1     Completed   0          1m
# ingress-nginx-admission-patch-xxxxx         0/1     Completed   0          1m
# ingress-nginx-controller-xxxxx              1/1     Running     0          1m
```

**For kind** (Kubernetes in Docker):
```bash
# Apply kind-specific manifest
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/main/deploy/static/provider/kind/deploy.yaml

# Wait for controller to be ready
kubectl wait --namespace ingress-nginx \
  --for=condition=ready pod \
  --selector=app.kubernetes.io/component=controller \
  --timeout=90s
```

**For cloud providers** (AWS, GCP, Azure):
```bash
# Apply cloud provider manifest
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.9.4/deploy/static/provider/cloud/deploy.yaml

# This creates a LoadBalancer service that gets an external IP
kubectl get svc -n ingress-nginx
```

**For bare metal**:
```bash
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.9.4/deploy/static/provider/baremetal/deploy.yaml

# Note: You'll need to use NodePort to access Ingress
```

### Step 3: Verify Ingress Controller

```bash
# Check Ingress controller is running
kubectl get pods -n ingress-nginx -w

# Check IngressClass
kubectl get ingressclass

# Expected output:
# NAME    CONTROLLER             PARAMETERS   AGE
# nginx   k8s.io/ingress-nginx   <none>       5m

# Get Ingress controller service
kubectl get svc -n ingress-nginx ingress-nginx-controller

# For cloud: Note the EXTERNAL-IP
# For minikube: Use `minikube ip` and NodePort
# For kind: Use localhost and NodePort
```

### Step 4: Get Ingress IP Address

```bash
# For cloud providers (AWS, GCP, Azure)
INGRESS_IP=$(kubectl get svc -n ingress-nginx ingress-nginx-controller \
  -o jsonpath='{.status.loadBalancer.ingress[0].ip}')

# If IP is empty, try hostname (AWS ELB)
INGRESS_HOST=$(kubectl get svc -n ingress-nginx ingress-nginx-controller \
  -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')

echo "Ingress IP: $INGRESS_IP"
echo "Ingress Host: $INGRESS_HOST"

# For minikube
INGRESS_IP=$(minikube ip)
echo "Ingress IP (Minikube): $INGRESS_IP"

# For kind
INGRESS_IP="localhost"
echo "Ingress IP (kind): $INGRESS_IP"
```

---

## Phase 2: Deploy Backend Applications (20 minutes)

### Step 5: Create Namespace and Backend Apps

**`manifests/01-namespace.yaml`**:
```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: ingress-demo
  labels:
    name: ingress-demo
```

**`manifests/02-backend-apps.yaml`** (partial):
```yaml
# API Backend (v1)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: backend-api-v1
  namespace: ingress-demo
spec:
  replicas: 2
  selector:
    matchLabels:
      app: backend-api
      version: v1
  template:
    metadata:
      labels:
        app: backend-api
        version: v1
    spec:
      containers:
      - name: api
        image: hashicorp/http-echo:0.2.3
        args:
          - "-text=API Response v1"
          - "-listen=:8080"
        ports:
        - containerPort: 8080
---
# API Backend (v2) - for canary testing
apiVersion: apps/v1
kind: Deployment
metadata:
  name: backend-api-v2
  namespace: ingress-demo
spec:
  replicas: 2
  selector:
    matchLabels:
      app: backend-api
      version: v2
  template:
    metadata:
      labels:
        app: backend-api
        version: v2
    spec:
      containers:
      - name: api
        image: hashicorp/http-echo:0.2.3
        args:
          - "-text=API Response v2 (CANARY)"
          - "-listen=:8080"
        ports:
        - containerPort: 8080
---
# Web Backend
apiVersion: apps/v1
kind: Deployment
metadata:
  name: backend-web
  namespace: ingress-demo
spec:
  replicas: 2
  selector:
    matchLabels:
      app: backend-web
  template:
    metadata:
      labels:
        app: backend-web
    spec:
      containers:
      - name: web
        image: hashicorp/http-echo:0.2.3
        args:
          - "-text=Web Application Response"
          - "-listen=:8080"
        ports:
        - containerPort: 8080
---
# Admin Backend
apiVersion: apps/v1
kind: Deployment
metadata:
  name: backend-admin
  namespace: ingress-demo
spec:
  replicas: 1
  selector:
    matchLabels:
      app: backend-admin
  template:
    metadata:
      labels:
        app: backend-admin
    spec:
      containers:
      - name: admin
        image: hashicorp/http-echo:0.2.3
        args:
          - "-text=Admin Panel (Restricted)"
          - "-listen=:8080"
        ports:
        - containerPort: 8080
```

**Apply backends**:
```bash
kubectl apply -f manifests/01-namespace.yaml
kubectl apply -f manifests/02-backend-apps.yaml

# Verify deployments
kubectl get deployments -n ingress-demo

# Expected output:
# NAME              READY   UP-TO-DATE   AVAILABLE   AGE
# backend-api-v1    2/2     2            2           1m
# backend-api-v2    2/2     2            2           1m
# backend-web       2/2     2            2           1m
# backend-admin     1/1     1            1           1m

# Check pods
kubectl get pods -n ingress-demo
```

---

## Phase 3: Create Services (25 minutes)

### Step 6: Understand Service Types

**Service Types Comparison**:

| Type | Use Case | External Access | Cost | Example |
|------|----------|----------------|------|---------|
| **ClusterIP** | Internal communication | No | Free | Microservices communication |
| **NodePort** | Testing, bare metal | Yes (NodeIP:Port) | Free | Dev environments |
| **LoadBalancer** | Production (cloud) | Yes (cloud LB) | $$$$ | Production apps (1 LB per service) |
| **ExternalName** | External service DNS | CNAME only | Free | Legacy service migration |

### Step 7: Create Services

**`manifests/03-services.yaml`**:
```yaml
# ClusterIP (default) - used by Ingress
apiVersion: v1
kind: Service
metadata:
  name: backend-api-v1
  namespace: ingress-demo
spec:
  type: ClusterIP
  selector:
    app: backend-api
    version: v1
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8080
---
# ClusterIP for v2 (canary)
apiVersion: v1
kind: Service
metadata:
  name: backend-api-v2
  namespace: ingress-demo
spec:
  type: ClusterIP
  selector:
    app: backend-api
    version: v2
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8080
---
# ClusterIP for web
apiVersion: v1
kind: Service
metadata:
  name: backend-web
  namespace: ingress-demo
spec:
  type: ClusterIP
  selector:
    app: backend-web
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8080
---
# ClusterIP for admin
apiVersion: v1
kind: Service
metadata:
  name: backend-admin
  namespace: ingress-demo
spec:
  type: ClusterIP
  selector:
    app: backend-admin
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8080
---
# NodePort example (for comparison)
apiVersion: v1
kind: Service
metadata:
  name: backend-nodeport-example
  namespace: ingress-demo
spec:
  type: NodePort
  selector:
    app: backend-api
    version: v1
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8080
    nodePort: 30080  # Must be 30000-32767
---
# Headless service (StatefulSets)
apiVersion: v1
kind: Service
metadata:
  name: backend-headless
  namespace: ingress-demo
spec:
  clusterIP: None  # No load balancing, direct pod access
  selector:
    app: backend-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8080
```

**Apply services**:
```bash
kubectl apply -f manifests/03-services.yaml

# Verify services
kubectl get svc -n ingress-demo

# Expected output:
# NAME                        TYPE        CLUSTER-IP       EXTERNAL-IP   PORT(S)        AGE
# backend-api-v1              ClusterIP   10.96.100.10     <none>        80/TCP         1m
# backend-api-v2              ClusterIP   10.96.100.11     <none>        80/TCP         1m
# backend-web                 ClusterIP   10.96.100.12     <none>        80/TCP         1m
# backend-admin               ClusterIP   10.96.100.13     <none>        80/TCP         1m
# backend-nodeport-example    NodePort    10.96.100.14     <none>        80:30080/TCP   1m

# Check endpoints (should show pod IPs)
kubectl get endpoints -n ingress-demo

# Test ClusterIP service from within cluster
kubectl run test -n ingress-demo --rm -it --image=curlimages/curl --restart=Never \
  -- curl http://backend-api-v1
# Expected: "API Response v1"
```

---

## Phase 4: Basic Ingress Patterns (40 minutes)

### Step 8: Create Simple Ingress

**`manifests/04-ingress-basic.yaml`** (Simple routing):
```yaml
# 1. Simple Ingress - Route all traffic to one service
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: simple-ingress
  namespace: ingress-demo
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
spec:
  ingressClassName: nginx
  rules:
  - http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: backend-api-v1
            port:
              number: 80
```

**Apply and test**:
```bash
kubectl apply -f manifests/04-ingress-basic.yaml

# Wait for Ingress to get an address
kubectl get ingress -n ingress-demo -w

# Expected output:
# NAME             CLASS   HOSTS   ADDRESS         PORTS   AGE
# simple-ingress   nginx   *       192.168.49.2    80      1m

# Test
curl http://$INGRESS_IP/
# Expected: "API Response v1"
```

### Step 9: Path-Based Routing

**Add to `manifests/04-ingress-basic.yaml`**:
```yaml
---
# 2. Path-based routing (Fanout)
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: path-based-ingress
  namespace: ingress-demo
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
spec:
  ingressClassName: nginx
  rules:
  - http:
      paths:
      - path: /api
        pathType: Prefix
        backend:
          service:
            name: backend-api-v1
            port:
              number: 80
      - path: /web
        pathType: Prefix
        backend:
          service:
            name: backend-web
            port:
              number: 80
      - path: /admin
        pathType: Prefix
        backend:
          service:
            name: backend-admin
            port:
              number: 80
```

**How Path Matching Works**:
```
Request: http://example.com/api/users
  ↓
Ingress matches: path: /api
  ↓
Routes to: backend-api-v1
  ↓
Rewrite target: / (removes /api prefix)
  ↓
Backend receives: GET /users
```

**Test path-based routing**:
```bash
kubectl apply -f manifests/04-ingress-basic.yaml

curl http://$INGRESS_IP/api
# Expected: "API Response v1"

curl http://$INGRESS_IP/web
# Expected: "Web Application Response"

curl http://$INGRESS_IP/admin
# Expected: "Admin Panel (Restricted)"
```

### Step 10: Host-Based Routing

**Add to `manifests/04-ingress-basic.yaml`**:
```yaml
---
# 3. Host-based routing (Virtual hosting)
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: host-based-ingress
  namespace: ingress-demo
spec:
  ingressClassName: nginx
  rules:
  - host: api.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: backend-api-v1
            port:
              number: 80
  - host: web.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: backend-web
            port:
              number: 80
  - host: admin.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: backend-admin
            port:
              number: 80
```

**Configure local DNS** (for testing):
```bash
# Add to /etc/hosts (Linux/Mac) or C:\Windows\System32\drivers\etc\hosts (Windows)
sudo tee -a /etc/hosts <<EOF
$INGRESS_IP api.example.com
$INGRESS_IP web.example.com
$INGRESS_IP admin.example.com
EOF

# Test host-based routing
curl http://api.example.com/
# Expected: "API Response v1"

curl http://web.example.com/
# Expected: "Web Application Response"

curl http://admin.example.com/
# Expected: "Admin Panel (Restricted)"
```

### Step 11: Combined Routing (Host + Path)

**Add to `manifests/04-ingress-basic.yaml`**:
```yaml
---
# 4. Combined routing (Host + Path)
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: combined-ingress
  namespace: ingress-demo
spec:
  ingressClassName: nginx
  rules:
  - host: app.example.com
    http:
      paths:
      - path: /api
        pathType: Prefix
        backend:
          service:
            name: backend-api-v1
            port:
              number: 80
      - path: /web
        pathType: Prefix
        backend:
          service:
            name: backend-web
            port:
              number: 80
      - path: /
        pathType: Prefix
        backend:
          service:
            name: backend-web
            port:
              number: 80
```

**Test**:
```bash
# Add to /etc/hosts
echo "$INGRESS_IP app.example.com" | sudo tee -a /etc/hosts

curl http://app.example.com/api
# Expected: "API Response v1"

curl http://app.example.com/web
# Expected: "Web Application Response"

curl http://app.example.com/
# Expected: "Web Application Response" (default path)
```

---

## Phase 5: TLS/SSL Termination (35 minutes)

### Step 12: Create TLS Certificates

**Generate self-signed certificate** (testing only):
```bash
# Create certificate for secure.example.com
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout tls.key \
  -out tls.crt \
  -subj "/CN=secure.example.com/O=Demo Org"

# Create Kubernetes secret
kubectl create secret tls tls-secret \
  --cert=tls.crt \
  --key=tls.key \
  -n ingress-demo

# Verify secret
kubectl get secret tls-secret -n ingress-demo

# Cleanup files
rm tls.key tls.crt
```

**For production**: Use cert-manager for automatic Let's Encrypt certificates:
```bash
# Install cert-manager (one-time)
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.2/cert-manager.yaml

# Create ClusterIssuer for Let's Encrypt
cat <<EOF | kubectl apply -f -
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: your-email@example.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx
EOF
```

### Step 13: Create TLS Ingress

**`manifests/05-ingress-tls.yaml`**:
```yaml
# Basic TLS Ingress
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: tls-ingress
  namespace: ingress-demo
  annotations:
    nginx.ingress.kubernetes.io/ssl-redirect: "true"  # Force HTTPS
    nginx.ingress.kubernetes.io/ssl-protocols: "TLSv1.2 TLSv1.3"
spec:
  ingressClassName: nginx
  tls:
  - hosts:
    - secure.example.com
    secretName: tls-secret
  rules:
  - host: secure.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: backend-api-v1
            port:
              number: 80
---
# TLS Ingress with cert-manager (production)
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: tls-auto-ingress
  namespace: ingress-demo
  annotations:
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
spec:
  ingressClassName: nginx
  tls:
  - hosts:
    - auto.example.com
    secretName: auto-tls-secret  # cert-manager creates this
  rules:
  - host: auto.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: backend-api-v1
            port:
              number: 80
```

**Test TLS**:
```bash
kubectl apply -f manifests/05-ingress-tls.yaml

# Add to /etc/hosts
echo "$INGRESS_IP secure.example.com" | sudo tee -a /etc/hosts

# Test HTTP -> HTTPS redirect
curl -v http://secure.example.com/
# Expected: 308 Permanent Redirect to https://

# Test HTTPS (-k skips certificate verification for self-signed)
curl -k https://secure.example.com/
# Expected: "API Response v1"

# Check certificate
openssl s_client -connect secure.example.com:443 -servername secure.example.com
```

---

## Phase 6: Advanced Ingress Patterns (45 minutes)

### Step 14: Canary Deployment (Weight-Based)

**Canary Deployment Strategy**:
```
Production (90% traffic)
    ↓
backend-api-v1

Canary (10% traffic)
    ↓
backend-api-v2
```

**`manifests/06-ingress-advanced.yaml`** (partial):
```yaml
# Production Ingress (primary)
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: canary-production
  namespace: ingress-demo
spec:
  ingressClassName: nginx
  rules:
  - host: canary.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: backend-api-v1
            port:
              number: 80
---
# Canary Ingress (10% weight)
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: canary-deployment
  namespace: ingress-demo
  annotations:
    nginx.ingress.kubernetes.io/canary: "true"
    nginx.ingress.kubernetes.io/canary-weight: "10"  # 10% to canary
spec:
  ingressClassName: nginx
  rules:
  - host: canary.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: backend-api-v2
            port:
              number: 80
```

**Test canary deployment**:
```bash
kubectl apply -f manifests/06-ingress-advanced.yaml

echo "$INGRESS_IP canary.example.com" | sudo tee -a /etc/hosts

# Send 20 requests, expect ~18 v1 and ~2 v2
for i in {1..20}; do
  curl http://canary.example.com/
  echo ""
done

# You should see mix of:
# "API Response v1" (~90%)
# "API Response v2 (CANARY)" (~10%)
```

### Step 15: Header-Based Canary

**Add to `manifests/06-ingress-advanced.yaml`**:
```yaml
---
# Header-based canary (test users)
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: canary-header-production
  namespace: ingress-demo
spec:
  ingressClassName: nginx
  rules:
  - host: canary-header.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: backend-api-v1
            port:
              number: 80
---
# Canary for users with specific header
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: canary-header
  namespace: ingress-demo
  annotations:
    nginx.ingress.kubernetes.io/canary: "true"
    nginx.ingress.kubernetes.io/canary-by-header: "X-Canary"
    nginx.ingress.kubernetes.io/canary-by-header-value: "always"
spec:
  ingressClassName: nginx
  rules:
  - host: canary-header.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: backend-api-v2
            port:
              number: 80
```

**Test header-based canary**:
```bash
echo "$INGRESS_IP canary-header.example.com" | sudo tee -a /etc/hosts

# Normal users get v1
curl http://canary-header.example.com/
# Expected: "API Response v1"

# Beta testers with header get v2
curl -H "X-Canary: always" http://canary-header.example.com/
# Expected: "API Response v2 (CANARY)"
```

### Step 16: Rate Limiting

**Add to `manifests/06-ingress-advanced.yaml`**:
```yaml
---
# Rate limiting (10 req/sec per IP)
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ratelimit-ingress
  namespace: ingress-demo
  annotations:
    nginx.ingress.kubernetes.io/limit-rps: "10"  # Requests per second
    nginx.ingress.kubernetes.io/limit-rpm: "100" # Requests per minute
    nginx.ingress.kubernetes.io/limit-burst-multiplier: "5"
spec:
  ingressClassName: nginx
  rules:
  - host: ratelimit.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: backend-api-v1
            port:
              number: 80
```

**Test rate limiting**:
```bash
kubectl apply -f manifests/06-ingress-advanced.yaml

echo "$INGRESS_IP ratelimit.example.com" | sudo tee -a /etc/hosts

# Rapid requests should get rate limited
for i in {1..20}; do
  curl -w "\nStatus: %{http_code}\n" http://ratelimit.example.com/
  sleep 0.05  # 20 req/sec
done

# Expected: Some requests will return 503 Service Temporarily Unavailable
```

### Step 17: Basic Authentication

**Create auth secret**:
```bash
# Install htpasswd (if not available)
# Ubuntu/Debian: sudo apt-get install apache2-utils
# Mac: brew install htpasswd

# Create password file
htpasswd -c auth admin
# Enter password: admin123

# Create secret
kubectl create secret generic auth-secret \
  --from-file=auth \
  -n ingress-demo

# Cleanup
rm auth
```

**Add to `manifests/06-ingress-advanced.yaml`**:
```yaml
---
# Basic authentication
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: auth-ingress
  namespace: ingress-demo
  annotations:
    nginx.ingress.kubernetes.io/auth-type: basic
    nginx.ingress.kubernetes.io/auth-secret: auth-secret
    nginx.ingress.kubernetes.io/auth-realm: "Authentication Required"
spec:
  ingressClassName: nginx
  rules:
  - host: auth.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: backend-admin
            port:
              number: 80
```

**Test authentication**:
```bash
kubectl apply -f manifests/06-ingress-advanced.yaml

echo "$INGRESS_IP auth.example.com" | sudo tee -a /etc/hosts

# Without credentials - 401 Unauthorized
curl -v http://auth.example.com/

# With credentials
curl -u admin:admin123 http://auth.example.com/
# Expected: "Admin Panel (Restricted)"
```

### Step 18: CORS Configuration

**Add to `manifests/06-ingress-advanced.yaml`**:
```yaml
---
# CORS enabled API
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: cors-ingress
  namespace: ingress-demo
  annotations:
    nginx.ingress.kubernetes.io/enable-cors: "true"
    nginx.ingress.kubernetes.io/cors-allow-origin: "https://app.example.com"
    nginx.ingress.kubernetes.io/cors-allow-methods: "GET, POST, PUT, DELETE, OPTIONS"
    nginx.ingress.kubernetes.io/cors-allow-headers: "Authorization, Content-Type"
    nginx.ingress.kubernetes.io/cors-allow-credentials: "true"
    nginx.ingress.kubernetes.io/cors-max-age: "86400"
spec:
  ingressClassName: nginx
  rules:
  - host: cors-api.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: backend-api-v1
            port:
              number: 80
```

**Test CORS**:
```bash
echo "$INGRESS_IP cors-api.example.com" | sudo tee -a /etc/hosts

# OPTIONS request (preflight)
curl -v -X OPTIONS http://cors-api.example.com/ \
  -H "Origin: https://app.example.com" \
  -H "Access-Control-Request-Method: POST"

# Expected: CORS headers in response:
# Access-Control-Allow-Origin: https://app.example.com
# Access-Control-Allow-Methods: GET, POST, PUT, DELETE, OPTIONS
```

---

## Phase 7: Network Policies (30 minutes)

### Step 19: Implement Network Security

**Key Concept**: Network Policies act as firewall rules for pods, controlling ingress and egress traffic.

**Default Deny All** (Security Best Practice):
```yaml
# manifests/07-network-policies.yaml
# 1. Default deny all ingress
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: deny-all-ingress
  namespace: ingress-demo
spec:
  podSelector: {}  # Applies to all pods
  policyTypes:
  - Ingress
```

**Allow from Ingress Controller**:
```yaml
---
# 2. Allow traffic from Ingress controller
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-from-ingress-controller
  namespace: ingress-demo
spec:
  podSelector:
    matchLabels:
      app: backend-api
  policyTypes:
  - Ingress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8080
```

**Allow Pod-to-Pod** (microservices):
```yaml
---
# 3. Allow pod-to-pod communication (API -> Database)
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-api-to-db
  namespace: ingress-demo
spec:
  podSelector:
    matchLabels:
      app: database
  policyTypes:
  - Ingress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: backend-api
    ports:
    - protocol: TCP
      port: 5432
```

**Allow DNS** (required for all pods):
```yaml
---
# 4. Allow DNS queries (required)
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-dns
  namespace: ingress-demo
spec:
  podSelector: {}
  policyTypes:
  - Egress
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: kube-system
    ports:
    - protocol: UDP
      port: 53
```

**Apply network policies**:
```bash
kubectl apply -f manifests/07-network-policies.yaml

# Verify policies
kubectl get networkpolicies -n ingress-demo

# Test connectivity
# From outside cluster (should work - allowed by Ingress controller policy)
curl http://api.example.com/
# Expected: "API Response v1"

# From test pod (should fail - no policy allows it)
kubectl run test -n ingress-demo --rm -it --image=curlimages/curl --restart=Never \
  -- curl --max-time 5 http://backend-api-v1
# Expected: Timeout (connection blocked by network policy)
```

---

## Phase 8: Production Patterns (20 minutes)

### Step 20: URL Rewriting

**Add to advanced manifest**:
```yaml
# URL rewriting - remove path prefix
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: rewrite-ingress
  namespace: ingress-demo
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /$2
spec:
  ingressClassName: nginx
  rules:
  - host: rewrite.example.com
    http:
      paths:
      - path: /api(/|$)(.*)
        pathType: Prefix
        backend:
          service:
            name: backend-api-v1
            port:
              number: 80
```

**How rewrite works**:
```
Request:  GET /api/users/123
Regex:    /api(/|$)(.*)
Capture:  $2 = users/123
Rewrite:  /$2 = /users/123
Backend:  GET /users/123
```

### Step 21: Custom Timeouts and Body Size

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: custom-limits-ingress
  namespace: ingress-demo
  annotations:
    nginx.ingress.kubernetes.io/proxy-read-timeout: "300"      # 5 minutes
    nginx.ingress.kubernetes.io/proxy-send-timeout: "300"      # 5 minutes
    nginx.ingress.kubernetes.io/proxy-body-size: "100m"        # Max upload size
    nginx.ingress.kubernetes.io/proxy-buffer-size: "8k"
spec:
  ingressClassName: nginx
  rules:
  - host: upload.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: backend-api-v1
            port:
              number: 80
```

### Step 22: Sticky Sessions (Session Affinity)

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: sticky-session-ingress
  namespace: ingress-demo
  annotations:
    nginx.ingress.kubernetes.io/affinity: "cookie"
    nginx.ingress.kubernetes.io/session-cookie-name: "route"
    nginx.ingress.kubernetes.io/session-cookie-expires: "3600"
    nginx.ingress.kubernetes.io/session-cookie-max-age: "3600"
    nginx.ingress.kubernetes.io/session-cookie-samesite: "Lax"
spec:
  ingressClassName: nginx
  rules:
  - host: sticky.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: backend-api-v1
            port:
              number: 80
```

---

## Phase 9: Monitoring and Troubleshooting (25 minutes)

### Step 23: Monitor Ingress Controller

```bash
# Check Ingress controller logs
kubectl logs -n ingress-nginx -l app.kubernetes.io/name=ingress-nginx --tail=100 -f

# Access Ingress controller metrics (Prometheus format)
kubectl port-forward -n ingress-nginx svc/ingress-nginx-controller 10254:10254

# In another terminal
curl http://localhost:10254/metrics | grep nginx_

# Key metrics:
# - nginx_ingress_controller_requests (total requests)
# - nginx_ingress_controller_request_duration_seconds (latency)
# - nginx_ingress_controller_ssl_expire_time_seconds (cert expiry)
```

### Step 24: Common Troubleshooting

**Ingress has no ADDRESS**:
```bash
# Check Ingress controller status
kubectl get pods -n ingress-nginx

# If pending or crashing, check events
kubectl describe pod -n ingress-nginx <pod-name>

# Check Ingress controller service
kubectl get svc -n ingress-nginx ingress-nginx-controller
```

**404 Not Found**:
```bash
# Check Ingress rules
kubectl describe ingress <name> -n ingress-demo

# Verify path matches request
# Common issue: path: /api (Exact) vs path: /api (Prefix)

# Check service exists
kubectl get svc backend-api-v1 -n ingress-demo

# Check service endpoints
kubectl get endpoints backend-api-v1 -n ingress-demo
# Should show pod IPs
```

**502 Bad Gateway**:
```bash
# Backend pods not ready
kubectl get pods -n ingress-demo

# Check pod logs
kubectl logs <pod-name> -n ingress-demo

# Check readiness probe
kubectl describe pod <pod-name> -n ingress-demo | grep -A 5 "Readiness"

# Verify service selector matches pod labels
kubectl get svc backend-api-v1 -n ingress-demo -o yaml | grep selector -A 3
kubectl get pods -n ingress-demo --show-labels
```

**503 Service Unavailable**:
```bash
# No backend pods available
kubectl get pods -n ingress-demo

# Scale deployment
kubectl scale deployment backend-api-v1 --replicas=2 -n ingress-demo

# Check network policies
kubectl get networkpolicies -n ingress-demo

# Temporarily remove network policies to test
kubectl delete networkpolicy deny-all-ingress -n ingress-demo
```

**TLS certificate issues**:
```bash
# Check secret exists
kubectl get secret tls-secret -n ingress-demo

# Check secret type (must be kubernetes.io/tls)
kubectl get secret tls-secret -n ingress-demo -o yaml | grep type:

# Check certificate
kubectl get secret tls-secret -n ingress-demo -o jsonpath='{.data.tls\.crt}' | base64 -d | openssl x509 -text -noout

# Verify hostname matches
# Common Name (CN) or Subject Alternative Name (SAN) must match host
```

---

## Phase 10: Cleanup and Summary (10 minutes)

### Step 25: Automated Cleanup

**`scripts/cleanup.sh`**:
```bash
#!/bin/bash

echo "Cleaning up ingress-demo resources..."

# Delete Ingress resources
kubectl delete ingress --all -n ingress-demo

# Delete services
kubectl delete svc --all -n ingress-demo

# Delete deployments
kubectl delete deployment --all -n ingress-demo

# Delete network policies
kubectl delete networkpolicy --all -n ingress-demo

# Delete secrets
kubectl delete secret auth-secret tls-secret -n ingress-demo --ignore-not-found

# Delete namespace
kubectl delete namespace ingress-demo

# Remove /etc/hosts entries
echo "MANUAL: Remove these lines from /etc/hosts:"
echo "  $INGRESS_IP api.example.com"
echo "  $INGRESS_IP web.example.com"
echo "  $INGRESS_IP admin.example.com"
echo "  ... (and all other test domains)"

echo "Cleanup complete!"
```

```bash
chmod +x scripts/cleanup.sh
./scripts/cleanup.sh
```

---

## Summary

**What You Built**:
- ✅ NGINX Ingress controller installation and verification
- ✅ Path-based routing (/api, /web, /admin)
- ✅ Host-based routing (api.example.com, web.example.com)
- ✅ Combined routing (host + path)
- ✅ TLS/SSL termination with self-signed and auto certificates
- ✅ Canary deployments (weight-based, header-based)
- ✅ Rate limiting (10 req/sec)
- ✅ Basic authentication
- ✅ CORS configuration
- ✅ Network policies for security
- ✅ URL rewriting
- ✅ Custom timeouts and body size limits
- ✅ Sticky sessions

**Key Ingress Annotations**:
```yaml
# Routing
nginx.ingress.kubernetes.io/rewrite-target: /$2

# TLS
nginx.ingress.kubernetes.io/ssl-redirect: "true"
nginx.ingress.kubernetes.io/ssl-protocols: "TLSv1.2 TLSv1.3"

# Canary
nginx.ingress.kubernetes.io/canary: "true"
nginx.ingress.kubernetes.io/canary-weight: "10"
nginx.ingress.kubernetes.io/canary-by-header: "X-Canary"

# Rate limiting
nginx.ingress.kubernetes.io/limit-rps: "10"
nginx.ingress.kubernetes.io/limit-rpm: "100"

# Authentication
nginx.ingress.kubernetes.io/auth-type: basic
nginx.ingress.kubernetes.io/auth-secret: auth-secret

# CORS
nginx.ingress.kubernetes.io/enable-cors: "true"

# Timeouts
nginx.ingress.kubernetes.io/proxy-read-timeout: "300"
nginx.ingress.kubernetes.io/proxy-body-size: "100m"

# Session affinity
nginx.ingress.kubernetes.io/affinity: "cookie"
```

**Production Checklist**:
- [ ] Ingress controller running with 2+ replicas
- [ ] TLS certificates from trusted CA (Let's Encrypt via cert-manager)
- [ ] Network policies implemented (default deny)
- [ ] Rate limiting configured for public APIs
- [ ] Authentication enabled for admin panels
- [ ] Monitoring and alerting set up
- [ ] Resource limits configured on Ingress controller
- [ ] Custom error pages configured
- [ ] Health checks and readiness probes on backends
- [ ] Backup Ingress controller in different availability zone

**Next Steps**:
- Exercise 07: Deploy ML workloads on Kubernetes
- Implement service mesh (Istio, Linkerd) for advanced traffic management
- Add observability (Prometheus, Grafana, Jaeger)
- Explore multi-cluster Ingress
