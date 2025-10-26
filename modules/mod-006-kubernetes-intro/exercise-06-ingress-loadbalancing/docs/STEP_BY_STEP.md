# Step-by-Step Implementation Guide: Ingress & Load Balancing

## Overview

Expose ML applications to the internet with Ingress! Learn path-based routing, TLS termination, load balancing strategies, canary deployments, and production ingress patterns for ML services.

**Time**: 2-3 hours | **Difficulty**: Intermediate to Advanced

---

## Learning Objectives

✅ Understand Kubernetes networking (ClusterIP, NodePort, LoadBalancer, Ingress)
✅ Install and configure Ingress Controllers (Nginx, Traefik)
✅ Implement path-based and host-based routing
✅ Configure TLS/SSL certificates
✅ Implement canary and blue-green deployments
✅ Set up rate limiting and authentication
✅ Load balance ML inference endpoints
✅ Monitor ingress traffic and performance

---

## Service Types Comparison

| Type | Use Case | Accessibility | Cost |
|------|----------|--------------|------|
| ClusterIP | Internal only | Within cluster | Free |
| NodePort | Development | External (port 30000-32767) | Free |
| LoadBalancer | Production | External (cloud LB) | $$$ |
| Ingress | Production | External (layer 7) | $ |

---

## Phase 1: Ingress Controller Setup

### Install Nginx Ingress Controller

```bash
# Using Helm
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
helm repo update

helm install ingress-nginx ingress-nginx/ingress-nginx \
  --namespace ingress-nginx \
  --create-namespace \
  --set controller.replicaCount=2 \
  --set controller.metrics.enabled=true \
  --set controller.podAnnotations."prometheus\.io/scrape"=true

# Verify installation
kubectl get pods -n ingress-nginx
kubectl get svc -n ingress-nginx

# Get LoadBalancer IP
kubectl get svc ingress-nginx-controller -n ingress-nginx
```

### Minikube Setup

```bash
# Enable ingress addon
minikube addons enable ingress

# Verify
kubectl get pods -n ingress-nginx
```

---

## Phase 2: Basic Ingress

### Simple HTTP Ingress

```yaml
# ml-api-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ml-api
  template:
    metadata:
      labels:
        app: ml-api
    spec:
      containers:
      - name: api
        image: ml-api:latest
        ports:
        - containerPort: 8080
---
apiVersion: v1
kind: Service
metadata:
  name: ml-api-service
spec:
  selector:
    app: ml-api
  ports:
  - port: 80
    targetPort: 8080
```

```yaml
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ml-api-ingress
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
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
            name: ml-api-service
            port:
              number: 80
```

```bash
# Apply resources
kubectl apply -f ml-api-deployment.yaml
kubectl apply -f ingress.yaml

# Test (update /etc/hosts or use curl -H)
curl -H "Host: api.example.com" http://<INGRESS_IP>/predict
```

---

## Phase 3: Path-Based Routing

### Multiple Services, Single Host

```yaml
# multi-service-ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ml-platform-ingress
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /$2
spec:
  ingressClassName: nginx
  rules:
  - host: ml.example.com
    http:
      paths:
      # Inference API
      - path: /api(/|$)(.*)
        pathType: ImplementationSpecific
        backend:
          service:
            name: ml-api-service
            port:
              number: 80

      # MLflow UI
      - path: /mlflow(/|$)(.*)
        pathType: ImplementationSpecific
        backend:
          service:
            name: mlflow-service
            port:
              number: 5000

      # TensorBoard
      - path: /tensorboard(/|$)(.*)
        pathType: ImplementationSpecific
        backend:
          service:
            name: tensorboard-service
            port:
              number: 6006

      # Prometheus
      - path: /metrics(/|$)(.*)
        pathType: ImplementationSpecific
        backend:
          service:
            name: prometheus-service
            port:
              number: 9090
```

**Access URLs**:
- `http://ml.example.com/api/predict`
- `http://ml.example.com/mlflow`
- `http://ml.example.com/tensorboard`
- `http://ml.example.com/metrics`

---

## Phase 4: TLS/SSL Configuration

### TLS with Cert-Manager

```bash
# Install cert-manager
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml

# Verify
kubectl get pods -n cert-manager
```

### ClusterIssuer for Let's Encrypt

```yaml
# cluster-issuer.yaml
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: admin@example.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx
```

### Ingress with TLS

```yaml
# tls-ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ml-api-ingress-tls
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
spec:
  ingressClassName: nginx
  tls:
  - hosts:
    - api.example.com
    secretName: api-tls-cert
  rules:
  - host: api.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: ml-api-service
            port:
              number: 80
```

```bash
# Apply
kubectl apply -f cluster-issuer.yaml
kubectl apply -f tls-ingress.yaml

# Check certificate
kubectl get certificate
kubectl describe certificate api-tls-cert

# Test HTTPS
curl https://api.example.com/health
```

---

## Phase 5: Advanced Routing

### Canary Deployment

```yaml
# Production deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-api-v1
spec:
  replicas: 9
  selector:
    matchLabels:
      app: ml-api
      version: v1
  template:
    metadata:
      labels:
        app: ml-api
        version: v1
    spec:
      containers:
      - name: api
        image: ml-api:v1.0.0
---
# Canary deployment (10% traffic)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-api-v2-canary
spec:
  replicas: 1
  selector:
    matchLabels:
      app: ml-api
      version: v2-canary
  template:
    metadata:
      labels:
        app: ml-api
        version: v2-canary
    spec:
      containers:
      - name: api
        image: ml-api:v2.0.0
---
# Service (both versions)
apiVersion: v1
kind: Service
metadata:
  name: ml-api-service
spec:
  selector:
    app: ml-api  # Matches both v1 and v2
  ports:
  - port: 80
    targetPort: 8080
```

**Result**: 90% traffic to v1, 10% to v2

### Header-Based Canary

```yaml
# canary-ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ml-api-canary
  annotations:
    nginx.ingress.kubernetes.io/canary: "true"
    nginx.ingress.kubernetes.io/canary-by-header: "X-Canary"
    nginx.ingress.kubernetes.io/canary-by-header-value: "true"
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
            name: ml-api-v2-service
            port:
              number: 80
```

```bash
# Production traffic
curl https://api.example.com/predict

# Canary traffic (with header)
curl -H "X-Canary: true" https://api.example.com/predict
```

### Weight-Based Canary

```yaml
metadata:
  annotations:
    nginx.ingress.kubernetes.io/canary: "true"
    nginx.ingress.kubernetes.io/canary-weight: "20"  # 20% traffic
```

---

## Phase 6: Load Balancing & Performance

### Connection Limits

```yaml
metadata:
  annotations:
    # Limit connections per IP
    nginx.ingress.kubernetes.io/limit-connections: "10"

    # Rate limiting
    nginx.ingress.kubernetes.io/limit-rps: "100"
    nginx.ingress.kubernetes.io/limit-rpm: "6000"

    # Request body size
    nginx.ingress.kubernetes.io/proxy-body-size: "50m"
```

### Timeouts

```yaml
metadata:
  annotations:
    # Proxy timeouts (for long-running ML inference)
    nginx.ingress.kubernetes.io/proxy-connect-timeout: "300"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "300"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "300"

    # Keep-alive
    nginx.ingress.kubernetes.io/upstream-keepalive-connections: "100"
```

### Session Affinity (Sticky Sessions)

```yaml
metadata:
  annotations:
    # Cookie-based affinity
    nginx.ingress.kubernetes.io/affinity: "cookie"
    nginx.ingress.kubernetes.io/affinity-mode: "persistent"
    nginx.ingress.kubernetes.io/session-cookie-name: "ml-api-affinity"
    nginx.ingress.kubernetes.io/session-cookie-max-age: "3600"
```

---

## Phase 7: Authentication & Security

### Basic Auth

```bash
# Create password file
htpasswd -c auth admin

# Create secret
kubectl create secret generic basic-auth \
  --from-file=auth

# Apply to ingress
```

```yaml
metadata:
  annotations:
    nginx.ingress.kubernetes.io/auth-type: basic
    nginx.ingress.kubernetes.io/auth-secret: basic-auth
    nginx.ingress.kubernetes.io/auth-realm: "ML API - Authentication Required"
```

### OAuth2 Proxy

```yaml
metadata:
  annotations:
    nginx.ingress.kubernetes.io/auth-url: "https://oauth2-proxy.example.com/oauth2/auth"
    nginx.ingress.kubernetes.io/auth-signin: "https://oauth2-proxy.example.com/oauth2/start?rd=$scheme://$host$request_uri"
```

### IP Whitelist

```yaml
metadata:
  annotations:
    nginx.ingress.kubernetes.io/whitelist-source-range: "10.0.0.0/8,172.16.0.0/12"
```

### CORS

```yaml
metadata:
  annotations:
    nginx.ingress.kubernetes.io/enable-cors: "true"
    nginx.ingress.kubernetes.io/cors-allow-origin: "https://frontend.example.com"
    nginx.ingress.kubernetes.io/cors-allow-methods: "GET, POST, OPTIONS"
    nginx.ingress.kubernetes.io/cors-allow-headers: "Authorization, Content-Type"
```

---

## Phase 8: Monitoring & Logging

### Enable Metrics

```yaml
# Install Prometheus ServiceMonitor
apiVersion: v1
kind: Service
metadata:
  name: ingress-nginx-metrics
  namespace: ingress-nginx
  labels:
    app.kubernetes.io/name: ingress-nginx
spec:
  ports:
  - name: metrics
    port: 10254
    targetPort: metrics
  selector:
    app.kubernetes.io/name: ingress-nginx
---
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: ingress-nginx
  namespace: ingress-nginx
spec:
  selector:
    matchLabels:
      app.kubernetes.io/name: ingress-nginx
  endpoints:
  - port: metrics
    interval: 30s
```

### Access Logs

```yaml
# ConfigMap for ingress controller
apiVersion: v1
kind: ConfigMap
metadata:
  name: ingress-nginx-controller
  namespace: ingress-nginx
data:
  log-format-upstream: |
    $remote_addr - $remote_user [$time_local] "$request" $status $body_bytes_sent "$http_referer" "$http_user_agent" $request_length $request_time [$proxy_upstream_name] $upstream_addr $upstream_response_length $upstream_response_time $upstream_status
```

### View Logs

```bash
# Ingress controller logs
kubectl logs -n ingress-nginx deployment/ingress-nginx-controller

# Follow logs
kubectl logs -n ingress-nginx deployment/ingress-nginx-controller -f

# Filter by host
kubectl logs -n ingress-nginx deployment/ingress-nginx-controller | grep "api.example.com"
```

---

## Production Patterns

### Multi-Region Load Balancing

```yaml
# Use external DNS for multi-region
apiVersion: v1
kind: Service
metadata:
  name: ingress-nginx
  annotations:
    external-dns.alpha.kubernetes.io/hostname: api.example.com
    service.beta.kubernetes.io/aws-load-balancer-cross-zone-load-balancing-enabled: "true"
spec:
  type: LoadBalancer
```

### Health Checks

```yaml
metadata:
  annotations:
    # Backend health check
    nginx.ingress.kubernetes.io/upstream-health-check-path: "/health"
    nginx.ingress.kubernetes.io/upstream-health-check-interval: "5s"
```

### High Availability

```bash
# Run multiple ingress controller replicas
helm upgrade ingress-nginx ingress-nginx/ingress-nginx \
  --set controller.replicaCount=3 \
  --set controller.affinity.podAntiAffinity.preferredDuringSchedulingIgnoredDuringExecution[0].weight=100 \
  --set controller.affinity.podAntiAffinity.preferredDuringSchedulingIgnoredDuringExecution[0].podAffinityTerm.topologyKey=kubernetes.io/hostname
```

---

## Troubleshooting

```bash
# Check ingress status
kubectl get ingress
kubectl describe ingress ml-api-ingress

# Test ingress rules
kubectl get ingress ml-api-ingress -o jsonpath='{.spec.rules[*].host}'

# Check backend health
kubectl get endpoints ml-api-service

# Debug with curl
curl -v -H "Host: api.example.com" http://<INGRESS_IP>/

# Check controller logs
kubectl logs -n ingress-nginx -l app.kubernetes.io/name=ingress-nginx

# Validate configuration
kubectl exec -n ingress-nginx deployment/ingress-nginx-controller -- nginx -T
```

---

## Best Practices

✅ Use TLS for all production ingresses
✅ Implement rate limiting
✅ Set appropriate timeouts for ML workloads
✅ Use canary deployments for rollouts
✅ Monitor ingress metrics
✅ Implement authentication
✅ Use multiple ingress controller replicas
✅ Configure health checks
✅ Set resource limits on controllers
✅ Use external DNS for multi-region

---

**Ingress and Load Balancing mastered!** 🌐

**Next Exercise**: ML Workloads on Kubernetes
