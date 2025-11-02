# Implementation Guide: Kubernetes Ingress and Load Balancing for ML Infrastructure

## Table of Contents

1. [Overview](#overview)
2. [Ingress Controllers](#ingress-controllers)
3. [Ingress Resources and Routing Rules](#ingress-resources-and-routing-rules)
4. [TLS/SSL Termination](#tlsssl-termination)
5. [Path-Based and Host-Based Routing](#path-based-and-host-based-routing)
6. [Rate Limiting and Authentication](#rate-limiting-and-authentication)
7. [Load Balancing Strategies](#load-balancing-strategies)
8. [Production ML API Ingress](#production-ml-api-ingress)
9. [Multi-Model Routing](#multi-model-routing)
10. [A/B Testing for ML Models](#ab-testing-for-ml-models)
11. [Canary Deployments for Models](#canary-deployments-for-models)
12. [Rate Limiting per API Key](#rate-limiting-per-api-key)
13. [Advanced ML Patterns](#advanced-ml-patterns)
14. [Monitoring and Observability](#monitoring-and-observability)
15. [Troubleshooting](#troubleshooting)
16. [Best Practices](#best-practices)

## Overview

This implementation guide provides comprehensive coverage of Kubernetes Ingress and Load Balancing with a specific focus on ML infrastructure requirements. You'll learn how to expose ML models, implement intelligent routing, manage traffic for model deployments, and ensure production-grade reliability.

### Why Ingress Matters for ML

In ML infrastructure, Ingress provides:
- **Cost Efficiency**: One load balancer for multiple model services
- **Smart Routing**: Route to different models based on version, features, or user segments
- **Traffic Control**: Gradual rollout of new models with canary deployments
- **Security**: API key validation, rate limiting, and TLS termination
- **Flexibility**: A/B testing, blue-green deployments, and traffic splitting

### Learning Objectives

By completing this guide, you will:
- Deploy and configure multiple ingress controllers
- Implement complex routing patterns for ML APIs
- Set up TLS/SSL for secure model serving
- Configure rate limiting and authentication for ML endpoints
- Deploy canary and A/B testing strategies for models
- Implement multi-model routing and version management
- Monitor and troubleshoot ingress-related issues

## Ingress Controllers

### What is an Ingress Controller?

An Ingress Controller is a specialized load balancer that:
- Watches for Ingress resources in the cluster
- Configures routing rules dynamically
- Handles TLS termination
- Provides advanced L7 (HTTP/HTTPS) load balancing

### Popular Ingress Controllers

#### 1. NGINX Ingress Controller

**Best for**: General purpose, well-documented, widely adopted

```bash
# Install NGINX Ingress Controller
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.9.4/deploy/static/provider/cloud/deploy.yaml

# Verify installation
kubectl get pods -n ingress-nginx
kubectl wait --namespace ingress-nginx \
  --for=condition=ready pod \
  --selector=app.kubernetes.io/component=controller \
  --timeout=120s
```

**Key Features**:
- Advanced rate limiting
- Extensive annotation support
- Built-in metrics (Prometheus)
- WebSocket and gRPC support
- Canary deployments

**NGINX ConfigMap for ML Workloads**:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: ingress-nginx-controller
  namespace: ingress-nginx
data:
  # Increase timeouts for model inference
  proxy-connect-timeout: "300"
  proxy-send-timeout: "300"
  proxy-read-timeout: "300"

  # Increase body size for large payloads
  proxy-body-size: "100m"

  # Enable real IP forwarding
  use-forwarded-headers: "true"
  compute-full-forwarded-for: "true"

  # Connection settings
  keep-alive: "75"
  keep-alive-requests: "1000"
  upstream-keepalive-connections: "200"

  # Enable gzip for JSON responses
  enable-gzip: "true"
  gzip-types: "application/json application/javascript text/css text/plain"
```

#### 2. Traefik

**Best for**: Microservices, automatic service discovery, modern stack

```bash
# Install Traefik via Helm
helm repo add traefik https://traefik.github.io/charts
helm repo update

helm install traefik traefik/traefik \
  --namespace traefik \
  --create-namespace \
  --set "ports.websecure.tls.enabled=true" \
  --set "providers.kubernetesCRD.enabled=true" \
  --set "providers.kubernetesIngress.enabled=true"
```

**Key Features**:
- Automatic HTTPS with Let's Encrypt
- Native support for multiple providers
- Built-in dashboard
- Middleware system for request/response modification
- Dynamic configuration

**Traefik IngressRoute for ML API**:

```yaml
apiVersion: traefik.containo.us/v1alpha1
kind: IngressRoute
metadata:
  name: ml-model-route
  namespace: ml-serving
spec:
  entryPoints:
    - websecure
  routes:
    - match: Host(`api.ml-platform.com`) && PathPrefix(`/v1/predict`)
      kind: Rule
      services:
        - name: model-service-v1
          port: 8080
      middlewares:
        - name: rate-limit
        - name: auth-api-key
  tls:
    secretName: ml-platform-tls
```

#### 3. Istio Ingress Gateway

**Best for**: Service mesh integration, advanced traffic management, observability

```bash
# Install Istio
curl -L https://istio.io/downloadIstio | sh -
cd istio-*/bin
./istioctl install --set profile=default -y

# Enable Istio injection
kubectl label namespace ml-serving istio-injection=enabled
```

**Key Features**:
- Service mesh capabilities (mTLS, circuit breaking, retries)
- Advanced traffic splitting (percentage-based)
- Distributed tracing (Jaeger, Zipkin)
- Fine-grained access control
- Multi-cluster support

**Istio VirtualService for ML Models**:

```yaml
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: ml-model-routing
  namespace: ml-serving
spec:
  hosts:
    - api.ml-platform.com
  gateways:
    - ml-gateway
  http:
    - match:
        - uri:
            prefix: "/v1/models/sentiment"
      route:
        - destination:
            host: sentiment-model
            subset: v2
          weight: 90
        - destination:
            host: sentiment-model
            subset: v3
          weight: 10
    - match:
        - uri:
            prefix: "/v1/models/classification"
      route:
        - destination:
            host: classification-model
            port:
              number: 8080
```

### Controller Comparison for ML Use Cases

| Feature | NGINX | Traefik | Istio |
|---------|-------|---------|-------|
| **Ease of Setup** | Medium | Easy | Complex |
| **Performance** | Excellent | Good | Good |
| **Rate Limiting** | Per IP/Key | Middleware | Envoy Filter |
| **Canary Deployments** | Annotations | Weighted Routes | Traffic Split |
| **ML-Specific Features** | Good timeout control | Auto HTTPS | Advanced observability |
| **Production Maturity** | Very High | High | High |
| **Learning Curve** | Medium | Low | High |
| **Best For ML** | Simple deployments | Rapid iteration | Complex scenarios |

## Ingress Resources and Routing Rules

### Basic Ingress Structure

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: example-ingress
  namespace: ml-serving
  annotations:
    # Controller-specific annotations
spec:
  ingressClassName: nginx  # or traefik, istio
  rules:
    - host: api.example.com
      http:
        paths:
          - path: /predict
            pathType: Prefix
            backend:
              service:
                name: model-service
                port:
                  number: 8080
  tls:
    - hosts:
        - api.example.com
      secretName: api-tls-cert
```

### Path Types

**Prefix**: Matches based on URL path prefix split by `/`

```yaml
# Matches: /predict, /predict/, /predict/image, /predict/text/sentiment
path: /predict
pathType: Prefix
```

**Exact**: Exact URL path match

```yaml
# Matches only: /predict (not /predict/ or /predict/image)
path: /predict
pathType: Exact
```

**ImplementationSpecific**: Controller-specific matching (supports regex)

```yaml
# NGINX regex: /api/v[0-9]+/predict
path: /api/v[0-9]+/predict
pathType: ImplementationSpecific
```

### Multiple Rules Example

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ml-api-ingress
  namespace: ml-serving
spec:
  ingressClassName: nginx
  rules:
    # Rule 1: Health checks
    - http:
        paths:
          - path: /health
            pathType: Exact
            backend:
              service:
                name: health-service
                port:
                  number: 8080

    # Rule 2: Versioned API
    - host: api.ml-platform.com
      http:
        paths:
          - path: /v1
            pathType: Prefix
            backend:
              service:
                name: api-v1
                port:
                  number: 8080
          - path: /v2
            pathType: Prefix
            backend:
              service:
                name: api-v2
                port:
                  number: 8080

    # Rule 3: Model-specific endpoints
    - host: models.ml-platform.com
      http:
        paths:
          - path: /sentiment
            pathType: Prefix
            backend:
              service:
                name: sentiment-model
                port:
                  number: 8080
          - path: /classification
            pathType: Prefix
            backend:
              service:
                name: classification-model
                port:
                  number: 8080
```

## TLS/SSL Termination

### Why TLS Termination at Ingress?

- **Centralized Certificate Management**: One place to manage certificates
- **Backend Simplification**: Services don't need TLS configuration
- **Performance**: Offload encryption/decryption from application pods
- **Compliance**: Ensure all external traffic is encrypted

### Creating TLS Certificates

#### Self-Signed Certificate (Development)

```bash
# Generate private key and certificate
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout tls.key \
  -out tls.crt \
  -subj "/CN=api.ml-platform.com/O=ml-platform"

# Create Kubernetes secret
kubectl create secret tls ml-platform-tls \
  --cert=tls.crt \
  --key=tls.key \
  --namespace=ml-serving
```

#### Let's Encrypt with cert-manager (Production)

```bash
# Install cert-manager
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml

# Create ClusterIssuer for Let's Encrypt
cat <<EOF | kubectl apply -f -
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: admin@ml-platform.com
    privateKeySecretRef:
      name: letsencrypt-prod-key
    solvers:
      - http01:
          ingress:
            class: nginx
EOF
```

### TLS Ingress Configuration

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ml-api-tls
  namespace: ml-serving
  annotations:
    # Force HTTPS redirect
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    # Use strong TLS protocols
    nginx.ingress.kubernetes.io/ssl-protocols: "TLSv1.2 TLSv1.3"
    # Strong cipher suites
    nginx.ingress.kubernetes.io/ssl-ciphers: "ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384"
    # HSTS header (force HTTPS for 1 year)
    nginx.ingress.kubernetes.io/hsts: "true"
    nginx.ingress.kubernetes.io/hsts-max-age: "31536000"
    nginx.ingress.kubernetes.io/hsts-include-subdomains: "true"
    # Automatic certificate with cert-manager
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  ingressClassName: nginx
  tls:
    - hosts:
        - api.ml-platform.com
        - models.ml-platform.com
      secretName: ml-platform-tls-auto
  rules:
    - host: api.ml-platform.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: ml-api-service
                port:
                  number: 8080
```

### Wildcard Certificates

```yaml
apiVersion: cert-manager.io/v1
kind: Certificate
metadata:
  name: wildcard-ml-platform
  namespace: ml-serving
spec:
  secretName: wildcard-ml-platform-tls
  issuerRef:
    name: letsencrypt-prod
    kind: ClusterIssuer
  dnsNames:
    - "*.ml-platform.com"
    - "ml-platform.com"
```

## Path-Based and Host-Based Routing

### Path-Based Routing for ML Models

Route different model types to different services based on URL path:

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ml-models-path-routing
  namespace: ml-serving
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /$2
spec:
  ingressClassName: nginx
  rules:
    - host: api.ml-platform.com
      http:
        paths:
          # Sentiment analysis model
          - path: /models/sentiment(/|$)(.*)
            pathType: ImplementationSpecific
            backend:
              service:
                name: sentiment-model-service
                port:
                  number: 8080

          # Image classification model
          - path: /models/classification(/|$)(.*)
            pathType: ImplementationSpecific
            backend:
              service:
                name: classification-model-service
                port:
                  number: 8080

          # Named Entity Recognition
          - path: /models/ner(/|$)(.*)
            pathType: ImplementationSpecific
            backend:
              service:
                name: ner-model-service
                port:
                  number: 8080

          # Object detection
          - path: /models/detection(/|$)(.*)
            pathType: ImplementationSpecific
            backend:
              service:
                name: detection-model-service
                port:
                  number: 8080
```

**Request Examples**:
- `POST /models/sentiment/predict` → `sentiment-model-service:8080/predict`
- `POST /models/classification/predict` → `classification-model-service:8080/predict`

### Host-Based Routing for Environments

Different hostnames for different environments or model versions:

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ml-environment-routing
  namespace: ml-serving
spec:
  ingressClassName: nginx
  rules:
    # Development environment
    - host: dev.api.ml-platform.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: ml-api-dev
                port:
                  number: 8080

    # Staging environment
    - host: staging.api.ml-platform.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: ml-api-staging
                port:
                  number: 8080

    # Production environment
    - host: api.ml-platform.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: ml-api-production
                port:
                  number: 8080
```

### Combined Routing: Host + Path

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ml-combined-routing
  namespace: ml-serving
spec:
  ingressClassName: nginx
  rules:
    # Production models
    - host: models.ml-platform.com
      http:
        paths:
          - path: /v1/sentiment
            pathType: Prefix
            backend:
              service:
                name: sentiment-v1
                port:
                  number: 8080
          - path: /v2/sentiment
            pathType: Prefix
            backend:
              service:
                name: sentiment-v2
                port:
                  number: 8080

    # Experimental models
    - host: experimental.ml-platform.com
      http:
        paths:
          - path: /sentiment
            pathType: Prefix
            backend:
              service:
                name: sentiment-experimental
                port:
                  number: 8080
```

## Rate Limiting and Authentication

### Rate Limiting Strategies

#### Per-IP Rate Limiting

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ml-api-rate-limited
  namespace: ml-serving
  annotations:
    # 10 requests per second per IP
    nginx.ingress.kubernetes.io/limit-rps: "10"
    # 100 requests per minute per IP
    nginx.ingress.kubernetes.io/limit-rpm: "100"
    # Maximum 5 concurrent connections per IP
    nginx.ingress.kubernetes.io/limit-connections: "5"
    # Burst size (allow temporary spikes)
    nginx.ingress.kubernetes.io/limit-burst-multiplier: "2"
spec:
  ingressClassName: nginx
  rules:
    - host: api.ml-platform.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: ml-api-service
                port:
                  number: 8080
```

#### Custom Rate Limiting by Header

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ml-api-custom-rate-limit
  namespace: ml-serving
  annotations:
    nginx.ingress.kubernetes.io/configuration-snippet: |
      # Rate limit by API key header
      limit_req_zone $http_x_api_key zone=api_key_limit:10m rate=100r/m;
      limit_req zone=api_key_limit burst=20 nodelay;
spec:
  ingressClassName: nginx
  rules:
    - host: api.ml-platform.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: ml-api-service
                port:
                  number: 8080
```

### Authentication Methods

#### 1. Basic Authentication

```bash
# Create htpasswd file
htpasswd -c auth ml-admin
# Enter password when prompted

# Create secret
kubectl create secret generic ml-api-auth \
  --from-file=auth \
  --namespace=ml-serving
```

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ml-api-basic-auth
  namespace: ml-serving
  annotations:
    nginx.ingress.kubernetes.io/auth-type: basic
    nginx.ingress.kubernetes.io/auth-secret: ml-api-auth
    nginx.ingress.kubernetes.io/auth-realm: 'ML Platform - Authentication Required'
spec:
  ingressClassName: nginx
  rules:
    - host: admin.ml-platform.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: ml-admin-service
                port:
                  number: 8080
```

#### 2. API Key Validation

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ml-api-key-auth
  namespace: ml-serving
  annotations:
    nginx.ingress.kubernetes.io/configuration-snippet: |
      # Validate API key header
      if ($http_x_api_key = "") {
        return 401 "API Key Required";
      }

      # Forward to auth service for validation
      auth_request /auth/validate;
      auth_request_set $auth_user $upstream_http_x_auth_user;
      auth_request_set $auth_tier $upstream_http_x_auth_tier;

      # Add auth info to backend request
      proxy_set_header X-Auth-User $auth_user;
      proxy_set_header X-Auth-Tier $auth_tier;
spec:
  ingressClassName: nginx
  rules:
    - host: api.ml-platform.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: ml-api-service
                port:
                  number: 8080
```

#### 3. OAuth2 Authentication

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ml-api-oauth2
  namespace: ml-serving
  annotations:
    nginx.ingress.kubernetes.io/auth-url: "https://oauth2-proxy.ml-platform.com/oauth2/auth"
    nginx.ingress.kubernetes.io/auth-signin: "https://oauth2-proxy.ml-platform.com/oauth2/start?rd=$escaped_request_uri"
    nginx.ingress.kubernetes.io/auth-response-headers: "X-Auth-Request-User,X-Auth-Request-Email"
spec:
  ingressClassName: nginx
  rules:
    - host: secure.ml-platform.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: ml-api-service
                port:
                  number: 8080
```

## Load Balancing Strategies

### Kubernetes Service Load Balancing

Default: Round-robin across healthy pods

```yaml
apiVersion: v1
kind: Service
metadata:
  name: model-service
  namespace: ml-serving
spec:
  selector:
    app: model
    version: v1
  ports:
    - port: 8080
      targetPort: 8080
  # Default: Round-robin
  sessionAffinity: None
```

### Session Affinity (Sticky Sessions)

Useful for stateful ML applications:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: model-service-sticky
  namespace: ml-serving
spec:
  selector:
    app: model
  ports:
    - port: 8080
      targetPort: 8080
  # Route same client to same pod
  sessionAffinity: ClientIP
  sessionAffinityConfig:
    clientIP:
      timeoutSeconds: 3600  # 1 hour
```

### Ingress-Level Session Affinity

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ml-api-sticky
  namespace: ml-serving
  annotations:
    nginx.ingress.kubernetes.io/affinity: "cookie"
    nginx.ingress.kubernetes.io/session-cookie-name: "ml-route"
    nginx.ingress.kubernetes.io/session-cookie-expires: "7200"
    nginx.ingress.kubernetes.io/session-cookie-max-age: "7200"
    nginx.ingress.kubernetes.io/session-cookie-path: "/predict"
    nginx.ingress.kubernetes.io/affinity-mode: "persistent"
spec:
  ingressClassName: nginx
  rules:
    - host: api.ml-platform.com
      http:
        paths:
          - path: /predict
            pathType: Prefix
            backend:
              service:
                name: model-service
                port:
                  number: 8080
```

### Load Balancing Algorithms

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ml-api-lb-algorithm
  namespace: ml-serving
  annotations:
    # Hash-based routing (consistent hashing by request URI)
    nginx.ingress.kubernetes.io/upstream-hash-by: "$request_uri"

    # Keep-alive for better performance
    nginx.ingress.kubernetes.io/upstream-keepalive-connections: "200"
    nginx.ingress.kubernetes.io/upstream-keepalive-timeout: "60"

    # Connection pooling
    nginx.ingress.kubernetes.io/upstream-keepalive-requests: "10000"
spec:
  ingressClassName: nginx
  rules:
    - host: api.ml-platform.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: model-service
                port:
                  number: 8080
```

## Production ML API Ingress

### Complete Production ML API Configuration

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: production-ml-api
  namespace: ml-serving
  annotations:
    # TLS Configuration
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/ssl-protocols: "TLSv1.2 TLSv1.3"

    # Timeouts (important for ML inference)
    nginx.ingress.kubernetes.io/proxy-connect-timeout: "300"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "300"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "300"

    # Body size (for large input data)
    nginx.ingress.kubernetes.io/proxy-body-size: "100m"

    # Rate limiting
    nginx.ingress.kubernetes.io/limit-rps: "50"
    nginx.ingress.kubernetes.io/limit-connections: "20"

    # CORS for web clients
    nginx.ingress.kubernetes.io/enable-cors: "true"
    nginx.ingress.kubernetes.io/cors-allow-origin: "https://app.ml-platform.com"
    nginx.ingress.kubernetes.io/cors-allow-methods: "POST, GET, OPTIONS"
    nginx.ingress.kubernetes.io/cors-allow-headers: "DNT,X-CustomHeader,Keep-Alive,User-Agent,X-Requested-With,If-Modified-Since,Cache-Control,Content-Type,Authorization,X-API-Key"
    nginx.ingress.kubernetes.io/cors-allow-credentials: "true"

    # Security headers
    nginx.ingress.kubernetes.io/configuration-snippet: |
      more_set_headers "X-Content-Type-Options: nosniff";
      more_set_headers "X-Frame-Options: DENY";
      more_set_headers "X-XSS-Protection: 1; mode=block";
      more_set_headers "Referrer-Policy: strict-origin-when-cross-origin";
      more_set_headers "Content-Security-Policy: default-src 'self'";

    # Monitoring
    nginx.ingress.kubernetes.io/enable-access-log: "true"
    nginx.ingress.kubernetes.io/enable-modsecurity: "true"

    # Load balancing
    nginx.ingress.kubernetes.io/upstream-keepalive-connections: "200"
    nginx.ingress.kubernetes.io/upstream-keepalive-timeout: "60"
spec:
  ingressClassName: nginx
  tls:
    - hosts:
        - api.ml-platform.com
      secretName: ml-platform-tls
  rules:
    - host: api.ml-platform.com
      http:
        paths:
          # Health check endpoint (no auth)
          - path: /health
            pathType: Exact
            backend:
              service:
                name: health-service
                port:
                  number: 8080

          # Metrics endpoint (internal only)
          - path: /metrics
            pathType: Exact
            backend:
              service:
                name: metrics-service
                port:
                  number: 9090

          # Main prediction API
          - path: /v1/predict
            pathType: Prefix
            backend:
              service:
                name: model-service-v1
                port:
                  number: 8080
```

## Multi-Model Routing

### Routing by Model Type

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: multi-model-routing
  namespace: ml-serving
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /predict
spec:
  ingressClassName: nginx
  rules:
    - host: models.ml-platform.com
      http:
        paths:
          # Text models
          - path: /text/sentiment
            pathType: Prefix
            backend:
              service:
                name: sentiment-analysis-service
                port:
                  number: 8080

          - path: /text/summarization
            pathType: Prefix
            backend:
              service:
                name: text-summarization-service
                port:
                  number: 8080

          - path: /text/translation
            pathType: Prefix
            backend:
              service:
                name: translation-service
                port:
                  number: 8080

          # Vision models
          - path: /vision/classification
            pathType: Prefix
            backend:
              service:
                name: image-classification-service
                port:
                  number: 8080

          - path: /vision/detection
            pathType: Prefix
            backend:
              service:
                name: object-detection-service
                port:
                  number: 8080

          - path: /vision/segmentation
            pathType: Prefix
            backend:
              service:
                name: segmentation-service
                port:
                  number: 8080

          # Audio models
          - path: /audio/transcription
            pathType: Prefix
            backend:
              service:
                name: speech-to-text-service
                port:
                  number: 8080

          - path: /audio/classification
            pathType: Prefix
            backend:
              service:
                name: audio-classification-service
                port:
                  number: 8080
```

### Version-Based Routing

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: model-version-routing
  namespace: ml-serving
spec:
  ingressClassName: nginx
  rules:
    - host: api.ml-platform.com
      http:
        paths:
          # Legacy API (v1)
          - path: /v1/models/sentiment
            pathType: Prefix
            backend:
              service:
                name: sentiment-v1-service
                port:
                  number: 8080

          # Current stable (v2)
          - path: /v2/models/sentiment
            pathType: Prefix
            backend:
              service:
                name: sentiment-v2-service
                port:
                  number: 8080

          # Beta (v3)
          - path: /v3/models/sentiment
            pathType: Prefix
            backend:
              service:
                name: sentiment-v3-beta-service
                port:
                  number: 8080
```

## A/B Testing for ML Models

### Simple A/B Test (50/50 Split)

Using NGINX canary with equal weight:

```yaml
---
# Production traffic (50%)
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: model-a
  namespace: ml-serving
spec:
  ingressClassName: nginx
  rules:
    - host: predict.ml-platform.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: model-variant-a
                port:
                  number: 8080

---
# A/B Test variant (50%)
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: model-b
  namespace: ml-serving
  annotations:
    nginx.ingress.kubernetes.io/canary: "true"
    nginx.ingress.kubernetes.io/canary-weight: "50"
spec:
  ingressClassName: nginx
  rules:
    - host: predict.ml-platform.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: model-variant-b
                port:
                  number: 8080
```

### User-Segment Based A/B Testing

Route specific users to variant B:

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: model-ab-user-segment
  namespace: ml-serving
  annotations:
    nginx.ingress.kubernetes.io/canary: "true"
    # Route users with header "X-User-Segment: beta" to variant B
    nginx.ingress.kubernetes.io/canary-by-header: "X-User-Segment"
    nginx.ingress.kubernetes.io/canary-by-header-value: "beta"
spec:
  ingressClassName: nginx
  rules:
    - host: predict.ml-platform.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: model-variant-b
                port:
                  number: 8080
```

### Cookie-Based A/B Testing

Consistent user experience across sessions:

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: model-ab-cookie
  namespace: ml-serving
  annotations:
    nginx.ingress.kubernetes.io/canary: "true"
    # Route if cookie "ab_test=variant_b"
    nginx.ingress.kubernetes.io/canary-by-cookie: "ab_test"
spec:
  ingressClassName: nginx
  rules:
    - host: predict.ml-platform.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: model-variant-b
                port:
                  number: 8080
```

## Canary Deployments for Models

### Progressive Traffic Shifting

**Step 1: Deploy Canary with 10% Traffic**

```yaml
---
# Stable production model (90%)
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: model-stable
  namespace: ml-serving
  labels:
    version: stable
spec:
  ingressClassName: nginx
  rules:
    - host: api.ml-platform.com
      http:
        paths:
          - path: /predict
            pathType: Prefix
            backend:
              service:
                name: sentiment-model-v2
                port:
                  number: 8080

---
# Canary model (10%)
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: model-canary
  namespace: ml-serving
  labels:
    version: canary
  annotations:
    nginx.ingress.kubernetes.io/canary: "true"
    nginx.ingress.kubernetes.io/canary-weight: "10"
spec:
  ingressClassName: nginx
  rules:
    - host: api.ml-platform.com
      http:
        paths:
          - path: /predict
            pathType: Prefix
            backend:
              service:
                name: sentiment-model-v3-canary
                port:
                  number: 8080
```

**Step 2: Gradually Increase Canary Traffic**

```bash
# Increase to 25%
kubectl annotate ingress model-canary \
  nginx.ingress.kubernetes.io/canary-weight=25 \
  --overwrite \
  -n ml-serving

# Monitor metrics, errors, latency
# If successful, increase to 50%
kubectl annotate ingress model-canary \
  nginx.ingress.kubernetes.io/canary-weight=50 \
  --overwrite \
  -n ml-serving

# Continue monitoring
# If successful, increase to 100%
kubectl annotate ingress model-canary \
  nginx.ingress.kubernetes.io/canary-weight=100 \
  --overwrite \
  -n ml-serving
```

**Step 3: Promote Canary to Stable**

```bash
# Update stable ingress to point to new version
kubectl patch ingress model-stable -n ml-serving --type=json \
  -p='[{"op": "replace", "path": "/spec/rules/0/http/paths/0/backend/service/name", "value": "sentiment-model-v3-canary"}]'

# Delete canary ingress
kubectl delete ingress model-canary -n ml-serving
```

### Header-Based Canary for Internal Testing

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: model-canary-internal
  namespace: ml-serving
  annotations:
    nginx.ingress.kubernetes.io/canary: "true"
    # Only route if header "X-Canary-Test: enabled"
    nginx.ingress.kubernetes.io/canary-by-header: "X-Canary-Test"
    nginx.ingress.kubernetes.io/canary-by-header-value: "enabled"
spec:
  ingressClassName: nginx
  rules:
    - host: api.ml-platform.com
      http:
        paths:
          - path: /predict
            pathType: Prefix
            backend:
              service:
                name: sentiment-model-v3-canary
                port:
                  number: 8080
```

Testing:

```bash
# Regular users get stable version
curl https://api.ml-platform.com/predict -d '{"text": "Great product!"}'

# Internal testers get canary version
curl https://api.ml-platform.com/predict \
  -H "X-Canary-Test: enabled" \
  -d '{"text": "Great product!"}'
```

## Rate Limiting per API Key

### API Key-Based Rate Limiting

**ConfigMap for Rate Limit Zones**:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: nginx-rate-limit-config
  namespace: ingress-nginx
data:
  http-snippet: |
    # Define rate limit zones per API tier
    limit_req_zone $http_x_api_key zone=free_tier:10m rate=10r/m;
    limit_req_zone $http_x_api_key zone=pro_tier:10m rate=100r/m;
    limit_req_zone $http_x_api_key zone=enterprise_tier:10m rate=1000r/m;

    # API key to tier mapping (in production, use external auth service)
    map $http_x_api_key $rate_limit_zone {
      default                           free_tier;
      "~^free_.*"                       free_tier;
      "~^pro_.*"                        pro_tier;
      "~^enterprise_.*"                 enterprise_tier;
    }
```

**Ingress with API Key Rate Limiting**:

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ml-api-key-rate-limit
  namespace: ml-serving
  annotations:
    nginx.ingress.kubernetes.io/configuration-snippet: |
      # Reject requests without API key
      if ($http_x_api_key = "") {
        return 401 '{"error": "API Key required in X-API-Key header"}';
      }

      # Apply rate limiting based on tier
      if ($rate_limit_zone = "free_tier") {
        limit_req zone=free_tier burst=5 nodelay;
      }
      if ($rate_limit_zone = "pro_tier") {
        limit_req zone=pro_tier burst=20 nodelay;
      }
      if ($rate_limit_zone = "enterprise_tier") {
        limit_req zone=enterprise_tier burst=50 nodelay;
      }

      # Add rate limit info to response headers
      add_header X-RateLimit-Zone $rate_limit_zone;
spec:
  ingressClassName: nginx
  rules:
    - host: api.ml-platform.com
      http:
        paths:
          - path: /predict
            pathType: Prefix
            backend:
              service:
                name: model-service
                port:
                  number: 8080
```

### External Auth Service for API Key Validation

**Auth Service Deployment**:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api-key-auth-service
  namespace: ml-serving
spec:
  replicas: 3
  selector:
    matchLabels:
      app: api-key-auth
  template:
    metadata:
      labels:
        app: api-key-auth
    spec:
      containers:
        - name: auth-service
          image: ml-platform/api-key-auth:v1
          ports:
            - containerPort: 8080
          env:
            - name: DATABASE_URL
              valueFrom:
                secretKeyRef:
                  name: api-key-db-credentials
                  key: url
---
apiVersion: v1
kind: Service
metadata:
  name: api-key-auth-service
  namespace: ml-serving
spec:
  selector:
    app: api-key-auth
  ports:
    - port: 8080
      targetPort: 8080
```

**Ingress with External Auth**:

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ml-api-external-auth
  namespace: ml-serving
  annotations:
    # Use external auth service
    nginx.ingress.kubernetes.io/auth-url: "http://api-key-auth-service.ml-serving.svc.cluster.local:8080/validate"
    nginx.ingress.kubernetes.io/auth-response-headers: "X-User-ID,X-User-Tier,X-Rate-Limit"

    # Apply rate limit from auth service response
    nginx.ingress.kubernetes.io/configuration-snippet: |
      # Auth service returns X-Rate-Limit header (e.g., "100r/m")
      limit_req_zone $http_x_api_key zone=dynamic_limit:10m rate=$upstream_http_x_rate_limit;
      limit_req zone=dynamic_limit burst=10 nodelay;

      # Add user info to backend request
      proxy_set_header X-User-ID $upstream_http_x_user_id;
      proxy_set_header X-User-Tier $upstream_http_x_user_tier;
spec:
  ingressClassName: nginx
  rules:
    - host: api.ml-platform.com
      http:
        paths:
          - path: /predict
            pathType: Prefix
            backend:
              service:
                name: model-service
                port:
                  number: 8080
```

## Advanced ML Patterns

### Model Ensemble Routing

Route to multiple models and aggregate results:

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: model-ensemble
  namespace: ml-serving
  annotations:
    nginx.ingress.kubernetes.io/configuration-snippet: |
      # This is a simplified example
      # In production, use a dedicated ensemble service
      location /ensemble/predict {
        # Send to aggregator service which handles ensemble logic
        proxy_pass http://ensemble-aggregator.ml-serving.svc.cluster.local:8080;
      }
spec:
  ingressClassName: nginx
  rules:
    - host: api.ml-platform.com
      http:
        paths:
          - path: /ensemble/predict
            pathType: Prefix
            backend:
              service:
                name: ensemble-aggregator-service
                port:
                  number: 8080
```

### Request Mirroring for Shadow Testing

Test new model without affecting production:

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: model-shadow-testing
  namespace: ml-serving
  annotations:
    # Mirror 10% of traffic to shadow model
    nginx.ingress.kubernetes.io/configuration-snippet: |
      set $mirror_backend "";

      # Randomly select 10% of requests for mirroring
      if ($request_id ~* "[0-9a-f]$") {
        set $mirror_backend "http://sentiment-model-v3-shadow.ml-serving.svc.cluster.local:8080";
      }

      if ($mirror_backend != "") {
        mirror /mirror;
        mirror_request_body on;
      }

      location /mirror {
        internal;
        proxy_pass $mirror_backend;
        proxy_set_header X-Mirror-Request "true";
      }
spec:
  ingressClassName: nginx
  rules:
    - host: api.ml-platform.com
      http:
        paths:
          - path: /predict
            pathType: Prefix
            backend:
              service:
                name: sentiment-model-v2
                port:
                  number: 8080
```

### Geographic Routing

Route based on client location:

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: model-geo-routing
  namespace: ml-serving
  annotations:
    # Use GeoIP database (requires GeoIP module in NGINX)
    nginx.ingress.kubernetes.io/configuration-snippet: |
      # Route based on country code
      set $backend_service "model-service-us";

      if ($geoip_country_code = "EU") {
        set $backend_service "model-service-eu";
      }

      if ($geoip_country_code = "ASIA") {
        set $backend_service "model-service-asia";
      }

      proxy_pass http://$backend_service.ml-serving.svc.cluster.local:8080;
spec:
  ingressClassName: nginx
  rules:
    - host: api.ml-platform.com
      http:
        paths:
          - path: /predict
            pathType: Prefix
            backend:
              service:
                name: model-service-us
                port:
                  number: 8080
```

## Monitoring and Observability

### Prometheus Metrics

Enable NGINX Ingress metrics:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: ingress-nginx-controller
  namespace: ingress-nginx
data:
  enable-prometheus-metrics: "true"
  prometheus-port: "10254"
```

**ServiceMonitor for Prometheus**:

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: nginx-ingress
  namespace: ingress-nginx
spec:
  selector:
    matchLabels:
      app.kubernetes.io/name: ingress-nginx
  endpoints:
    - port: metrics
      interval: 30s
```

### Custom Metrics and Logging

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ml-api-monitoring
  namespace: ml-serving
  annotations:
    # Custom log format
    nginx.ingress.kubernetes.io/configuration-snippet: |
      # Log detailed request info for ML APIs
      log_format ml_api_logs '$remote_addr - $remote_user [$time_local] '
                             '"$request" $status $body_bytes_sent '
                             '"$http_referer" "$http_user_agent" '
                             '$request_time $upstream_response_time '
                             '$http_x_api_key $http_x_model_version';

      access_log /var/log/nginx/ml_api_access.log ml_api_logs;

      # Add response time headers
      add_header X-Response-Time $request_time;
      add_header X-Upstream-Response-Time $upstream_response_time;
spec:
  ingressClassName: nginx
  rules:
    - host: api.ml-platform.com
      http:
        paths:
          - path: /predict
            pathType: Prefix
            backend:
              service:
                name: model-service
                port:
                  number: 8080
```

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: 502 Bad Gateway

**Symptoms**: Requests return 502 error

**Diagnosis**:
```bash
# Check if backend pods are running
kubectl get pods -n ml-serving -l app=model-service

# Check pod logs
kubectl logs -n ml-serving -l app=model-service --tail=50

# Check service endpoints
kubectl get endpoints model-service -n ml-serving

# Check ingress controller logs
kubectl logs -n ingress-nginx -l app.kubernetes.io/component=controller --tail=100
```

**Common Causes**:
- Backend pods not ready (failing health checks)
- Service selector doesn't match pod labels
- Wrong target port in Service spec
- Backend timeout (increase timeout annotations)

#### Issue 2: Rate Limiting Not Working

**Diagnosis**:
```bash
# Test rate limit
for i in {1..20}; do
  curl -H "X-API-Key: test-key" https://api.ml-platform.com/predict
  echo "Request $i"
  sleep 0.1
done

# Check ingress controller config
kubectl exec -n ingress-nginx <controller-pod> -- cat /etc/nginx/nginx.conf | grep limit_req
```

**Common Causes**:
- Rate limit zone not defined in ConfigMap
- Incorrect annotation syntax
- Burst value too high

#### Issue 3: TLS Certificate Issues

**Diagnosis**:
```bash
# Check certificate secret
kubectl describe secret ml-platform-tls -n ml-serving

# Check cert-manager certificate status
kubectl get certificate -n ml-serving
kubectl describe certificate ml-platform-tls -n ml-serving

# Test TLS connection
openssl s_client -connect api.ml-platform.com:443 -servername api.ml-platform.com
```

## Best Practices

### 1. Security

- Always use TLS/HTTPS in production
- Implement API key authentication
- Use rate limiting to prevent abuse
- Add security headers (HSTS, CSP, X-Frame-Options)
- Whitelist internal endpoints (admin, metrics)
- Regularly rotate TLS certificates

### 2. Performance

- Enable keep-alive connections
- Use connection pooling
- Set appropriate timeouts for ML inference
- Configure body size limits based on input data
- Use caching where applicable
- Enable gzip compression for JSON responses

### 3. Reliability

- Run multiple ingress controller replicas
- Use anti-affinity for controller pods
- Implement health checks and readiness probes
- Set up monitoring and alerting
- Have rollback procedures ready
- Test canary deployments thoroughly before promotion

### 4. Observability

- Enable access logging
- Export metrics to Prometheus
- Set up dashboards (Grafana)
- Track key metrics: request rate, latency, error rate
- Monitor rate limit hits
- Log API key usage

### 5. ML-Specific

- Use longer timeouts for model inference
- Implement request mirroring for shadow testing
- Version your models and APIs
- Use canary deployments for gradual rollouts
- Monitor model performance metrics
- Implement circuit breakers for unhealthy models

---

## Conclusion

This implementation guide covered comprehensive Ingress and Load Balancing patterns specifically designed for ML infrastructure. Key takeaways:

1. **Choose the right controller**: NGINX for simplicity, Istio for advanced features
2. **Secure your APIs**: TLS, authentication, rate limiting
3. **Smart routing**: Multi-model, version-based, geographic
4. **Safe deployments**: Canary, A/B testing, blue-green
5. **Monitor everything**: Metrics, logs, traces

By mastering these patterns, you can build production-ready ML infrastructure that is secure, scalable, and reliable.

**Next Steps**:
- Practice implementing each pattern
- Set up monitoring and alerting
- Test failure scenarios
- Explore service mesh solutions (Istio, Linkerd)
- Move to Exercise 07: ML Workloads
