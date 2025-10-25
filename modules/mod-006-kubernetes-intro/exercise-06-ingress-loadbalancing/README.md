# Exercise 06: Ingress & Load Balancing

Managing external access to services and implementing advanced traffic routing in Kubernetes.

## Overview

This exercise demonstrates how to expose applications to external traffic, implement load balancing strategies, and manage network security in Kubernetes. You'll learn:

- Ingress controllers and resources
- Path-based and host-based routing
- TLS/SSL termination
- Load balancing strategies
- Canary and blue-green deployments
- Rate limiting and authentication
- Network policies for security
- Service types (ClusterIP, NodePort, LoadBalancer)

## Prerequisites

- Completed Exercise 05 (ConfigMaps & Secrets) or equivalent Kubernetes knowledge
- kubectl configured and connected to a Kubernetes cluster
- **Ingress controller installed** (NGINX, Traefik, HAProxy, etc.)
- Basic understanding of HTTP/HTTPS
- Familiarity with DNS and load balancing concepts

## Installing NGINX Ingress Controller

If you don't have an Ingress controller, install NGINX Ingress Controller:

```bash
# For cloud providers (AWS, GCP, Azure)
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.9.4/deploy/static/provider/cloud/deploy.yaml

# For minikube
minikube addons enable ingress

# For kind
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/main/deploy/static/provider/kind/deploy.yaml

# For bare metal
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.9.4/deploy/static/provider/baremetal/deploy.yaml
```

Verify installation:
```bash
kubectl get pods -n ingress-nginx
kubectl get ingressclass
```

## Learning Objectives

By the end of this exercise, you will be able to:

1. Configure and use Ingress resources
2. Implement path-based and host-based routing
3. Set up TLS/SSL termination
4. Deploy canary and blue-green deployments
5. Configure rate limiting and authentication
6. Implement Network Policies for security
7. Use different Service types appropriately
8. Troubleshoot Ingress and networking issues

## Directory Structure

```
exercise-06-ingress-loadbalancing/
├── manifests/
│   ├── 01-namespace.yaml                  # Namespace definition
│   ├── 02-backend-apps.yaml               # Backend applications (v1, v2, API, web, admin, health)
│   ├── 03-services.yaml                   # Service examples (all types)
│   ├── 04-ingress-basic.yaml              # Basic Ingress patterns
│   ├── 05-ingress-tls.yaml                # TLS/SSL Ingress
│   ├── 06-ingress-advanced.yaml           # Advanced patterns (canary, auth, rate limiting)
│   └── 07-network-policies.yaml           # Network security policies
├── scripts/
│   ├── deploy-all.sh                      # Automated deployment
│   └── cleanup.sh                         # Cleanup script
├── README.md                               # This file
└── STEP_BY_STEP.md                        # Detailed walkthrough
```

## Quick Start

### 1. Deploy Everything

```bash
# Deploy all resources
./scripts/deploy-all.sh
```

### 2. Verify Deployment

```bash
# Check namespace
kubectl get all -n ingress-demo

# Check Ingress resources
kubectl get ingress -n ingress-demo

# Get Ingress controller IP
kubectl get svc -n ingress-nginx
```

### 3. Configure Local DNS (for testing)

```bash
# Get Ingress controller IP
INGRESS_IP=$(kubectl get svc -n ingress-nginx ingress-nginx-controller \
  -o jsonpath='{.status.loadBalancer.ingress[0].ip}')

# Add to /etc/hosts
echo "$INGRESS_IP api.example.com" | sudo tee -a /etc/hosts
echo "$INGRESS_IP web.example.com" | sudo tee -a /etc/hosts
echo "$INGRESS_IP admin.example.com" | sudo tee -a /etc/hosts
echo "$INGRESS_IP canary.example.com" | sudo tee -a /etc/hosts
```

### 4. Test Endpoints

```bash
# Test basic Ingress
curl http://$INGRESS_IP/

# Test path-based routing
curl http://$INGRESS_IP/api
curl http://$INGRESS_IP/web
curl http://$INGRESS_IP/admin

# Test host-based routing
curl http://api.example.com/
curl http://web.example.com/
```

### 5. Cleanup

```bash
./scripts/cleanup.sh
```

## Key Concepts

### Ingress vs Service

| Feature | Service | Ingress |
|---------|---------|---------|
| **Layer** | L4 (TCP/UDP) | L7 (HTTP/HTTPS) |
| **Load Balancing** | Simple (round-robin) | Advanced (path, host, headers) |
| **SSL/TLS** | Pass-through | Termination |
| **Cost** | One LB per service | One LB for multiple services |
| **Routing** | Port-based | Path/host-based |
| **URL Rewriting** | No | Yes |
| **Authentication** | No | Yes (with annotations) |

### Service Types

#### 1. ClusterIP (Default)
- Internal cluster IP only
- Not accessible from outside
- Used by Ingress to route traffic

```yaml
spec:
  type: ClusterIP
  selector:
    app: backend
  ports:
  - port: 80
    targetPort: 8080
```

#### 2. NodePort
- Exposes service on each node's IP at a static port (30000-32767)
- Accessible via `<NodeIP>:<NodePort>`

```yaml
spec:
  type: NodePort
  selector:
    app: backend
  ports:
  - port: 80
    targetPort: 8080
    nodePort: 30080
```

#### 3. LoadBalancer
- Creates external load balancer (cloud provider specific)
- Gets external IP automatically
- Costs more (one LB per service)

```yaml
spec:
  type: LoadBalancer
  selector:
    app: backend
  ports:
  - port: 80
    targetPort: 8080
```

#### 4. ExternalName
- Maps service to external DNS name
- No proxying, just DNS CNAME

```yaml
spec:
  type: ExternalName
  externalName: api.external-service.com
```

### Ingress Patterns

#### Path-Based Routing

Route traffic based on URL path:

```yaml
rules:
- http:
    paths:
    - path: /api
      backend:
        service:
          name: api-service
    - path: /web
      backend:
        service:
          name: web-service
```

Request to `/api/users` → `api-service`
Request to `/web/home` → `web-service`

#### Host-Based Routing

Route traffic based on hostname:

```yaml
rules:
- host: api.example.com
  http:
    paths:
    - path: /
      backend:
        service:
          name: api-service
- host: web.example.com
  http:
    paths:
    - path: /
      backend:
        service:
          name: web-service
```

Request to `api.example.com` → `api-service`
Request to `web.example.com` → `web-service`

#### Combined Routing

Combine both host and path:

```yaml
rules:
- host: app.example.com
  http:
    paths:
    - path: /api
      backend:
        service:
          name: api-service
    - path: /
      backend:
        service:
          name: web-service
```

### TLS/SSL Termination

Ingress can terminate TLS, so backend services don't need to:

```yaml
spec:
  tls:
  - hosts:
    - secure.example.com
    secretName: tls-secret
  rules:
  - host: secure.example.com
    http:
      paths:
      - path: /
        backend:
          service:
            name: backend
```

### Canary Deployments

Gradually roll out new version by routing percentage of traffic:

```yaml
# Production ingress (90%)
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: production
spec:
  rules:
  - host: app.example.com
    http:
      paths:
      - backend:
          service:
            name: backend-v1
---
# Canary ingress (10%)
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: canary
  annotations:
    nginx.ingress.kubernetes.io/canary: "true"
    nginx.ingress.kubernetes.io/canary-weight: "10"
spec:
  rules:
  - host: app.example.com
    http:
      paths:
      - backend:
          service:
            name: backend-v2
```

### Network Policies

Control network traffic between pods:

```yaml
# Default deny all ingress
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: deny-all
spec:
  podSelector: {}
  policyTypes:
  - Ingress
---
# Allow from ingress controller
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-from-ingress
spec:
  podSelector:
    matchLabels:
      app: backend
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8080
```

## Examples Included

### Service Examples (13 types)

1. **ClusterIP** - Default internal service
2. **NodePort** - External access via node ports
3. **LoadBalancer** - Cloud provider load balancer
4. **Headless** - Direct pod access (no load balancing)
5. **Session Affinity** - Sticky sessions
6. **ExternalName** - DNS CNAME to external service
7. **Multi-port** - Multiple ports on one service
8. **Specific selectors** - Version-specific services

### Ingress Examples (30+ patterns)

#### Basic Ingress
1. Simple routing
2. Path-based routing
3. Host-based routing
4. Combined (host + path) routing
5. Default backend
6. Fanout (multiple paths)
7. Custom error pages

#### TLS Ingress
8. Basic TLS/HTTPS
9. Multiple TLS certificates
10. Secure cipher suites
11. TLS passthrough
12. Mutual TLS (mTLS)
13. Auto-certificates (cert-manager)
14. Wildcard certificates

#### Advanced Ingress
15. Canary deployment (weight-based)
16. Canary deployment (header-based)
17. Canary deployment (cookie-based)
18. Rate limiting (requests per second/minute)
19. Basic authentication
20. OAuth2 authentication
21. IP whitelist
22. CORS configuration
23. Custom timeouts
24. Custom response headers
25. HTTP redirects
26. URL rewriting
27. Sticky sessions
28. Load balancing algorithms
29. WebSocket support
30. gRPC support
31. Circuit breaker
32. Blue-green deployment

### Network Policy Examples (15 types)

1. Deny all ingress (default deny)
2. Allow from Ingress controller
3. Pod-to-pod communication
4. Namespace-based access
5. DNS access
6. External egress control
7. Database access restrictions
8. Deny egress
9. Multiple rules
10. Admin IP restrictions
11. Health check access
12. Version-specific policies
13. Multi-port policies

## Common Annotations

### NGINX Ingress Controller

```yaml
annotations:
  # SSL
  nginx.ingress.kubernetes.io/ssl-redirect: "true"
  nginx.ingress.kubernetes.io/ssl-protocols: "TLSv1.2 TLSv1.3"

  # Rewriting
  nginx.ingress.kubernetes.io/rewrite-target: /$2

  # Rate limiting
  nginx.ingress.kubernetes.io/limit-rps: "10"
  nginx.ingress.kubernetes.io/limit-rpm: "100"

  # Authentication
  nginx.ingress.kubernetes.io/auth-type: basic
  nginx.ingress.kubernetes.io/auth-secret: auth-secret

  # CORS
  nginx.ingress.kubernetes.io/enable-cors: "true"
  nginx.ingress.kubernetes.io/cors-allow-origin: "*"

  # Timeouts
  nginx.ingress.kubernetes.io/proxy-read-timeout: "120"
  nginx.ingress.kubernetes.io/proxy-body-size: "50m"

  # Canary
  nginx.ingress.kubernetes.io/canary: "true"
  nginx.ingress.kubernetes.io/canary-weight: "10"

  # Session affinity
  nginx.ingress.kubernetes.io/affinity: "cookie"
  nginx.ingress.kubernetes.io/session-cookie-name: "route"

  # IP whitelist
  nginx.ingress.kubernetes.io/whitelist-source-range: "10.0.0.0/8"
```

## Common Commands

### Ingress Operations

```bash
# List Ingress resources
kubectl get ingress -n ingress-demo

# Describe Ingress
kubectl describe ingress basic-ingress -n ingress-demo

# Get Ingress YAML
kubectl get ingress basic-ingress -n ingress-demo -o yaml

# Edit Ingress
kubectl edit ingress basic-ingress -n ingress-demo

# Delete Ingress
kubectl delete ingress basic-ingress -n ingress-demo
```

### Service Operations

```bash
# List services
kubectl get svc -n ingress-demo

# Get service endpoints
kubectl get endpoints backend-v1 -n ingress-demo

# Describe service
kubectl describe svc backend-v1 -n ingress-demo

# Test service from within cluster
kubectl run test --rm -it --image=busybox -n ingress-demo \
  -- wget -O- http://backend-v1
```

### Network Policy Operations

```bash
# List network policies
kubectl get networkpolicies -n ingress-demo

# Describe network policy
kubectl describe networkpolicy allow-from-ingress-controller -n ingress-demo

# Test network connectivity
kubectl run test --rm -it --image=busybox -n ingress-demo \
  -- wget -O- http://backend-v1 --timeout=5
```

### Debugging

```bash
# Check Ingress controller logs
kubectl logs -n ingress-nginx -l app.kubernetes.io/name=ingress-nginx

# Check Ingress controller service
kubectl get svc -n ingress-nginx

# Describe Ingress for events
kubectl describe ingress <name> -n ingress-demo

# Test from pod
kubectl run test --rm -it --image=curlimages/curl -n ingress-demo \
  -- curl -v http://backend-v1

# Check DNS resolution
kubectl run test --rm -it --image=busybox -n ingress-demo \
  -- nslookup backend-v1
```

## Testing Scenarios

### Test Path-Based Routing

```bash
INGRESS_IP=<your-ingress-ip>

curl http://$INGRESS_IP/api
curl http://$INGRESS_IP/web
curl http://$INGRESS_IP/admin
curl http://$INGRESS_IP/health
```

### Test Host-Based Routing

```bash
curl -H "Host: api.example.com" http://$INGRESS_IP/
curl -H "Host: web.example.com" http://$INGRESS_IP/
curl -H "Host: admin.example.com" http://$INGRESS_IP/
```

### Test Canary Deployment

```bash
# Should see mix of v1 and v2 responses (90/10 split)
for i in {1..20}; do
  curl http://canary.example.com/
  echo ""
done
```

### Test Header-Based Canary

```bash
# Regular traffic goes to v1
curl http://canary-header.example.com/

# With canary header goes to v2
curl -H "X-Canary: always" http://canary-header.example.com/
```

### Test TLS

```bash
# Should redirect to HTTPS
curl -v http://secure.example.com/

# HTTPS request
curl -k https://secure.example.com/
```

### Test Rate Limiting

```bash
# Rapid requests should get rate limited
for i in {1..20}; do
  curl http://ratelimit.example.com/
  echo ""
done
```

### Test Basic Auth

```bash
# Without credentials - 401
curl http://auth.example.com/

# With credentials
curl -u admin:admin123 http://auth.example.com/
```

## Troubleshooting

### Ingress Not Working

```bash
# Check if Ingress controller is running
kubectl get pods -n ingress-nginx

# Check Ingress resource
kubectl describe ingress <name> -n ingress-demo

# Check service exists
kubectl get svc backend-v1 -n ingress-demo

# Check service endpoints
kubectl get endpoints backend-v1 -n ingress-demo

# Check Ingress controller logs
kubectl logs -n ingress-nginx -l app.kubernetes.io/name=ingress-nginx --tail=100
```

### 404 Not Found

- Check path in Ingress matches request path
- Verify `rewrite-target` annotation if using path rewrites
- Check service name and port in Ingress spec
- Verify service selector matches pod labels

### 502 Bad Gateway

- Backend pods are not running
- Backend pods are not ready (readiness probe failing)
- Service selector doesn't match any pods
- Wrong target port in Service spec

### 503 Service Unavailable

- No backend pods available
- All backend pods are failing health checks
- Network policy blocking traffic

### TLS Issues

- Certificate secret doesn't exist
- Certificate hostname doesn't match request hostname
- Certificate expired
- Wrong secret type (should be `kubernetes.io/tls`)

### Network Policy Blocking Traffic

```bash
# Temporarily disable network policies
kubectl delete networkpolicies --all -n ingress-demo

# Test if issue is resolved
curl http://$INGRESS_IP/

# Re-apply with correct rules
kubectl apply -f manifests/07-network-policies.yaml
```

## Best Practices

1. **Use Ingress Instead of LoadBalancer Services**
   - One Ingress can handle multiple services
   - Reduces cloud costs
   - Centralized TLS termination

2. **Implement Network Policies**
   - Start with default deny
   - Explicitly allow required traffic
   - Test thoroughly in non-production first

3. **Use TLS/HTTPS**
   - Use cert-manager for automatic certificate management
   - Enforce HTTPS with `ssl-redirect: "true"`
   - Use strong cipher suites

4. **Configure Resource Limits**
   - Set request/response timeouts
   - Configure body size limits
   - Implement rate limiting for public APIs

5. **Monitoring and Logging**
   - Monitor Ingress controller metrics
   - Log all requests
   - Set up alerts for error rates

6. **Security**
   - Use authentication (basic auth, OAuth2)
   - Implement IP whitelisting for admin panels
   - Add security headers (HSTS, X-Frame-Options, etc.)

7. **High Availability**
   - Run multiple Ingress controller replicas
   - Use anti-affinity to spread across nodes
   - Configure proper health checks

8. **Testing**
   - Use canary deployments for gradual rollouts
   - Test in staging environment first
   - Have rollback plan ready

## Additional Resources

### Official Documentation

- [Kubernetes Ingress](https://kubernetes.io/docs/concepts/services-networking/ingress/)
- [Kubernetes Services](https://kubernetes.io/docs/concepts/services-networking/service/)
- [Network Policies](https://kubernetes.io/docs/concepts/services-networking/network-policies/)
- [NGINX Ingress Controller](https://kubernetes.github.io/ingress-nginx/)

### Tools

- [cert-manager](https://cert-manager.io/) - Automatic TLS certificate management
- [ExternalDNS](https://github.com/kubernetes-sigs/external-dns) - Automatic DNS configuration
- [Calico](https://www.tigera.io/project-calico/) - Network policy enforcement
- [Cilium](https://cilium.io/) - eBPF-based networking and security

## Next Steps

After completing this exercise, you should:

1. Review the step-by-step guide (`STEP_BY_STEP.md`)
2. Practice creating custom Ingress patterns
3. Implement Network Policies for your applications
4. Explore service mesh solutions (Istio, Linkerd)
5. Move on to Exercise 07: ML Workloads

## License

This exercise is part of the AI Infrastructure Junior Engineer curriculum.

---

**Remember:** Ingress and load balancing are critical for exposing applications securely and efficiently. Master these patterns to build production-ready Kubernetes infrastructure.
