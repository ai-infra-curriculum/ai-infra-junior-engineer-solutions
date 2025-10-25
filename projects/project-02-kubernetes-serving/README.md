# Project 02: Kubernetes Model Serving - Solution

Production-grade Kubernetes deployment of the ML model API with auto-scaling, load balancing, and comprehensive monitoring.

## Quick Start

```bash
# Local deployment with Minikube
minikube start --cpus=4 --memory=8192
minikube addons enable metrics-server
minikube addons enable ingress

# Build image (use Minikube's Docker)
eval $(minikube docker-env)
docker build -t model-api:v1.0 -f ../project-01-simple-model-api/docker/Dockerfile ../project-01-simple-model-api/

# Deploy
kubectl apply -f kubernetes/

# Verify
kubectl get all -n ml-serving
kubectl port-forward -n ml-serving svc/model-api 8080:80
curl http://localhost:8080/health
```

## Features

- **Auto-scaling**: HPA scales 3-10 pods based on CPU/memory
- **Load Balancing**: Service distributes traffic across pods
- **Zero-downtime Updates**: Rolling deployment strategy
- **Health Checks**: Liveness and readiness probes
- **Resource Management**: Requests and limits configured
- **Monitoring**: Prometheus metrics collection
- **Ingress**: NGINX controller with rate limiting

## Architecture

```
Internet → Ingress → Service → Pods (HPA) → Model API
              ↓
        Prometheus (Monitoring)
```

## Project Structure

```
project-02-kubernetes-serving/
├── kubernetes/
│   ├── configmap.yaml     # Configuration
│   ├── deployment.yaml    # Pod deployment
│   ├── service.yaml       # Load balancer
│   ├── hpa.yaml          # Auto-scaler
│   └── ingress.yaml      # HTTP routing
├── helm/
│   └── model-api/        # Helm chart
├── monitoring/
│   └── servicemonitor.yaml # Prometheus config
├── tests/
│   └── test_k8s.py       # Deployment tests
├── README.md             # This file
└── SOLUTION_GUIDE.md     # Detailed guide
```

## Deployment Options

### Using kubectl (Direct)
```bash
kubectl apply -f kubernetes/
```

### Using Helm (Recommended)
```bash
helm install model-api ./helm/model-api -n ml-serving --create-namespace
helm upgrade model-api ./helm/model-api
helm rollback model-api
```

### Cloud Deployment (GKE)
```bash
gcloud container clusters create ml-cluster --num-nodes=3
gcloud container clusters get-credentials ml-cluster
kubectl apply -f kubernetes/
```

## Testing

```bash
# Unit tests
pytest tests/test_k8s.py -v

# Load testing
k6 run tests/load-test.js

# Verify auto-scaling
kubectl get hpa -n ml-serving -w
```

## Monitoring

Access Grafana dashboards:
```bash
kubectl port-forward -n monitoring svc/grafana 3000:3000
```

Default credentials: admin/admin

## Key Metrics

- Pod count: 3-10 (auto-scaled)
- CPU target: 70%
- Memory target: 80%
- Request limits: 10MB
- Response timeout: 30s

## Documentation

See [SOLUTION_GUIDE.md](SOLUTION_GUIDE.md) for:
- Detailed architecture
- Configuration explanations
- Troubleshooting guide
- Best practices
- Performance optimization

## Requirements

- Kubernetes 1.28+
- kubectl
- Helm 3+
- Metrics server
- Ingress controller

## License

Educational use only - AI Infrastructure Curriculum
