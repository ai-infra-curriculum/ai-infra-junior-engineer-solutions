# Project 05: Production-Ready ML System (Capstone) - Solution Guide

## Overview

This capstone project integrates all previous projects (1-4) into a comprehensive, production-ready ML system with CI/CD, security, high availability, and complete observability.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Production ML System                        │
└─────────────────────────────────────────────────────────────────┘

GitHub → CI/CD Pipeline → Kubernetes Cluster
         ↓                         ↓
    [Build & Test]          [Rolling Deployment]
         ↓                         ↓
    Container Registry      Model API (HPA: 3-20)
         ↓                         ↓
    Security Scan           MLflow (Training Pipeline)
         ↓                         ↓
    Deploy Approval         Prometheus + Grafana (Monitoring)
                                   ↓
                            Alertmanager → Notifications
```

## Key Components Integration

### 1. CI/CD Pipeline (`.github/workflows/`)

**CI Pipeline (`ci.yml`):**
```yaml
name: CI Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  code-quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install black flake8 mypy pylint
          pip install -r requirements.txt

      - name: Code formatting (Black)
        run: black --check src/

      - name: Linting (Flake8)
        run: flake8 src/ --max-line-length=120

      - name: Type checking (MyPy)
        run: mypy src/ --ignore-missing-imports

      - name: Code quality (Pylint)
        run: pylint src/ --disable=C0111,R0903

  security-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Security scan (Bandit)
        run: |
          pip install bandit
          bandit -r src/ -ll

      - name: Dependency check
        run: |
          pip install safety
          safety check

  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Run tests
        run: pytest tests/ -v --cov=src --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage.xml

  integration-tests:
    runs-on: ubuntu-latest
    needs: [unit-tests]
    steps:
      - uses: actions/checkout@v3

      - name: Build Docker image
        run: docker build -t model-api:test .

      - name: Run container
        run: |
          docker run -d -p 5000:5000 --name test-api model-api:test
          sleep 10

      - name: Test endpoints
        run: |
          curl -f http://localhost:5000/health
          curl -f http://localhost:5000/info

      - name: Stop container
        run: docker stop test-api

  build-push:
    runs-on: ubuntu-latest
    needs: [code-quality, security-scan, integration-tests]
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v3

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v2

      - name: Login to Container Registry
        uses: docker/login-action@v2
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Build and push
        uses: docker/build-push-action@v4
        with:
          context: .
          push: true
          tags: |
            ghcr.io/${{ github.repository }}/model-api:latest
            ghcr.io/${{ github.repository }}/model-api:${{ github.sha }}

      - name: Scan image for vulnerabilities
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: ghcr.io/${{ github.repository }}/model-api:latest
          format: 'sarif'
          output: 'trivy-results.sarif'

      - name: Upload Trivy results
        uses: github/codeql-action/upload-sarif@v2
        with:
          sarif_file: 'trivy-results.sarif'
```

**CD Pipeline (`cd.yml`):**
```yaml
name: CD Pipeline

on:
  workflow_run:
    workflows: ["CI Pipeline"]
    branches: [main]
    types: [completed]

jobs:
  deploy-staging:
    runs-on: ubuntu-latest
    if: ${{ github.event.workflow_run.conclusion == 'success' }}
    environment:
      name: staging
    steps:
      - uses: actions/checkout@v3

      - name: Configure kubectl
        uses: azure/k8s-set-context@v3
        with:
          kubeconfig: ${{ secrets.KUBE_CONFIG }}

      - name: Deploy to staging
        run: |
          helm upgrade --install model-api ./helm/model-api \
            --namespace staging \
            --create-namespace \
            --set image.tag=${{ github.sha }} \
            --set environment=staging \
            --wait

      - name: Run smoke tests
        run: |
          kubectl wait --for=condition=ready pod -l app=model-api -n staging --timeout=300s
          kubectl port-forward -n staging svc/model-api 8080:80 &
          sleep 5
          curl -f http://localhost:8080/health

      - name: Run integration tests
        run: pytest tests/integration/ --env=staging

  deploy-production:
    runs-on: ubuntu-latest
    needs: [deploy-staging]
    environment:
      name: production
      url: https://model-api.production.example.com
    steps:
      - uses: actions/checkout@v3

      - name: Configure kubectl
        uses: azure/k8s-set-context@v3
        with:
          kubeconfig: ${{ secrets.KUBE_CONFIG_PROD }}

      - name: Canary deployment (10%)
        run: |
          helm upgrade model-api-canary ./helm/model-api \
            --namespace production \
            --set image.tag=${{ github.sha }} \
            --set replicaCount=1 \
            --set canary.enabled=true \
            --set canary.weight=10

      - name: Monitor canary metrics
        run: |
          sleep 600  # Wait 10 minutes
          # Check error rate and latency
          ./scripts/check-canary-metrics.sh

      - name: Promote to production
        run: |
          helm upgrade --install model-api ./helm/model-api \
            --namespace production \
            --set image.tag=${{ github.sha }} \
            --set environment=production \
            --set replicaCount=5 \
            --wait

      - name: Cleanup canary
        run: helm uninstall model-api-canary -n production

      - name: Send notification
        uses: slackapi/slack-github-action@v1
        with:
          webhook-url: ${{ secrets.SLACK_WEBHOOK }}
          payload: |
            {
              "text": "Production deployment successful: model-api:${{ github.sha }}"
            }
```

### 2. Security Implementation

**TLS with cert-manager (`security/cert-manager.yaml`):**
```yaml
apiVersion: cert-manager.io/v1
kind: Certificate
metadata:
  name: model-api-tls
  namespace: production
spec:
  secretName: model-api-tls-secret
  issuer: letsencrypt-prod
  commonName: model-api.production.example.com
  dnsNames:
    - model-api.production.example.com
```

**Secrets Management (`security/vault-config.yaml`):**
```yaml
apiVersion: secrets.hashicorp.com/v1beta1
kind: VaultAuth
metadata:
  name: vault-auth
  namespace: production
spec:
  method: kubernetes
  mount: kubernetes
  kubernetes:
    role: model-api
    serviceAccount: model-api-sa
---
apiVersion: secrets.hashicorp.com/v1beta1
kind: VaultStaticSecret
metadata:
  name: api-credentials
  namespace: production
spec:
  vaultAuthRef: vault-auth
  mount: secret
  type: kv-v2
  path: production/api-credentials
  destination:
    name: api-credentials
    create: true
```

**Network Policies:**
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: model-api-netpol
  namespace: production
spec:
  podSelector:
    matchLabels:
      app: model-api
  policyTypes:
    - Ingress
    - Egress
  ingress:
    - from:
        - namespaceSelector:
            matchLabels:
              name: ingress-nginx
      ports:
        - protocol: TCP
          port: 5000
  egress:
    - to:
        - namespaceSelector:
            matchLabels:
              name: kube-system
      ports:
        - protocol: TCP
          port: 53  # DNS
    - to:
        - namespaceSelector:
            matchLabels:
              name: monitoring
      ports:
        - protocol: TCP
          port: 9090  # Prometheus
```

### 3. High Availability Configuration

**Multi-zone Deployment:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-api
spec:
  replicas: 5
  template:
    spec:
      topologySpreadConstraints:
        - maxSkew: 1
          topologyKey: topology.kubernetes.io/zone
          whenUnsatisfiable: DoNotSchedule
          labelSelector:
            matchLabels:
              app: model-api
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
            - weight: 100
              podAffinityTerm:
                labelSelector:
                  matchLabels:
                    app: model-api
                topologyKey: kubernetes.io/hostname
```

**Pod Disruption Budget:**
```yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: model-api-pdb
  namespace: production
spec:
  minAvailable: 3
  selector:
    matchLabels:
      app: model-api
```

**Vertical Pod Autoscaler:**
```yaml
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: model-api-vpa
  namespace: production
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: model-api
  updatePolicy:
    updateMode: "Auto"
  resourcePolicy:
    containerPolicies:
      - containerName: model-api
        minAllowed:
          cpu: 500m
          memory: 1Gi
        maxAllowed:
          cpu: 4000m
          memory: 8Gi
```

### 4. Disaster Recovery

**Backup Strategy:**
```bash
# Backup MLflow database
kubectl exec -n ml-platform postgres-0 -- pg_dump mlflow > mlflow-backup-$(date +%Y%m%d).sql

# Backup Kubernetes resources
kubectl get all -n production -o yaml > production-backup-$(date +%Y%m%d).yaml

# Backup persistent volumes
kubectl get pvc -n production -o yaml > pvc-backup-$(date +%Y%m%d).yaml
```

**Recovery Procedures:**
```bash
# Restore MLflow database
kubectl exec -i -n ml-platform postgres-0 -- psql mlflow < mlflow-backup-20250101.sql

# Restore Kubernetes resources
kubectl apply -f production-backup-20250101.yaml

# Verify recovery
kubectl get all -n production
```

### 5. Observability Integration

**Distributed Tracing (Jaeger):**
```python
from opentelemetry import trace
from opentelemetry.exporter.jaeger import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Set up tracing
trace.set_tracer_provider(TracerProvider())
jaeger_exporter = JaegerExporter(
    agent_host_name="jaeger",
    agent_port=6831,
)
trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(jaeger_exporter)
)

# Instrument application
tracer = trace.get_tracer(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    with tracer.start_as_current_span("predict") as span:
        span.set_attribute("model.name", model_name)

        with tracer.start_as_current_span("preprocess"):
            image = preprocess(file)

        with tracer.start_as_current_span("inference"):
            predictions = model.predict(image)

        with tracer.start_as_current_span("postprocess"):
            result = format_predictions(predictions)

        return jsonify(result)
```

**Service Level Objectives (`monitoring/slos.yaml`):**
```yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: model-api-slos
  namespace: production
spec:
  groups:
    - name: slo-availability
      interval: 30s
      rules:
        - record: slo:availability:ratio_rate5m
          expr: |
            sum(rate(http_requests_total{job="model-api",status!~"5.."}[5m]))
            /
            sum(rate(http_requests_total{job="model-api"}[5m]))

        - alert: SLOAvailabilityBudgetBurn
          expr: |
            slo:availability:ratio_rate5m < 0.999
          for: 5m
          labels:
            severity: critical
          annotations:
            summary: "Availability SLO budget is burning"
            description: "Current availability: {{ $value | humanizePercentage }}"

    - name: slo-latency
      interval: 30s
      rules:
        - record: slo:latency:p95_rate5m
          expr: |
            histogram_quantile(0.95,
              sum(rate(http_request_duration_seconds_bucket{job="model-api"}[5m])) by (le)
            )

        - alert: SLOLatencyBudgetBurn
          expr: |
            slo:latency:p95_rate5m > 0.5
          for: 5m
          labels:
            severity: warning
          annotations:
            summary: "Latency SLO budget is burning"
            description: "P95 latency: {{ $value }}s"
```

## Deployment Workflow

### 1. Development
```bash
# Feature branch
git checkout -b feature/new-model

# Make changes
# ...

# Test locally
pytest tests/
docker-compose up

# Push branch
git push origin feature/new-model

# Create PR
gh pr create --title "Add new model" --body "Description"
```

### 2. Staging Deployment
```bash
# Merge PR triggers CI/CD
# Automatic deployment to staging

# Verify staging
kubectl get pods -n staging
curl https://model-api.staging.example.com/health

# Run integration tests
pytest tests/integration/ --env=staging
```

### 3. Production Deployment
```bash
# Approve production deployment in GitHub
# Canary deployment starts

# Monitor canary
kubectl logs -n production -l app=model-api,version=canary
kubectl top pods -n production

# Promote or rollback
# Automatic promotion if metrics are good
# Manual rollback if issues detected:
helm rollback model-api -n production
```

## Testing Strategy

### Unit Tests
```python
# tests/unit/test_model.py
def test_model_loading():
    loader = ModelLoader("resnet50", "cpu")
    assert loader.model is not None

def test_prediction():
    loader = ModelLoader("resnet50", "cpu")
    predictions = loader.predict("test_image.jpg")
    assert len(predictions) == 5
    assert all(0 <= p['confidence'] <= 1 for p in predictions)
```

### Integration Tests
```python
# tests/integration/test_api.py
def test_health_endpoint():
    response = requests.get(f"{API_URL}/health")
    assert response.status_code == 200
    assert response.json()['status'] == 'healthy'

def test_prediction_flow():
    with open('test_image.jpg', 'rb') as f:
        response = requests.post(
            f"{API_URL}/predict",
            files={'file': f}
        )
    assert response.status_code == 200
    assert 'predictions' in response.json()
```

### Load Tests
```javascript
// tests/load/k6-test.js
import http from 'k6/http';
import { check, sleep } from 'k6';

export const options = {
  stages: [
    { duration: '2m', target: 100 },
    { duration: '5m', target: 100 },
    { duration: '2m', target: 200 },
    { duration: '5m', target: 200 },
    { duration: '2m', target: 0 },
  ],
  thresholds: {
    http_req_duration: ['p(95)<500'],
    http_req_failed: ['rate<0.01'],
  },
};

export default function () {
  const response = http.get('https://model-api.example.com/health');
  check(response, {
    'status is 200': (r) => r.status === 200,
  });
  sleep(1);
}
```

## Conclusion

This capstone demonstrates:
- **Full CI/CD automation** from code to production
- **Production-grade security** (TLS, secrets, network policies)
- **High availability** (multi-zone, auto-scaling, PDB)
- **Complete observability** (metrics, logs, traces, SLOs)
- **Disaster recovery** (backups, rollback procedures)
- **ML pipeline integration** (training, tracking, deployment)

The implementation showcases all skills from the Junior AI Infrastructure Engineer track and represents a portfolio-quality production ML system.
