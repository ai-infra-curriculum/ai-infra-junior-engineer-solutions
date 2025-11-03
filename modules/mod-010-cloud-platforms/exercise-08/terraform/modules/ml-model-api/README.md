# Cloud-Agnostic ML Model API Terraform Module

This Terraform module deploys an ML model API to any Kubernetes cluster, regardless of the underlying cloud provider (AWS EKS, GCP GKE, Azure AKS, or on-premise).

## Key Features

- **Cloud-Agnostic:** Works on AWS, GCP, Azure, or on-premise Kubernetes
- **Production-Ready:** Includes health checks, HPA, PDB, and proper resource limits
- **Highly Configurable:** 50+ variables for customization
- **IAM Integration:** Supports AWS IRSA, GCP Workload Identity, Azure AAD Pod Identity
- **Auto-Scaling:** Built-in Horizontal Pod Autoscaler with custom metrics support
- **High Availability:** Pod anti-affinity and Pod Disruption Budget
- **Observability:** Prometheus annotations for metrics scraping

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Ingress                              │
│              (NGINX / ALB / GKE Ingress)                    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
                  ┌─────────────┐
                  │   Service   │
                  │  (ClusterIP)│
                  └──────┬──────┘
                         │
           ┌─────────────┼─────────────┐
           │             │             │
     ┌─────▼─────┐ ┌────▼─────┐ ┌────▼─────┐
     │   Pod 1   │ │   Pod 2  │ │   Pod 3  │
     │           │ │          │ │          │
     │ Model API │ │ Model API│ │ Model API│
     └───────────┘ └──────────┘ └──────────┘
           │             │             │
           └─────────────┼─────────────┘
                         │
              ┌──────────▼──────────┐
              │ Horizontal Pod      │
              │ Autoscaler (HPA)    │
              │ Min: 2 / Max: 10    │
              └─────────────────────┘
```

## Usage

### Basic Example (Minimum Configuration)

```hcl
module "model_api" {
  source = "../../modules/ml-model-api"

  service_name   = "sentiment-analysis-api"
  namespace      = "production"
  cloud_provider = "aws"
  image          = "123456789.dkr.ecr.us-east-1.amazonaws.com/sentiment-api:v1.0.0"

  replicas = 3
}
```

### Production Example (AWS with IRSA)

```hcl
module "model_api_aws" {
  source = "../../modules/ml-model-api"

  # Basic configuration
  service_name   = "model-api"
  namespace      = "production"
  environment    = "production"
  cloud_provider = "aws"
  image          = "123456789.dkr.ecr.us-east-1.amazonaws.com/model-api:v2.1.0"
  model_version  = "v2.1.0"

  # Scaling
  replicas    = 5
  enable_hpa  = true
  hpa_min_replicas = 5
  hpa_max_replicas = 20
  hpa_cpu_target_percentage = 70
  hpa_memory_target_percentage = 80

  # Resources
  cpu_request    = "1000m"
  cpu_limit      = "2000m"
  memory_request = "2Gi"
  memory_limit   = "4Gi"

  # High availability
  enable_pod_anti_affinity = true
  enable_pdb               = true
  pdb_min_available        = "60%"

  # IAM integration (AWS IRSA)
  service_account_annotations = {
    "eks.amazonaws.com/role-arn" = "arn:aws:iam::123456789:role/model-api-role"
  }

  # Ingress
  enable_ingress = true
  ingress_class_name = "alb"
  ingress_host       = "api.example.com"
  ingress_annotations = {
    "kubernetes.io/ingress.class"           = "alb"
    "alb.ingress.kubernetes.io/scheme"      = "internet-facing"
    "alb.ingress.kubernetes.io/target-type" = "ip"
    "alb.ingress.kubernetes.io/healthcheck-path" = "/health"
  }
  ingress_tls_enabled     = true
  ingress_tls_secret_name = "api-tls-cert"

  # Secrets (e.g., API keys, database passwords)
  secrets = {
    "DATABASE_URL"     = "postgresql://user:pass@db:5432/models"
    "API_KEY"          = "sk-secret-api-key-here"
    "MODEL_S3_BUCKET"  = "my-models-bucket"
  }

  # Extra environment variables
  extra_env_vars = {
    "ENABLE_METRICS" = "true"
    "LOG_FORMAT"     = "json"
    "MAX_BATCH_SIZE" = "32"
  }
}
```

### Multi-Cloud Example (Same App on AWS + GCP)

**AWS Deployment:**

```hcl
# environments/aws/main.tf

provider "kubernetes" {
  host                   = data.aws_eks_cluster.cluster.endpoint
  cluster_ca_certificate = base64decode(data.aws_eks_cluster.cluster.certificate_authority[0].data)
  token                  = data.aws_eks_cluster_auth.cluster.token
}

module "model_api_aws" {
  source = "../../modules/ml-model-api"

  service_name   = "model-api"
  namespace      = "production"
  cloud_provider = "aws"
  image          = "123456789.dkr.ecr.us-east-1.amazonaws.com/model-api:v1.0.0"

  replicas       = 5
  cpu_request    = "1000m"
  memory_request = "2Gi"

  service_account_annotations = {
    "eks.amazonaws.com/role-arn" = "arn:aws:iam::123456789:role/model-api-role"
  }

  extra_env_vars = {
    "REGION" = "us-east-1"
  }
}
```

**GCP Deployment (Identical Application!):**

```hcl
# environments/gcp/main.tf

provider "kubernetes" {
  host                   = "https://${data.google_container_cluster.cluster.endpoint}"
  token                  = data.google_client_config.default.access_token
  cluster_ca_certificate = base64decode(data.google_container_cluster.cluster.master_auth[0].cluster_ca_certificate)
}

module "model_api_gcp" {
  source = "../../modules/ml-model-api"

  service_name   = "model-api"
  namespace      = "production"
  cloud_provider = "gcp"  # Only difference: cloud provider name
  image          = "gcr.io/my-project/model-api:v1.0.0"  # GCR image

  # SAME configuration as AWS!
  replicas       = 5
  cpu_request    = "1000m"
  memory_request = "2Gi"

  service_account_annotations = {
    "iam.gke.io/gcp-service-account" = "model-api@my-project.iam.gserviceaccount.com"
  }

  extra_env_vars = {
    "REGION" = "us-central1"
  }
}
```

**Key Insight:** The application code and Kubernetes manifests are **identical**. Only the container image registry and IAM annotations differ!

## Advanced Features

### Custom Metrics Autoscaling

```hcl
module "model_api" {
  source = "../../modules/ml-model-api"

  # ... basic config ...

  enable_hpa = true
  hpa_custom_metrics = [
    {
      name         = "http_requests_per_second"
      target_value = "1000"
    },
    {
      name         = "model_inference_queue_depth"
      target_value = "50"
    }
  ]
}
```

### Persistent Volumes for Model Caching

```hcl
module "model_api" {
  source = "../../modules/ml-model-api"

  # ... basic config ...

  volumes = [
    {
      name       = "model-cache"
      type       = "persistentVolumeClaim"
      claim_name = "model-cache-pvc"
    }
  ]

  volume_mounts = [
    {
      name       = "model-cache"
      mount_path = "/app/models"
      read_only  = false
    }
  ]
}
```

### Blue-Green Deployments

```hcl
# Deploy "blue" version
module "model_api_blue" {
  source = "../../modules/ml-model-api"

  service_name   = "model-api-blue"
  image          = "registry/model-api:v1.0.0"
  model_version  = "v1.0.0"
  # ...
}

# Deploy "green" version
module "model_api_green" {
  source = "../../modules/ml-model-api"

  service_name   = "model-api-green"
  image          = "registry/model-api:v2.0.0"
  model_version  = "v2.0.0"
  # ...
}

# Switch traffic via ingress or service mesh
```

## Cloud Provider Differences

### AWS (EKS)

**IAM Integration:** IRSA (IAM Roles for Service Accounts)

```hcl
service_account_annotations = {
  "eks.amazonaws.com/role-arn" = "arn:aws:iam::ACCOUNT_ID:role/ROLE_NAME"
}
```

**Ingress:** AWS ALB Ingress Controller

```hcl
ingress_class_name = "alb"
ingress_annotations = {
  "kubernetes.io/ingress.class"           = "alb"
  "alb.ingress.kubernetes.io/scheme"      = "internet-facing"
  "alb.ingress.kubernetes.io/target-type" = "ip"
}
```

### GCP (GKE)

**IAM Integration:** Workload Identity

```hcl
service_account_annotations = {
  "iam.gke.io/gcp-service-account" = "SERVICE_ACCOUNT@PROJECT.iam.gserviceaccount.com"
}
```

**Ingress:** GKE Ingress (GCE Load Balancer)

```hcl
ingress_class_name = "gce"
ingress_annotations = {
  "kubernetes.io/ingress.class" = "gce"
  "kubernetes.io/ingress.global-static-ip-name" = "api-ip"
}
```

### Azure (AKS)

**IAM Integration:** AAD Pod Identity

```hcl
service_account_annotations = {
  "aadpodidbinding" = "model-api-identity"
}
```

**Ingress:** Azure Application Gateway

```hcl
ingress_class_name = "azure/application-gateway"
ingress_annotations = {
  "kubernetes.io/ingress.class" = "azure/application-gateway"
  "appgw.ingress.kubernetes.io/backend-path-prefix" = "/"
}
```

## Variables

See [variables.tf](./variables.tf) for complete list. Key variables:

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `service_name` | string | *required* | Name of the service |
| `namespace` | string | `"default"` | Kubernetes namespace |
| `cloud_provider` | string | *required* | Cloud provider (aws/gcp/azure/on-prem) |
| `image` | string | *required* | Container image |
| `replicas` | number | `3` | Number of pod replicas |
| `cpu_request` | string | `"500m"` | CPU request |
| `memory_request` | string | `"512Mi"` | Memory request |
| `enable_hpa` | bool | `false` | Enable auto-scaling |
| `enable_ingress` | bool | `false` | Enable ingress |

## Outputs

| Output | Description |
|--------|-------------|
| `service_endpoint` | Internal service endpoint |
| `service_cluster_ip` | ClusterIP of the service |
| `ingress_hostname` | Ingress hostname (if enabled) |
| `model_version` | ML model version |

## Requirements

- Terraform >= 1.0
- Kubernetes cluster (EKS, GKE, AKS, or on-prem)
- `kubectl` configured with cluster access
- Container registry with model API image

## Testing

Deploy to a test namespace:

```bash
# Create test namespace
kubectl create namespace test

# Deploy module
terraform init
terraform plan -var="namespace=test"
terraform apply -var="namespace=test"

# Test deployment
kubectl get pods -n test
kubectl logs -f deployment/model-api -n test

# Test API
kubectl port-forward service/model-api 8000:80 -n test
curl http://localhost:8000/health

# Cleanup
terraform destroy -var="namespace=test"
kubectl delete namespace test
```

## Best Practices

1. **Use specific image tags** - Never use `:latest` in production
2. **Set resource limits** - Prevent resource exhaustion
3. **Enable HPA** - Auto-scale based on traffic
4. **Enable PDB** - Maintain availability during node maintenance
5. **Use health checks** - Ensure fast failure detection
6. **Monitor metrics** - Use Prometheus annotations
7. **Use secrets properly** - Never commit secrets to git
8. **Test deployments** - Use staging environment first

## Troubleshooting

### Pods not starting

```bash
kubectl describe pod <pod-name> -n production
kubectl logs <pod-name> -n production
```

Common issues:
- Image pull errors (check registry credentials)
- Resource limits too low (OOMKilled)
- Liveness probe failing too fast

### HPA not scaling

```bash
kubectl get hpa -n production
kubectl describe hpa model-api -n production
```

Common issues:
- Metrics server not installed
- Resource requests not set
- Metrics not available

### Ingress not working

```bash
kubectl get ingress -n production
kubectl describe ingress model-api -n production
```

Common issues:
- Ingress controller not installed
- TLS certificate missing
- Annotations incorrect for cloud provider

## Examples

See [examples/](../../examples/) directory for complete working examples:

- `examples/aws/` - AWS EKS deployment
- `examples/gcp/` - GCP GKE deployment
- `examples/azure/` - Azure AKS deployment
- `examples/multi-cloud/` - Active-Passive DR across AWS + GCP

## License

MIT
