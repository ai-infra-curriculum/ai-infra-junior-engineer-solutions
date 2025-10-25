# Exercise 07: ML Workloads

Deploying and managing machine learning workloads on Kubernetes for production environments.

## Overview

This exercise demonstrates how to deploy, manage, and scale machine learning workloads on Kubernetes. You'll learn:

- Model serving (TensorFlow Serving, REST APIs, ONNX Runtime)
- Training jobs (batch, distributed, GPU-accelerated)
- Model storage and versioning
- A/B testing and canary deployments for models
- Auto-scaling for ML workloads
- Resource management (CPU, memory, GPU)
- Model monitoring and metrics
- Production ML best practices

## Prerequisites

- Completed Exercise 06 (Ingress & Load Balancing) or equivalent Kubernetes knowledge
- kubectl configured and connected to a Kubernetes cluster
- Understanding of machine learning concepts
- Familiarity with model inference and training
- **Optional:** GPU-enabled nodes for GPU training examples

## Learning Objectives

By the end of this exercise, you will be able to:

1. Deploy model serving infrastructure
2. Run training jobs on Kubernetes
3. Manage model storage and versioning
4. Implement A/B testing for models
5. Configure auto-scaling for ML workloads
6. Use GPU resources effectively
7. Monitor model performance
8. Apply production ML best practices

## Directory Structure

```
exercise-07-ml-workloads/
├── manifests/
│   ├── 01-namespace.yaml                  # Namespace with resource quotas
│   ├── 02-model-storage.yaml              # PVCs and model registry
│   ├── 03-model-serving.yaml              # TF Serving, REST APIs, ONNX
│   ├── 04-training-jobs.yaml              # Training jobs (batch, distributed, GPU)
│   └── 05-ab-testing-canary.yaml          # A/B testing and canary deployments
├── scripts/
│   ├── deploy-all.sh                      # Automated deployment
│   └── cleanup.sh                         # Cleanup script
├── models/                                 # Model storage directory
├── README.md                               # This file
└── STEP_BY_STEP.md                        # Detailed walkthrough
```

## Quick Start

### 1. Deploy Everything

```bash
# Deploy all ML workload resources
./scripts/deploy-all.sh
```

### 2. Verify Deployment

```bash
# Check namespace and resources
kubectl get all -n ml-workloads

# Check model storage
kubectl get pvc -n ml-workloads

# Check model serving
kubectl get pods -n ml-workloads -l component=model-server
```

### 3. Test Model Inference

```bash
# Test model API
kubectl run test-client --rm -it --image=curlimages/curl -n ml-workloads \
  -- curl -X POST http://model-api-server/predict \
  -H 'Content-Type: application/json' \
  -d '{"text": "This is a test input"}'
```

### 4. Run Training Job

```bash
# Check training jobs
kubectl get jobs -n ml-workloads

# View training logs
kubectl logs job/model-training-simple -n ml-workloads

# Run a new training job
kubectl create job manual-training --from=job/model-training-simple -n ml-workloads
```

### 5. Cleanup

```bash
./scripts/cleanup.sh
```

## Key Concepts

### Model Serving

Expose trained models for inference:

**Options:**
1. **TensorFlow Serving** - High-performance serving for TensorFlow models
2. **REST API** - Custom Flask/FastAPI server
3. **ONNX Runtime** - Framework-agnostic model serving
4. **GPU Serving** - GPU-accelerated inference

**Key Features:**
- Load balancing across replicas
- Health checks and readiness probes
- Batch inference support
- Auto-scaling based on load
- Model versioning

### Training Jobs

Run model training workloads:

**Job Types:**
1. **Batch Job** - One-time training job
2. **CronJob** - Scheduled periodic retraining
3. **Distributed Training** - Multi-worker parallel training
4. **GPU Training** - GPU-accelerated training
5. **Hyperparameter Tuning** - Multiple parallel experiments
6. **Pipeline** - Multi-stage training workflow

### A/B Testing & Canary Deployments

Gradually roll out new model versions:

**Strategies:**
1. **Weight-based** - Split traffic by percentage (e.g., 90% v1, 10% v2)
2. **Header-based** - Route by HTTP header
3. **Cookie-based** - Route by user cookie
4. **Gradual rollout** - Increase canary traffic over time

### Resource Management

**Resource Quotas:**
- Limit total CPU/memory/GPU usage per namespace
- Prevent resource exhaustion
- Fair sharing across teams

**Limit Ranges:**
- Default resource requests/limits
- Min/max constraints per pod

**GPU Scheduling:**
- Request GPUs with `nvidia.com/gpu: 1`
- Node selectors for GPU nodes
- Tolerations for GPU taints

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         Ingress                              │
│                    (Load Balancer)                           │
└──────────────┬──────────────────────────────┬────────────────┘
               │                              │
     ┌─────────▼──────────┐       ┌──────────▼─────────┐
     │  Model v1 (90%)    │       │  Model v2 (10%)    │
     │  3 replicas        │       │  1 replica         │
     │  (Production)      │       │  (Canary)          │
     └─────────┬──────────┘       └──────────┬─────────┘
               │                              │
               └──────────────┬───────────────┘
                              │
                    ┌─────────▼──────────┐
                    │  Model Storage      │
                    │  (PVC - 50Gi)      │
                    └────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    Training Jobs                             │
├──────────────┬──────────────┬──────────────┬────────────────┤
│  Batch       │  Scheduled   │ Distributed  │  GPU           │
│  Training    │  (CronJob)   │  (4 workers) │  Training      │
└──────────────┴──────────────┴──────────────┴────────────────┘
                              │
                    ┌─────────▼──────────┐
                    │  Training Data      │
                    │  (PVC - 100Gi)     │
                    └────────────────────┘
```

## Examples Included

### Model Serving (6 types)

1. **TensorFlow Serving**
   - gRPC and REST APIs
   - Model versioning
   - Batch inference
   - Health checks

2. **REST API Server**
   - Flask-based custom server
   - Multiple models support
   - Batch predictions
   - Model listing endpoint

3. **ONNX Runtime Server**
   - Framework-agnostic serving
   - ONNX model support
   - High performance

4. **GPU Model Server**
   - GPU-accelerated inference
   - Node selectors
   - GPU resource requests

5. **Horizontal Pod Autoscaler**
   - CPU-based scaling
   - Memory-based scaling
   - Scale 2-10 replicas

6. **Pod Disruption Budget**
   - High availability
   - Minimum available pods

### Training Jobs (7 types)

1. **Simple Training Job**
   - One-time batch training
   - Model metadata generation
   - Artifact storage

2. **Scheduled Training (CronJob)**
   - Daily automatic retraining
   - New data detection
   - Scheduled at 2 AM

3. **Distributed Training**
   - 4 parallel workers
   - Coordinated training
   - Worker synchronization

4. **GPU Training Job**
   - GPU resource allocation
   - CUDA support
   - Node selection

5. **Hyperparameter Tuning**
   - 9 parallel experiments
   - Random hyperparameter search
   - Results aggregation

6. **Training Pipeline**
   - Multi-stage workflow
   - Preprocessing → Training → Evaluation
   - Init containers for stages

7. **Storage Initialization**
   - Create directory structure
   - Setup model repository
   - Initialize metadata

### A/B Testing (5 patterns)

1. **Model v1 (Production)**
   - 3 replicas
   - Stable version
   - Prometheus metrics

2. **Model v2 (Canary)**
   - 1 replica
   - New version testing
   - Metrics comparison

3. **Weight-based Canary**
   - 75% to v1, 25% to v2
   - Gradual traffic shift
   - Ingress-based routing

4. **Header-based Routing**
   - `X-Model-Version: v2` header
   - Explicit version selection
   - Testing/debugging

5. **Load Testing**
   - 5 parallel clients
   - 100 requests per client
   - Performance comparison

## Common Commands

### Model Serving Operations

```bash
# List model serving pods
kubectl get pods -n ml-workloads -l component=model-server

# Scale model serving
kubectl scale deployment model-api-server -n ml-workloads --replicas=5

# Test inference
kubectl run test --rm -it --image=curlimages/curl -n ml-workloads \
  -- curl -X POST http://model-api-server/predict \
  -d '{"text": "sample input"}'

# View logs
kubectl logs -n ml-workloads deployment/model-api-server

# Port forward for local testing
kubectl port-forward -n ml-workloads deployment/model-api-server 8080:8080

# Check HPA status
kubectl get hpa -n ml-workloads

# View metrics
kubectl top pods -n ml-workloads -l app=model-api-server
```

### Training Job Operations

```bash
# List training jobs
kubectl get jobs -n ml-workloads -l job-type=training

# View training logs
kubectl logs job/model-training-simple -n ml-workloads

# Create manual training job
kubectl create job manual-train --from=job/model-training-simple -n ml-workloads

# Monitor distributed training
kubectl get pods -n ml-workloads -l job-type=distributed-training -w

# Check GPU job status
kubectl describe job gpu-training-job -n ml-workloads

# View CronJob schedule
kubectl get cronjobs -n ml-workloads

# Trigger CronJob manually
kubectl create job --from=cronjob/scheduled-model-training test-run -n ml-workloads

# Delete completed jobs
kubectl delete jobs -n ml-workloads --field-selector status.successful=1
```

### Storage Operations

```bash
# List PVCs
kubectl get pvc -n ml-workloads

# Check storage usage
kubectl exec -n ml-workloads deployment/model-api-server \
  -- df -h /models

# Browse model storage
kubectl exec -n ml-workloads deployment/model-api-server \
  -- ls -la /models/

# View model metadata
kubectl exec -n ml-workloads deployment/model-api-server \
  -- cat /models/sentiment-classifier/1/metadata.json

# Copy model to local
kubectl cp ml-workloads/model-api-server-xxx:/models/sentiment-classifier ./local-models/
```

### A/B Testing Operations

```bash
# View model versions
kubectl get deployments -n ml-workloads -l app=ml-model

# Check canary traffic split
kubectl get ingress -n ml-workloads -o yaml | grep canary

# Update canary weight to 50%
kubectl patch ingress model-canary -n ml-workloads \
  -p '{"metadata":{"annotations":{"nginx.ingress.kubernetes.io/canary-weight":"50"}}}'

# Test with specific version header
kubectl run test --rm -it --image=curlimages/curl -n ml-workloads \
  -- curl -H "X-Model-Version: v2" http://ml-model-service/predict

# Run model comparison
kubectl create job compare --from=job/model-comparison -n ml-workloads
kubectl logs job/compare -n ml-workloads

# Promote canary to production
kubectl scale deployment model-v2 -n ml-workloads --replicas=3
kubectl patch ingress model-canary -n ml-workloads \
  -p '{"metadata":{"annotations":{"nginx.ingress.kubernetes.io/canary-weight":"100"}}}'

# Rollback canary
kubectl patch ingress model-canary -n ml-workloads \
  -p '{"metadata":{"annotations":{"nginx.ingress.kubernetes.io/canary-weight":"0"}}}'
kubectl scale deployment model-v2 -n ml-workloads --replicas=0
```

## GPU Configuration

### Enable GPU Support

For NVIDIA GPUs:

```bash
# Install NVIDIA device plugin
kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml

# Verify GPU nodes
kubectl get nodes -o=jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.capacity.nvidia\.com/gpu}{"\n"}{end}'

# Label GPU nodes
kubectl label nodes <node-name> accelerator=nvidia-gpu

# Taint GPU nodes (optional)
kubectl taint nodes <node-name> nvidia.com/gpu=present:NoSchedule
```

### Request GPUs in Pods

```yaml
resources:
  requests:
    nvidia.com/gpu: 1  # Request 1 GPU
  limits:
    nvidia.com/gpu: 1
```

### GPU Node Selection

```yaml
nodeSelector:
  accelerator: nvidia-gpu

tolerations:
- key: nvidia.com/gpu
  operator: Exists
  effect: NoSchedule
```

## Monitoring & Metrics

### Prometheus Metrics

Models expose metrics at `/metrics`:

```bash
# Port forward to model
kubectl port-forward -n ml-workloads deployment/model-v1 8080:8080

# View metrics
curl http://localhost:8080/metrics

# Key metrics:
# - model_requests_total
# - model_request_latency_seconds
# - model_predictions_total
```

### Resource Usage

```bash
# View resource usage
kubectl top pods -n ml-workloads

# View node usage
kubectl top nodes

# Describe resource quotas
kubectl describe resourcequota -n ml-workloads

# Check limit ranges
kubectl describe limitrange -n ml-workloads
```

## Troubleshooting

### Pod Pending (Insufficient Resources)

```bash
# Check pod status
kubectl describe pod <pod-name> -n ml-workloads

# Common issues:
# - Insufficient CPU/memory
# - GPU not available
# - PVC not bound
# - Resource quota exceeded

# Check resource quota
kubectl describe resourcequota -n ml-workloads

# Check PVC status
kubectl get pvc -n ml-workloads
```

### GPU Pod Not Scheduling

```bash
# Check GPU availability
kubectl get nodes -o=jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.allocatable.nvidia\.com/gpu}{"\n"}{end}'

# Check node selectors and tolerations
kubectl get pod <pod-name> -n ml-workloads -o yaml | grep -A 5 "nodeSelector\|tolerations"

# Check GPU device plugin
kubectl get pods -n kube-system | grep nvidia
```

### Model Serving Not Responding

```bash
# Check pod logs
kubectl logs -n ml-workloads deployment/model-api-server

# Check service endpoints
kubectl get endpoints model-api-server -n ml-workloads

# Test from another pod
kubectl run test --rm -it --image=busybox -n ml-workloads \
  -- wget -O- http://model-api-server/health

# Check resource limits
kubectl describe pod -n ml-workloads -l app=model-api-server
```

### Training Job Failing

```bash
# Check job status
kubectl describe job model-training-simple -n ml-workloads

# View job logs
kubectl logs job/model-training-simple -n ml-workloads

# Check storage access
kubectl exec -n ml-workloads <training-pod> -- ls -la /models /data

# Verify resource availability
kubectl top pod -n ml-workloads <training-pod>
```

### Storage Issues

```bash
# Check PVC status
kubectl describe pvc models-pvc -n ml-workloads

# Verify storage class
kubectl get storageclass

# Check volume mounts
kubectl describe pod -n ml-workloads <pod-name> | grep -A 10 "Mounts:"

# Test write access
kubectl exec -n ml-workloads <pod-name> -- touch /models/test.txt
```

## Best Practices

### 1. Resource Management

- **Set resource requests and limits** for all containers
- **Use resource quotas** to prevent cluster overload
- **Monitor resource usage** and adjust as needed
- **Use GPU scheduling** efficiently
- **Implement auto-scaling** for variable loads

### 2. Model Versioning

- **Version all models** (e.g., v1.0.0, v2.1.0)
- **Store metadata** with each model (accuracy, date, etc.)
- **Use immutable storage** for production models
- **Implement rollback capability**
- **Track model lineage** (training data, code version)

### 3. High Availability

- **Run multiple replicas** of model serving
- **Use Pod Disruption Budgets**
- **Configure health checks** (liveness, readiness)
- **Implement graceful shutdown**
- **Use load balancing** across replicas

### 4. Security

- **Use RBAC** for access control
- **Encrypt data at rest and in transit**
- **Scan container images** for vulnerabilities
- **Use network policies** to restrict traffic
- **Rotate credentials** regularly
- **Audit model access**

### 5. Monitoring

- **Expose Prometheus metrics** from all services
- **Monitor model performance** (latency, accuracy)
- **Set up alerts** for anomalies
- **Log all predictions** for debugging
- **Track data drift** and model degradation

### 6. A/B Testing & Canary

- **Start with small canary** (5-10%)
- **Monitor closely** during rollout
- **Compare metrics** between versions
- **Have rollback plan** ready
- **Gradual increase** of canary traffic
- **Document rollout criteria**

### 7. Training Jobs

- **Use checkpointing** for long-running jobs
- **Save artifacts** to persistent storage
- **Implement retry logic** for failures
- **Use distributed training** for large models
- **Schedule training** during low-usage periods
- **Clean up completed jobs** regularly

### 8. Cost Optimization

- **Use spot/preemptible instances** for training
- **Implement auto-scaling** to match demand
- **Use CPU for inference** when possible (cheaper than GPU)
- **Batch predictions** for efficiency
- **Clean up unused resources**
- **Right-size resource requests**

## Production Checklist

- [ ] Resource requests and limits configured
- [ ] Resource quotas in place
- [ ] Health checks configured (liveness, readiness)
- [ ] Auto-scaling enabled (HPA)
- [ ] Pod Disruption Budget configured
- [ ] Multiple replicas for high availability
- [ ] Persistent storage for models
- [ ] Model versioning implemented
- [ ] Monitoring and metrics enabled
- [ ] Logging configured
- [ ] Alerts set up for errors
- [ ] Security policies applied (RBAC, Network Policies)
- [ ] A/B testing or canary deployment strategy
- [ ] Rollback procedure documented
- [ ] Training job scheduling configured
- [ ] GPU resources properly allocated
- [ ] Cost monitoring enabled
- [ ] Disaster recovery plan

## Additional Resources

### Official Documentation

- [Kubernetes Jobs](https://kubernetes.io/docs/concepts/workloads/controllers/job/)
- [Kubernetes CronJobs](https://kubernetes.io/docs/concepts/workloads/controllers/cron-jobs/)
- [Resource Management](https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/)
- [GPU Scheduling](https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/)
- [Horizontal Pod Autoscaler](https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/)

### ML on Kubernetes Tools

- [Kubeflow](https://www.kubeflow.org/) - ML toolkit for Kubernetes
- [MLflow](https://mlflow.org/) - ML lifecycle management
- [Seldon Core](https://www.seldon.io/) - ML deployment platform
- [KServe](https://kserve.github.io/website/) - Model serving platform
- [Argo Workflows](https://argoproj.github.io/argo-workflows/) - Workflow engine

### Model Serving

- [TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving)
- [TorchServe](https://pytorch.org/serve/)
- [ONNX Runtime](https://onnxruntime.ai/)
- [BentoML](https://www.bentoml.com/)
- [NVIDIA Triton](https://developer.nvidia.com/nvidia-triton-inference-server)

## Next Steps

After completing this exercise, you should:

1. Explore Kubeflow for end-to-end ML pipelines
2. Implement model monitoring and drift detection
3. Set up continuous training pipelines
4. Integrate with MLOps tools (MLflow, Weights & Biases)
5. Study service mesh for ML (Istio, Linkerd)
6. Learn about model optimization (quantization, pruning)

## License

This exercise is part of the AI Infrastructure Junior Engineer curriculum.

---

**Congratulations!** You've completed Module 006: Kubernetes Introduction. You now have hands-on experience with deploying production ML workloads on Kubernetes.
