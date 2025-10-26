# Step-by-Step Implementation Guide: ML Workloads on Kubernetes

## Overview

Deploy production ML workloads on Kubernetes! Learn to run training jobs, batch inference, model serving, GPU scheduling, distributed training, and complete ML pipelines on K8s.

**Time**: 3-4 hours | **Difficulty**: Advanced

---

## Learning Objectives

✅ Deploy ML training Jobs and CronJobs
✅ Configure GPU resources and scheduling
✅ Implement distributed training with StatefulSets
✅ Deploy model serving with autoscaling
✅ Run batch inference pipelines
✅ Manage ML experiment tracking
✅ Implement model versioning and rollback
✅ Monitor ML workloads

---

## Phase 1: GPU Support

### Enable GPU in Cluster

```bash
# Install NVIDIA device plugin
kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml

# Verify GPU nodes
kubectl get nodes -o json | jq '.items[].status.capacity'

# Check GPU availability
kubectl describe nodes | grep -A 10 "Capacity:"
```

### GPU Pod Example

```yaml
# gpu-test-pod.yaml
apiVersion: v1
kind: Pod
metadata:
  name: gpu-test
spec:
  containers:
  - name: cuda
    image: nvidia/cuda:11.8.0-base-ubuntu22.04
    command: ["nvidia-smi"]
    resources:
      limits:
        nvidia.com/gpu: 1
```

```bash
# Run and check output
kubectl apply -f gpu-test-pod.yaml
kubectl logs gpu-test
```

---

## Phase 2: Training Jobs

### Single Training Job

```yaml
# training-job.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: model-training
spec:
  backoffLimit: 3
  completions: 1
  parallelism: 1
  template:
    metadata:
      labels:
        app: training
    spec:
      restartPolicy: OnFailure
      containers:
      - name: trainer
        image: pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
        command:
        - python
        - /app/train.py
        - --epochs=100
        - --batch-size=32
        - --learning-rate=0.001
        - --checkpoint-dir=/checkpoints
        env:
        - name: MLFLOW_TRACKING_URI
          value: "http://mlflow:5000"
        - name: EXPERIMENT_NAME
          value: "resnet50-imagenet"
        resources:
          requests:
            nvidia.com/gpu: 1
            memory: "16Gi"
            cpu: "4"
          limits:
            nvidia.com/gpu: 1
            memory: "32Gi"
            cpu: "8"
        volumeMounts:
        - name: training-data
          mountPath: /data
          readOnly: true
        - name: checkpoints
          mountPath: /checkpoints
      volumes:
      - name: training-data
        persistentVolumeClaim:
          claimName: imagenet-dataset
      - name: checkpoints
        persistentVolumeClaim:
          claimName: model-checkpoints
```

### Training Script

```python
# train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import mlflow
import argparse
import os

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--checkpoint-dir', type=str, default='/checkpoints')
    args = parser.parse_args()

    # MLflow tracking
    mlflow.set_experiment(os.getenv('EXPERIMENT_NAME', 'training'))

    with mlflow.start_run():
        mlflow.log_params({
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate
        })

        # Training loop
        for epoch in range(args.epochs):
            # ... training code ...

            # Log metrics
            mlflow.log_metrics({
                'loss': loss,
                'accuracy': accuracy
            }, step=epoch)

            # Save checkpoints
            if epoch % 10 == 0:
                checkpoint_path = f"{args.checkpoint_dir}/checkpoint_epoch_{epoch}.pth"
                torch.save(model.state_dict(), checkpoint_path)
                mlflow.log_artifact(checkpoint_path)

        # Save final model
        model_path = f"{args.checkpoint_dir}/final_model.pth"
        torch.save(model.state_dict(), model_path)
        mlflow.pytorch.log_model(model, "model")

if __name__ == "__main__":
    train()
```

### Deploy and Monitor

```bash
# Create job
kubectl apply -f training-job.yaml

# Watch job progress
kubectl get jobs -w

# View logs
kubectl logs -f job/model-training

# Check GPU usage
kubectl exec <pod> -- nvidia-smi

# Get job status
kubectl describe job model-training
```

---

## Phase 3: Distributed Training

### PyTorch Distributed (StatefulSet)

```yaml
# distributed-training.yaml
apiVersion: v1
kind: Service
metadata:
  name: pytorch-dist
spec:
  clusterIP: None
  selector:
    app: pytorch-training
  ports:
  - port: 29500
    name: dist-backend
---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: pytorch-training
spec:
  serviceName: pytorch-dist
  replicas: 4
  selector:
    matchLabels:
      app: pytorch-training
  template:
    metadata:
      labels:
        app: pytorch-training
    spec:
      containers:
      - name: pytorch
        image: pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
        command:
        - python
        - -m
        - torch.distributed.launch
        - --nproc_per_node=1
        - --nnodes=4
        - --node_rank=$(NODE_RANK)
        - --master_addr=pytorch-training-0.pytorch-dist
        - --master_port=29500
        - /app/distributed_train.py
        env:
        - name: NODE_RANK
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
        - name: WORLD_SIZE
          value: "4"
        resources:
          limits:
            nvidia.com/gpu: 1
        volumeMounts:
        - name: training-data
          mountPath: /data
  volumeClaimTemplates:
  - metadata:
      name: training-data
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 100Gi
```

### Distributed Training Script

```python
# distributed_train.py
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import os

def setup():
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', 0)))

def cleanup():
    dist.destroy_process_group()

def train():
    setup()

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Create model and move to GPU
    model = YourModel().cuda()
    model = DDP(model, device_ids=[rank])

    # Training loop
    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)

        for batch in dataloader:
            # Forward pass
            outputs = model(batch)
            loss = criterion(outputs, targets)

            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    cleanup()

if __name__ == "__main__":
    train()
```

---

## Phase 4: Model Serving

### Inference Deployment with Autoscaling

```yaml
# inference-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-inference
spec:
  replicas: 3
  selector:
    matchLabels:
      app: inference
      version: v2
  template:
    metadata:
      labels:
        app: inference
        version: v2
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "9090"
    spec:
      initContainers:
      - name: download-model
        image: amazon/aws-cli
        command:
        - sh
        - -c
        - |
          aws s3 cp s3://ml-models/resnet50/v2.0.0/model.pth /models/model.pth
        volumeMounts:
        - name: model-storage
          mountPath: /models
      containers:
      - name: inference
        image: ml-inference-server:latest
        ports:
        - containerPort: 8080
          name: http
        - containerPort: 9090
          name: metrics
        env:
        - name: MODEL_PATH
          value: "/models/model.pth"
        - name: BATCH_SIZE
          value: "32"
        - name: WORKERS
          value: "4"
        resources:
          requests:
            nvidia.com/gpu: 1
            memory: "8Gi"
            cpu: "2"
          limits:
            nvidia.com/gpu: 1
            memory: "16Gi"
            cpu: "4"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 60
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 5
        volumeMounts:
        - name: model-storage
          mountPath: /models
      volumes:
      - name: model-storage
        emptyDir: {}
---
apiVersion: v1
kind: Service
metadata:
  name: inference-service
spec:
  selector:
    app: inference
  ports:
  - port: 80
    targetPort: 8080
    name: http
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: inference-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: model-inference
  minReplicas: 2
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Pods
    pods:
      metric:
        name: inference_request_duration_seconds
      target:
        type: AverageValue
        averageValue: "100m"
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100
        periodSeconds: 30
```

---

## Phase 5: Batch Inference

### Batch Processing Job

```yaml
# batch-inference.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: batch-inference
spec:
  parallelism: 5
  completions: 100
  backoffLimit: 3
  template:
    spec:
      restartPolicy: OnFailure
      containers:
      - name: inference
        image: ml-batch-inference:latest
        command:
        - python
        - /app/batch_inference.py
        - --input-bucket=s3://input-data
        - --output-bucket=s3://predictions
        - --batch-size=1000
        env:
        - name: JOB_COMPLETION_INDEX
          valueFrom:
            fieldRef:
              fieldPath: metadata.annotations['batch.kubernetes.io/job-completion-index']
        resources:
          requests:
            nvidia.com/gpu: 1
            memory: "8Gi"
          limits:
            nvidia.com/gpu: 1
            memory: "16Gi"
```

### Batch Script

```python
# batch_inference.py
import argparse
import os
import torch
from pathlib import Path

def process_batch():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-bucket', type=str)
    parser.add_argument('--output-bucket', type=str)
    parser.add_argument('--batch-size', type=int, default=1000)
    args = parser.parse_args()

    # Get job index for parallel processing
    job_index = int(os.getenv('JOB_COMPLETION_INDEX', 0))

    # Load model
    model = torch.load('/models/model.pth')
    model.eval()
    model.cuda()

    # Process assigned batch
    input_files = list_files_from_s3(args.input_bucket)
    my_files = input_files[job_index::100]  # Split work

    predictions = []
    for file in my_files:
        data = load_data(file)
        with torch.no_grad():
            pred = model(data.cuda())
        predictions.append(pred.cpu())

    # Save results
    save_to_s3(predictions, args.output_bucket, job_index)

if __name__ == "__main__":
    process_batch()
```

---

## Phase 6: Scheduled Training (CronJob)

### Daily Model Retraining

```yaml
# scheduled-training.yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: daily-model-retrain
spec:
  schedule: "0 2 * * *"  # 2 AM daily
  successfulJobsHistoryLimit: 3
  failedJobsHistoryLimit: 1
  jobTemplate:
    spec:
      template:
        spec:
          restartPolicy: OnFailure
          containers:
          - name: retrain
            image: ml-trainer:latest
            command:
            - python
            - /app/retrain.py
            - --data-start-date=$(date -d '7 days ago' +%Y-%m-%d)
            - --data-end-date=$(date +%Y-%m-%d)
            env:
            - name: MLFLOW_TRACKING_URI
              value: "http://mlflow:5000"
            resources:
              requests:
                nvidia.com/gpu: 2
                memory: "32Gi"
              limits:
                nvidia.com/gpu: 2
                memory: "64Gi"
```

---

## Phase 7: MLOps Pipeline

### Complete ML Pipeline

```yaml
# ml-pipeline.yaml
apiVersion: argoproj.io/v1alpha1
kind: Workflow
metadata:
  name: ml-training-pipeline
spec:
  entrypoint: ml-pipeline
  templates:
  - name: ml-pipeline
    steps:
    - - name: data-validation
        template: validate-data
    - - name: preprocessing
        template: preprocess
    - - name: training
        template: train-model
    - - name: evaluation
        template: evaluate
    - - name: deploy
        template: deploy-model

  - name: validate-data
    container:
      image: data-validator:latest
      command: [python, validate.py]

  - name: preprocess
    container:
      image: preprocessor:latest
      command: [python, preprocess.py]

  - name: train-model
    container:
      image: trainer:latest
      command: [python, train.py]
      resources:
        requests:
          nvidia.com/gpu: 1

  - name: evaluate
    container:
      image: evaluator:latest
      command: [python, evaluate.py]

  - name: deploy-model
    container:
      image: deployer:latest
      command: [python, deploy.py]
```

---

## Phase 8: Monitoring ML Workloads

### Prometheus Metrics

```python
# metrics.py
from prometheus_client import Counter, Histogram, Gauge

# Training metrics
training_loss = Gauge('training_loss', 'Current training loss')
training_accuracy = Gauge('training_accuracy', 'Current training accuracy')
epoch_duration = Histogram('epoch_duration_seconds', 'Epoch duration')

# Inference metrics
inference_requests = Counter('inference_requests_total', 'Total inference requests')
inference_duration = Histogram('inference_duration_seconds', 'Inference duration')
batch_size = Histogram('inference_batch_size', 'Batch size')
gpu_utilization = Gauge('gpu_utilization_percent', 'GPU utilization')
gpu_memory_used = Gauge('gpu_memory_used_mb', 'GPU memory used')
```

### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "ML Workloads",
    "panels": [
      {
        "title": "GPU Utilization",
        "targets": [{
          "expr": "avg(gpu_utilization_percent)"
        }]
      },
      {
        "title": "Inference Latency",
        "targets": [{
          "expr": "histogram_quantile(0.95, rate(inference_duration_seconds_bucket[5m]))"
        }]
      },
      {
        "title": "Training Loss",
        "targets": [{
          "expr": "training_loss"
        }]
      }
    ]
  }
}
```

---

## Best Practices

✅ Use Jobs for one-time training
✅ Use CronJobs for scheduled retraining
✅ Implement checkpointing for long-running training
✅ Use StatefulSets for distributed training
✅ Set appropriate GPU resource limits
✅ Implement health checks for inference services
✅ Use HPA for autoscaling inference
✅ Monitor GPU utilization and memory
✅ Implement model versioning
✅ Use init containers to download models
✅ Store models in object storage (S3, GCS)
✅ Log metrics to MLflow/Weights & Biases

---

## Troubleshooting

```bash
# GPU not detected
kubectl describe node <node> | grep nvidia.com/gpu

# Training job failed
kubectl logs job/model-training
kubectl describe job model-training

# OOM during training
kubectl top pod <pod>
# Reduce batch size or increase memory limit

# Slow inference
kubectl get hpa
# Check if autoscaling is working

# Check GPU usage
kubectl exec <pod> -- nvidia-smi
```

---

**ML Workloads on Kubernetes mastered!** 🤖

**Congratulations!** You've completed the Kubernetes Introduction module!
