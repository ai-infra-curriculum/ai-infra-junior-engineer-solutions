# Implementation Guide: ML-Specific Workloads on Kubernetes

## Overview

This guide provides comprehensive, production-ready implementations for deploying machine learning workloads on Kubernetes. You'll learn how to run training jobs, implement distributed training with PyTorch, manage GPU resources, serve models with KServe/Seldon, and orchestrate complete ML workflows.

**Target Audience**: ML Engineers, DevOps Engineers, Platform Engineers
**Prerequisites**: Kubernetes fundamentals, basic ML concepts, Docker experience
**Estimated Time**: 6-8 hours for full implementation

---

## Table of Contents

1. [Kubernetes Jobs for ML Training](#1-kubernetes-jobs-for-ml-training)
2. [CronJobs for Scheduled Training](#2-cronjobs-for-scheduled-training)
3. [GPU Node Selection and Allocation](#3-gpu-node-selection-and-allocation)
4. [Training Job Patterns](#4-training-job-patterns)
5. [Model Serving with KServe and Seldon](#5-model-serving-with-kserve-and-seldon)
6. [Resource Quotas and Priorities](#6-resource-quotas-and-priorities)
7. [Production ML Workflow Orchestration](#7-production-ml-workflow-orchestration)

---

## 1. Kubernetes Jobs for ML Training

### 1.1 Understanding Kubernetes Jobs for ML

Kubernetes Jobs are perfect for ML training workloads because they:
- Run to completion (train until done)
- Support automatic retries on failure
- Provide resource isolation
- Enable parallel execution
- Automatically clean up after completion

**Key Differences from Deployments**:

| Feature | Deployment | Job |
|---------|-----------|-----|
| **Purpose** | Long-running services | Run-to-completion tasks |
| **Restart** | Always restart on failure | Limited retries (backoffLimit) |
| **Completion** | Never completes | Terminates when done |
| **Use Case** | Model serving | Model training |

### 1.2 Basic Training Job

**`training-job-basic.yaml`**:

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: pytorch-training-basic
  namespace: ml-workloads
  labels:
    app: model-training
    framework: pytorch
    model: resnet50
spec:
  # Retry up to 3 times on failure
  backoffLimit: 3

  # Automatically delete job after 1 hour of completion
  ttlSecondsAfterFinished: 3600

  # Keep completed pods for debugging
  completions: 1
  parallelism: 1

  template:
    metadata:
      labels:
        app: model-training
        job-name: pytorch-training-basic
    spec:
      restartPolicy: OnFailure

      containers:
      - name: pytorch-trainer
        image: pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

        command: ["python"]
        args:
          - "-c"
          - |
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from datetime import datetime
            import json
            import os

            print(f"PyTorch version: {torch.__version__}")
            print(f"CUDA available: {torch.cuda.is_available()}")
            print(f"Training started at: {datetime.now()}")

            # Simple model for demonstration
            class SimpleNet(nn.Module):
                def __init__(self):
                    super(SimpleNet, self).__init__()
                    self.fc1 = nn.Linear(784, 128)
                    self.fc2 = nn.Linear(128, 10)

                def forward(self, x):
                    x = torch.relu(self.fc1(x))
                    return self.fc2(x)

            # Training configuration
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = SimpleNet().to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            # Simulated training loop
            num_epochs = int(os.getenv('EPOCHS', 10))
            for epoch in range(num_epochs):
                # Simulate training
                loss = 1.0 / (epoch + 1)
                accuracy = 0.5 + (epoch * 0.04)
                print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

            # Save model checkpoint
            checkpoint_path = "/models/resnet50/checkpoints/checkpoint.pth"
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

            torch.save({
                'epoch': num_epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, checkpoint_path)

            # Save metadata
            metadata = {
                "model_name": "resnet50",
                "version": "1.0.0",
                "framework": "pytorch",
                "framework_version": torch.__version__,
                "trained_at": datetime.now().isoformat(),
                "epochs": num_epochs,
                "final_loss": float(loss),
                "final_accuracy": float(accuracy),
                "device": str(device)
            }

            metadata_path = "/models/resnet50/metadata.json"
            os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            print(f"Training completed at: {datetime.now()}")
            print(f"Model saved to: {checkpoint_path}")
            print(f"Metadata saved to: {metadata_path}")

        env:
        - name: EPOCHS
          value: "10"
        - name: BATCH_SIZE
          value: "32"
        - name: LEARNING_RATE
          value: "0.001"

        # Resource requests and limits
        resources:
          requests:
            memory: "4Gi"
            cpu: "2000m"
          limits:
            memory: "8Gi"
            cpu: "4000m"

        # Mount model storage
        volumeMounts:
        - name: model-storage
          mountPath: /models
        - name: training-data
          mountPath: /data
          readOnly: true

      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: models-pvc
      - name: training-data
        persistentVolumeClaim:
          claimName: training-data-pvc
```

### 1.3 Advanced Training Job with Checkpointing

**`training-job-checkpoint.yaml`**:

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: pytorch-training-checkpoint
  namespace: ml-workloads
spec:
  backoffLimit: 5
  template:
    spec:
      restartPolicy: OnFailure

      initContainers:
      # Check for existing checkpoint to resume training
      - name: checkpoint-checker
        image: busybox
        command: ["sh", "-c"]
        args:
          - |
            if [ -f /models/resnet50/checkpoints/latest.pth ]; then
              echo "Found existing checkpoint - will resume training"
              echo "resume" > /tmp/training-mode
            else
              echo "No checkpoint found - starting fresh"
              echo "fresh" > /tmp/training-mode
            fi
        volumeMounts:
        - name: model-storage
          mountPath: /models
        - name: shared-data
          mountPath: /tmp

      containers:
      - name: pytorch-trainer
        image: pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
        command: ["python"]
        args:
          - "-c"
          - |
            import torch
            import torch.nn as nn
            import torch.optim as optim
            import os
            from datetime import datetime

            # Check training mode
            with open('/tmp/training-mode', 'r') as f:
                mode = f.read().strip()

            class SimpleNet(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc1 = nn.Linear(784, 128)
                    self.fc2 = nn.Linear(128, 10)
                def forward(self, x):
                    x = torch.relu(self.fc1(x))
                    return self.fc2(x)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = SimpleNet().to(device)
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            start_epoch = 0
            checkpoint_path = "/models/resnet50/checkpoints/latest.pth"

            # Resume from checkpoint if exists
            if mode == "resume" and os.path.exists(checkpoint_path):
                print(f"Resuming from checkpoint: {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch']
                print(f"Resumed from epoch {start_epoch}")
            else:
                print("Starting fresh training")

            # Training loop with periodic checkpointing
            total_epochs = int(os.getenv('EPOCHS', 20))
            checkpoint_interval = int(os.getenv('CHECKPOINT_INTERVAL', 5))

            for epoch in range(start_epoch, total_epochs):
                # Simulate training
                loss = 1.0 / (epoch + 1)
                accuracy = 0.5 + (epoch * 0.02)
                print(f"Epoch [{epoch+1}/{total_epochs}] - Loss: {loss:.4f}, Acc: {accuracy:.4f}")

                # Save checkpoint periodically
                if (epoch + 1) % checkpoint_interval == 0:
                    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,
                        'accuracy': accuracy
                    }, checkpoint_path)
                    print(f"Checkpoint saved at epoch {epoch+1}")

            # Save final model
            final_model_path = "/models/resnet50/model.pth"
            torch.save(model.state_dict(), final_model_path)
            print(f"Final model saved to: {final_model_path}")

        env:
        - name: EPOCHS
          value: "20"
        - name: CHECKPOINT_INTERVAL
          value: "5"

        resources:
          requests:
            memory: "4Gi"
            cpu: "2000m"
          limits:
            memory: "8Gi"
            cpu: "4000m"

        volumeMounts:
        - name: model-storage
          mountPath: /models
        - name: training-data
          mountPath: /data
        - name: shared-data
          mountPath: /tmp

      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: models-pvc
      - name: training-data
        persistentVolumeClaim:
          claimName: training-data-pvc
      - name: shared-data
        emptyDir: {}
```

**Usage**:

```bash
# Deploy training job
kubectl apply -f training-job-basic.yaml

# Monitor job progress
kubectl get jobs -n ml-workloads -w

# View training logs
kubectl logs -n ml-workloads job/pytorch-training-basic -f

# Check job status
kubectl describe job pytorch-training-basic -n ml-workloads

# Get trained model
kubectl exec -n ml-workloads -it deployment/model-server -- ls -lh /models/resnet50/
```

---

## 2. CronJobs for Scheduled Training

### 2.1 Understanding CronJobs for ML

CronJobs enable automated, scheduled model retraining to:
- Retrain models with fresh data daily/weekly
- Perform batch inference at scheduled times
- Run data validation pipelines
- Generate model performance reports
- Trigger automated model updates

### 2.2 Daily Model Retraining

**`cronjob-daily-training.yaml`**:

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: daily-model-retraining
  namespace: ml-workloads
  labels:
    app: scheduled-training
    frequency: daily
spec:
  # Run every day at 2:00 AM UTC
  schedule: "0 2 * * *"

  # Keep history of last 3 successful jobs
  successfulJobsHistoryLimit: 3

  # Keep history of last 1 failed job
  failedJobsHistoryLimit: 1

  # Don't start new job if previous one still running
  concurrencyPolicy: Forbid

  # Suspend scheduling (useful for maintenance)
  suspend: false

  jobTemplate:
    metadata:
      labels:
        app: scheduled-training
    spec:
      backoffLimit: 2
      ttlSecondsAfterFinished: 86400  # Clean up after 24 hours

      template:
        spec:
          restartPolicy: OnFailure

          containers:
          - name: retrainer
            image: pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

            command: ["bash"]
            args:
              - "-c"
              - |
                set -e

                echo "===== Scheduled Training Job ====="
                echo "Job Name: ${JOB_NAME}"
                echo "Start Time: $(date -Iseconds)"
                echo ""

                # Install dependencies
                pip install --quiet pandas scikit-learn boto3

                # Check for new training data
                echo "Checking for new training data..."
                NEW_FILES=$(find /data -type f -mtime -1 | wc -l)
                echo "Found ${NEW_FILES} new data files in last 24 hours"

                if [ "${NEW_FILES}" -lt "${MIN_NEW_FILES}" ]; then
                  echo "Insufficient new data (${NEW_FILES} < ${MIN_NEW_FILES})"
                  echo "Skipping retraining"
                  exit 0
                fi

                # Data validation
                echo "Validating data quality..."
                python << 'VALIDATION_EOF'
                import os
                import json

                # Simulate data validation
                data_quality_score = 0.95
                print(f"Data quality score: {data_quality_score}")

                if data_quality_score < 0.9:
                    print("ERROR: Data quality below threshold")
                    exit(1)

                print("Data validation passed")
                VALIDATION_EOF

                # Train model
                echo "Starting model training..."
                python << 'TRAIN_EOF'
                import torch
                import torch.nn as nn
                import json
                from datetime import datetime

                print("Training new model version...")

                class Model(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.fc = nn.Linear(10, 1)
                    def forward(self, x):
                        return self.fc(x)

                model = Model()

                # Simulate training
                for epoch in range(10):
                    loss = 1.0 / (epoch + 1)
                    print(f"Epoch {epoch+1}/10 - Loss: {loss:.4f}")

                # Generate version based on date
                version = datetime.now().strftime("%Y%m%d-%H%M%S")
                model_path = f"/models/scheduled/v-{version}"
                os.makedirs(model_path, exist_ok=True)

                # Save model
                torch.save(model.state_dict(), f"{model_path}/model.pth")

                # Save metadata
                metadata = {
                    "version": version,
                    "trained_at": datetime.now().isoformat(),
                    "training_job": "daily-retraining",
                    "final_loss": float(loss)
                }

                with open(f"{model_path}/metadata.json", 'w') as f:
                    json.dump(metadata, f, indent=2)

                # Update latest symlink
                import os
                latest_link = "/models/scheduled/latest"
                if os.path.islink(latest_link):
                    os.unlink(latest_link)
                os.symlink(f"v-{version}", latest_link)

                print(f"Model saved: {model_path}")
                print(f"Latest link updated: {latest_link}")
                TRAIN_EOF

                # Model evaluation
                echo "Evaluating new model..."
                python << 'EVAL_EOF'
                import json

                # Simulate evaluation
                metrics = {
                    "accuracy": 0.94,
                    "precision": 0.92,
                    "recall": 0.93,
                    "f1_score": 0.925
                }

                print("Evaluation metrics:")
                for key, value in metrics.items():
                    print(f"  {key}: {value:.4f}")

                # Check if new model is better
                # (In production, compare with current production model)
                if metrics["accuracy"] < 0.90:
                    print("WARNING: Model accuracy below threshold")
                    exit(1)

                print("Model evaluation passed")
                EVAL_EOF

                echo ""
                echo "===== Training Job Completed Successfully ====="
                echo "End Time: $(date -Iseconds)"

            env:
            - name: JOB_NAME
              valueFrom:
                fieldRef:
                  fieldPath: metadata.name
            - name: MIN_NEW_FILES
              value: "10"
            - name: PYTHONUNBUFFERED
              value: "1"

            resources:
              requests:
                memory: "4Gi"
                cpu: "2000m"
              limits:
                memory: "8Gi"
                cpu: "4000m"

            volumeMounts:
            - name: training-data
              mountPath: /data
            - name: model-storage
              mountPath: /models

          volumes:
          - name: training-data
            persistentVolumeClaim:
              claimName: training-data-pvc
          - name: model-storage
            persistentVolumeClaim:
              claimName: models-pvc
```

### 2.3 Advanced: Conditional Retraining with Model Drift Detection

**`cronjob-conditional-training.yaml`**:

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: conditional-retraining
  namespace: ml-workloads
spec:
  # Check for drift every 6 hours
  schedule: "0 */6 * * *"
  concurrencyPolicy: Forbid

  jobTemplate:
    spec:
      template:
        spec:
          restartPolicy: OnFailure

          containers:
          - name: drift-detector
            image: python:3.11-slim
            command: ["bash", "-c"]
            args:
              - |
                pip install --quiet numpy scipy scikit-learn

                python << 'DRIFT_EOF'
                import numpy as np
                import json
                from datetime import datetime

                print("Checking for model drift...")

                # Simulate drift detection
                # In production: compare prediction distributions,
                # feature distributions, or performance metrics
                drift_score = np.random.uniform(0, 1)
                drift_threshold = 0.7

                print(f"Drift score: {drift_score:.4f}")
                print(f"Threshold: {drift_threshold}")

                drift_detected = drift_score > drift_threshold

                # Save drift report
                report = {
                    "timestamp": datetime.now().isoformat(),
                    "drift_score": float(drift_score),
                    "threshold": drift_threshold,
                    "drift_detected": drift_detected,
                    "action_required": drift_detected
                }

                with open('/tmp/drift_report.json', 'w') as f:
                    json.dump(report, f, indent=2)

                if drift_detected:
                    print("DRIFT DETECTED - Triggering retraining")
                    # Create marker file to trigger retraining
                    open('/tmp/trigger_retraining', 'w').close()
                else:
                    print("No significant drift detected")

                print(f"Report saved to /tmp/drift_report.json")
                DRIFT_EOF

                # Check if retraining is needed
                if [ -f /tmp/trigger_retraining ]; then
                  echo "Initiating retraining job..."
                  # In production: trigger training job via Kubernetes API or Argo
                  kubectl create job --from=cronjob/daily-model-retraining \
                    drift-triggered-training-$(date +%s) -n ml-workloads || true
                fi

            volumeMounts:
            - name: model-storage
              mountPath: /models
              readOnly: true

          volumes:
          - name: model-storage
            persistentVolumeClaim:
              claimName: models-pvc
```

**CronJob Management Commands**:

```bash
# Create CronJob
kubectl apply -f cronjob-daily-training.yaml

# View CronJobs
kubectl get cronjobs -n ml-workloads

# Trigger manual run
kubectl create job --from=cronjob/daily-model-retraining manual-run-$(date +%s) -n ml-workloads

# Suspend CronJob (for maintenance)
kubectl patch cronjob daily-model-retraining -n ml-workloads -p '{"spec":{"suspend":true}}'

# Resume CronJob
kubectl patch cronjob daily-model-retraining -n ml-workloads -p '{"spec":{"suspend":false}}'

# View job history
kubectl get jobs -n ml-workloads -l app=scheduled-training --sort-by=.metadata.creationTimestamp

# Clean up old jobs
kubectl delete jobs -n ml-workloads --field-selector status.successful=1
```

---

## 3. GPU Node Selection and Allocation

### 3.1 GPU Infrastructure Setup

**Prerequisites**:

1. **Install NVIDIA Device Plugin**:

```bash
# Deploy NVIDIA device plugin
kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.5/nvidia-device-plugin.yml

# Verify installation
kubectl get pods -n kube-system | grep nvidia-device-plugin

# Check GPU availability on nodes
kubectl get nodes -o=custom-columns='NODE:.metadata.name,GPU:.status.capacity.nvidia\.com/gpu'
```

2. **Label GPU Nodes**:

```bash
# Label nodes with GPU type
kubectl label nodes gpu-node-1 accelerator=nvidia-tesla-v100
kubectl label nodes gpu-node-2 accelerator=nvidia-a100

# Label nodes by GPU memory
kubectl label nodes gpu-node-1 gpu-memory=16gb
kubectl label nodes gpu-node-2 gpu-memory=40gb

# Verify labels
kubectl get nodes --show-labels | grep accelerator
```

3. **Optional: Taint GPU Nodes** (to reserve for GPU workloads only):

```bash
# Taint GPU nodes
kubectl taint nodes gpu-node-1 nvidia.com/gpu=present:NoSchedule
kubectl taint nodes gpu-node-2 nvidia.com/gpu=present:NoSchedule
```

### 3.2 GPU Training Job

**`gpu-training-job.yaml`**:

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: pytorch-gpu-training
  namespace: ml-workloads
  labels:
    app: gpu-training
    model: bert-large
spec:
  backoffLimit: 2
  template:
    metadata:
      labels:
        app: gpu-training
    spec:
      restartPolicy: OnFailure

      # Schedule on GPU nodes only
      nodeSelector:
        accelerator: nvidia-tesla-v100

      # Tolerate GPU taints
      tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule

      containers:
      - name: gpu-trainer
        image: pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

        command: ["python"]
        args:
          - "-c"
          - |
            import torch
            import torch.nn as nn
            from datetime import datetime
            import os

            print("="*50)
            print("GPU Training Job Started")
            print("="*50)

            # GPU availability check
            print(f"\nPyTorch version: {torch.__version__}")
            print(f"CUDA available: {torch.cuda.is_available()}")
            print(f"CUDA version: {torch.version.cuda}")
            print(f"cuDNN version: {torch.backends.cudnn.version()}")

            if torch.cuda.is_available():
                print(f"\nGPU Device Count: {torch.cuda.device_count()}")
                for i in range(torch.cuda.device_count()):
                    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                    print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
            else:
                print("\nWARNING: No GPU detected!")
                exit(1)

            # Set device
            device = torch.device("cuda:0")
            print(f"\nUsing device: {device}")

            # Create larger model for GPU
            class LargeModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.layers = nn.Sequential(
                        nn.Linear(1024, 2048),
                        nn.ReLU(),
                        nn.Linear(2048, 2048),
                        nn.ReLU(),
                        nn.Linear(2048, 1024),
                        nn.ReLU(),
                        nn.Linear(1024, 512),
                        nn.ReLU(),
                        nn.Linear(512, 10)
                    )

                def forward(self, x):
                    return self.layers(x)

            model = LargeModel().to(device)
            total_params = sum(p.numel() for p in model.parameters())
            print(f"\nModel parameters: {total_params:,}")

            # Training loop
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()

            batch_size = 128
            num_epochs = 50

            print(f"\nTraining configuration:")
            print(f"  Batch size: {batch_size}")
            print(f"  Epochs: {num_epochs}")
            print(f"  Learning rate: 0.001")

            print("\nStarting training...")
            for epoch in range(num_epochs):
                # Simulate training batch
                inputs = torch.randn(batch_size, 1024).to(device)
                targets = torch.randint(0, 10, (batch_size,)).to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                if (epoch + 1) % 10 == 0:
                    # GPU memory usage
                    allocated = torch.cuda.memory_allocated(0) / 1e9
                    cached = torch.cuda.memory_reserved(0) / 1e9
                    print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {loss.item():.4f} - "
                          f"GPU Memory: {allocated:.2f}GB allocated, {cached:.2f}GB cached")

            # Save model
            model_path = "/models/bert-large-gpu/model.pth"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(model.state_dict(), model_path)

            print(f"\nModel saved to: {model_path}")
            print("\nGPU Training Completed Successfully!")

        env:
        - name: CUDA_VISIBLE_DEVICES
          value: "0"
        - name: NVIDIA_VISIBLE_DEVICES
          value: "all"
        - name: NVIDIA_DRIVER_CAPABILITIES
          value: "compute,utility"

        # GPU resource request
        resources:
          requests:
            memory: "16Gi"
            cpu: "4000m"
            nvidia.com/gpu: 1  # Request 1 GPU
          limits:
            memory: "32Gi"
            cpu: "8000m"
            nvidia.com/gpu: 1  # Limit to 1 GPU

        volumeMounts:
        - name: model-storage
          mountPath: /models
        - name: training-data
          mountPath: /data

      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: models-pvc
      - name: training-data
        persistentVolumeClaim:
          claimName: training-data-pvc
```

### 3.3 Multi-GPU Training Job

**`multi-gpu-training.yaml`**:

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: pytorch-multi-gpu-training
  namespace: ml-workloads
spec:
  backoffLimit: 2
  template:
    spec:
      restartPolicy: OnFailure

      nodeSelector:
        accelerator: nvidia-a100  # Node with multiple GPUs

      tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule

      containers:
      - name: multi-gpu-trainer
        image: pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

        command: ["python"]
        args:
          - "-c"
          - |
            import torch
            import torch.nn as nn
            import torch.distributed as dist

            print(f"Available GPUs: {torch.cuda.device_count()}")

            if torch.cuda.device_count() > 1:
                print("Using DataParallel for multi-GPU training")

                class Model(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.layers = nn.Sequential(
                            nn.Linear(1024, 2048),
                            nn.ReLU(),
                            nn.Linear(2048, 512),
                            nn.ReLU(),
                            nn.Linear(512, 10)
                        )
                    def forward(self, x):
                        return self.layers(x)

                # Wrap model with DataParallel
                model = Model()
                model = nn.DataParallel(model)
                model = model.cuda()

                print(f"Model replicated across {torch.cuda.device_count()} GPUs")

                # Training loop
                for epoch in range(20):
                    inputs = torch.randn(256, 1024).cuda()
                    outputs = model(inputs)
                    loss = outputs.mean()

                    if (epoch + 1) % 5 == 0:
                        print(f"Epoch [{epoch+1}/20] - Loss: {loss.item():.4f}")
                        for i in range(torch.cuda.device_count()):
                            mem = torch.cuda.memory_allocated(i) / 1e9
                            print(f"  GPU {i} memory: {mem:.2f}GB")
            else:
                print("Single GPU available, using standard training")

        resources:
          requests:
            nvidia.com/gpu: 2  # Request 2 GPUs
          limits:
            nvidia.com/gpu: 2

        volumeMounts:
        - name: model-storage
          mountPath: /models

      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: models-pvc
```

### 3.4 GPU Sharing with MIG (Multi-Instance GPU)

**`mig-training-job.yaml`**:

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: mig-training-job
  namespace: ml-workloads
spec:
  parallelism: 7  # Run 7 jobs on single A100 (7 MIG instances)
  completions: 7

  template:
    spec:
      restartPolicy: OnFailure

      containers:
      - name: mig-trainer
        image: pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

        command: ["python", "-c"]
        args:
          - |
            import torch
            import os

            worker_id = os.environ.get('WORKER_ID', 'unknown')
            print(f"Worker {worker_id} starting on MIG instance")
            print(f"GPU Name: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}GB")

            # Train small model on MIG instance
            model = torch.nn.Linear(100, 10).cuda()
            for epoch in range(10):
                inputs = torch.randn(32, 100).cuda()
                outputs = model(inputs)
                loss = outputs.mean()
                loss.backward()

            print(f"Worker {worker_id} completed")

        env:
        - name: WORKER_ID
          valueFrom:
            fieldRef:
              fieldPath: metadata.name

        # Request MIG instance (1g.5gb = 1/7th of A100)
        resources:
          requests:
            nvidia.com/mig-1g.5gb: 1
          limits:
            nvidia.com/mig-1g.5gb: 1
```

**GPU Management Commands**:

```bash
# Check GPU allocation
kubectl describe node gpu-node-1 | grep nvidia.com/gpu

# View GPU jobs
kubectl get jobs -n ml-workloads -l app=gpu-training

# Monitor GPU usage
kubectl exec -n ml-workloads <pod-name> -- nvidia-smi

# Check GPU memory
kubectl exec -n ml-workloads <pod-name> -- \
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

---

## 4. Training Job Patterns

### 4.1 Single-Node Training Pattern

**Best for**:
- Small to medium datasets (< 100GB)
- Models that fit in single GPU memory
- Quick experimentation
- Prototyping

**`single-node-training.yaml`**:

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: single-node-training
  namespace: ml-workloads
spec:
  template:
    spec:
      containers:
      - name: trainer
        image: pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
        command: ["python", "train.py"]
        args:
          - "--data-dir=/data"
          - "--output-dir=/models"
          - "--epochs=50"
          - "--batch-size=32"

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
        - name: data
          mountPath: /data
        - name: models
          mountPath: /models

      volumes:
      - name: data
        persistentVolumeClaim:
          claimName: training-data-pvc
      - name: models
        persistentVolumeClaim:
          claimName: models-pvc
```

### 4.2 Distributed Data Parallel (DDP) Pattern

**Best for**:
- Large datasets (> 100GB)
- Models requiring multi-GPU training
- Production training workloads
- Faster training times

**`distributed-ddp-training.yaml`**:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: pytorch-ddp-master
  namespace: ml-workloads
spec:
  clusterIP: None  # Headless service
  selector:
    job-name: pytorch-ddp-training
  ports:
  - name: master
    port: 23456
    targetPort: 23456

---
apiVersion: batch/v1
kind: Job
metadata:
  name: pytorch-ddp-training
  namespace: ml-workloads
  labels:
    app: distributed-training
    pattern: ddp
spec:
  parallelism: 4  # 4 worker nodes
  completions: 4

  template:
    metadata:
      labels:
        job-name: pytorch-ddp-training
    spec:
      restartPolicy: OnFailure

      nodeSelector:
        accelerator: nvidia-tesla-v100

      containers:
      - name: ddp-worker
        image: pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

        command: ["bash"]
        args:
          - "-c"
          - |
            # Install dependencies
            pip install --quiet torch torchvision

            # Calculate world size and rank
            export WORLD_SIZE=4
            export RANK=$(echo $HOSTNAME | grep -o '[0-9]*$')
            export MASTER_ADDR="pytorch-ddp-master"
            export MASTER_PORT="23456"

            echo "Distributed Training Worker"
            echo "  Rank: $RANK"
            echo "  World Size: $WORLD_SIZE"
            echo "  Master: $MASTER_ADDR:$MASTER_PORT"

            # Run distributed training
            python << 'TRAIN_EOF'
            import torch
            import torch.nn as nn
            import torch.distributed as dist
            from torch.nn.parallel import DistributedDataParallel as DDP
            import os

            # Initialize process group
            dist.init_process_group(
                backend='nccl',  # Use NCCL for GPU
                init_method='env://',
                world_size=int(os.environ['WORLD_SIZE']),
                rank=int(os.environ['RANK'])
            )

            # Set device for this process
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
            torch.cuda.set_device(local_rank)
            device = torch.device(f'cuda:{local_rank}')

            print(f"Rank {dist.get_rank()}/{dist.get_world_size()} initialized")

            # Create model and wrap with DDP
            class SimpleModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.layers = nn.Sequential(
                        nn.Linear(1024, 512),
                        nn.ReLU(),
                        nn.Linear(512, 256),
                        nn.ReLU(),
                        nn.Linear(256, 10)
                    )
                def forward(self, x):
                    return self.layers(x)

            model = SimpleModel().to(device)
            model = DDP(model, device_ids=[local_rank])

            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()

            print(f"Rank {dist.get_rank()}: Starting training...")

            # Training loop
            for epoch in range(50):
                # Create dummy data
                inputs = torch.randn(32, 1024).to(device)
                targets = torch.randint(0, 10, (32,)).to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                if dist.get_rank() == 0 and (epoch + 1) % 10 == 0:
                    print(f"Epoch [{epoch+1}/50] - Loss: {loss.item():.4f}")

            # Save model from rank 0 only
            if dist.get_rank() == 0:
                model_path = "/models/ddp-model/model.pth"
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                torch.save(model.module.state_dict(), model_path)
                print(f"Model saved to: {model_path}")

            # Cleanup
            dist.destroy_process_group()
            print(f"Rank {os.environ['RANK']} completed")
            TRAIN_EOF

        env:
        - name: NCCL_DEBUG
          value: "INFO"
        - name: PYTHONUNBUFFERED
          value: "1"

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
        - name: models
          mountPath: /models
        - name: training-data
          mountPath: /data

        # Required for NCCL communication
        ports:
        - containerPort: 23456
          name: master

      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: models-pvc
      - name: training-data
        persistentVolumeClaim:
          claimName: training-data-pvc
```

### 4.3 PyTorch Elastic Training Pattern (Fault-Tolerant)

**Best for**:
- Long-running training jobs
- Spot/preemptible instances
- Dynamic cluster sizing
- Fault-tolerant training

**`elastic-training.yaml`**:

```yaml
apiVersion: "kubeflow.org/v1"
kind: PyTorchJob
metadata:
  name: elastic-training
  namespace: ml-workloads
spec:
  elasticPolicy:
    rdzvBackend: c10d
    minReplicas: 1
    maxReplicas: 4
    maxRestarts: 10

  pytorchReplicaSpecs:
    Worker:
      replicas: 4
      restartPolicy: OnFailure

      template:
        spec:
          nodeSelector:
            accelerator: nvidia-tesla-v100

          containers:
          - name: pytorch
            image: pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

            command:
              - python
              - -m
              - torch.distributed.run
              - --nnodes=1:4  # Min 1, max 4 nodes
              - --nproc_per_node=1
              - --max_restarts=10
              - --rdzv_backend=c10d
              - --rdzv_endpoint=$(MASTER_ADDR):$(MASTER_PORT)
              - train.py

            args:
              - --epochs=100
              - --checkpoint-interval=10

            resources:
              requests:
                nvidia.com/gpu: 1
              limits:
                nvidia.com/gpu: 1

            volumeMounts:
            - name: models
              mountPath: /models

          volumes:
          - name: models
            persistentVolumeClaim:
              claimName: models-pvc
```

### 4.4 Parameter Server Pattern

**Best for**:
- Very large models (billions of parameters)
- Asynchronous training
- Heterogeneous hardware

**`parameter-server-training.yaml`**:

```yaml
apiVersion: "kubeflow.org/v1"
kind: TFJob
metadata:
  name: parameter-server-training
  namespace: ml-workloads
spec:
  tfReplicaSpecs:
    # Parameter servers (store model parameters)
    PS:
      replicas: 2
      template:
        spec:
          containers:
          - name: tensorflow
            image: tensorflow/tensorflow:2.13.0-gpu
            command:
              - python
              - train.py
              - --job-type=ps
            resources:
              requests:
                memory: "32Gi"
                cpu: "4"

    # Workers (compute gradients)
    Worker:
      replicas: 4
      template:
        spec:
          nodeSelector:
            accelerator: nvidia-tesla-v100

          containers:
          - name: tensorflow
            image: tensorflow/tensorflow:2.13.0-gpu
            command:
              - python
              - train.py
              - --job-type=worker

            resources:
              requests:
                nvidia.com/gpu: 1
                memory: "16Gi"
              limits:
                nvidia.com/gpu: 1

    # Chief (coordinate training)
    Chief:
      replicas: 1
      template:
        spec:
          containers:
          - name: tensorflow
            image: tensorflow/tensorflow:2.13.0-gpu
            command:
              - python
              - train.py
              - --job-type=chief

            volumeMounts:
            - name: models
              mountPath: /models

          volumes:
          - name: models
            persistentVolumeClaim:
              claimName: models-pvc
```

---

## 5. Model Serving with KServe and Seldon

### 5.1 KServe Setup and Deployment

**Install KServe**:

```bash
# Install Istio (required for KServe)
curl -L https://istio.io/downloadIstio | sh -
cd istio-*
export PATH=$PWD/bin:$PATH
istioctl install --set profile=default -y

# Install Knative Serving
kubectl apply -f https://github.com/knative/serving/releases/download/knative-v1.11.0/serving-crds.yaml
kubectl apply -f https://github.com/knative/serving/releases/download/knative-v1.11.0/serving-core.yaml

# Install KServe
kubectl apply -f https://github.com/kserve/kserve/releases/download/v0.11.0/kserve.yaml
kubectl apply -f https://github.com/kserve/kserve/releases/download/v0.11.0/kserve-runtimes.yaml
```

**5.1.1 PyTorch Model Serving with KServe**:

**`kserve-pytorch-inference.yaml`**:

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: pytorch-model
  namespace: ml-workloads
spec:
  predictor:
    pytorch:
      # Storage URI for model artifacts
      storageUri: "pvc://models-pvc/resnet50"

      # Runtime version
      runtimeVersion: "2.1.0-gpu"

      # Resource configuration
      resources:
        requests:
          memory: "4Gi"
          cpu: "2"
          nvidia.com/gpu: 1
        limits:
          memory: "8Gi"
          cpu: "4"
          nvidia.com/gpu: 1

      # Environment variables
      env:
        - name: STORAGE_URI
          value: "pvc://models-pvc/resnet50"

      # Node selection for GPU
      nodeSelector:
        accelerator: nvidia-tesla-v100

      tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule

  # Auto-scaling configuration
  minReplicas: 1
  maxReplicas: 5

  # Traffic configuration
  canaryTrafficPercent: 10
```

**5.1.2 Custom Predictor with KServe**:

**`kserve-custom-predictor.yaml`**:

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: custom-pytorch-model
  namespace: ml-workloads
spec:
  predictor:
    containers:
    - name: kserve-container
      image: pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

      command: ["python"]
      args:
        - "-c"
        - |
          from kserve import Model, ModelServer
          import torch
          import torch.nn as nn
          import json

          class CustomModel(Model):
              def __init__(self, name: str):
                  super().__init__(name)
                  self.name = name
                  self.model = None
                  self.ready = False

              def load(self):
                  # Load PyTorch model
                  class SimpleNet(nn.Module):
                      def __init__(self):
                          super().__init__()
                          self.fc = nn.Linear(10, 1)
                      def forward(self, x):
                          return self.fc(x)

                  self.model = SimpleNet()
                  model_path = "/mnt/models/model.pth"

                  try:
                      self.model.load_state_dict(torch.load(model_path))
                      self.model.eval()
                      self.ready = True
                      print(f"Model loaded from {model_path}")
                  except Exception as e:
                      print(f"Error loading model: {e}")
                      raise

              def predict(self, request):
                  inputs = torch.tensor(request["instances"])

                  with torch.no_grad():
                      outputs = self.model(inputs)

                  return {"predictions": outputs.tolist()}

          if __name__ == "__main__":
              model = CustomModel("custom-pytorch-model")
              model.load()
              ModelServer().start([model])

      ports:
      - containerPort: 8080
        protocol: TCP

      resources:
        requests:
          memory: "2Gi"
          cpu: "1"
        limits:
          memory: "4Gi"
          cpu: "2"

      volumeMounts:
      - name: model-storage
        mountPath: /mnt/models

    volumes:
    - name: model-storage
      persistentVolumeClaim:
        claimName: models-pvc
```

### 5.2 Seldon Core Deployment

**Install Seldon Core**:

```bash
# Install Seldon Core operator
kubectl create namespace seldon-system
kubectl apply -f https://github.com/SeldonIO/seldon-core/releases/download/v1.17.0/seldon-core-operator.yaml
```

**5.2.1 Seldon PyTorch Deployment**:

**`seldon-pytorch-deployment.yaml`**:

```yaml
apiVersion: machinelearning.seldon.io/v1
kind: SeldonDeployment
metadata:
  name: pytorch-classifier
  namespace: ml-workloads
spec:
  name: pytorch-classifier
  protocol: v2

  predictors:
  - name: default
    replicas: 3

    componentSpecs:
    - spec:
        containers:
        - name: classifier
          image: pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

          command:
            - python
            - -c

          args:
            - |
              from seldon_core.user_model import SeldonComponent
              import torch
              import torch.nn as nn
              import numpy as np

              class PyTorchModel(SeldonComponent):
                  def __init__(self):
                      self.model = None
                      self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                  def load(self):
                      # Load model
                      class Model(nn.Module):
                          def __init__(self):
                              super().__init__()
                              self.fc = nn.Linear(784, 10)
                          def forward(self, x):
                              return self.fc(x)

                      self.model = Model().to(self.device)

                      # Load weights
                      model_path = "/mnt/models/model.pth"
                      self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                      self.model.eval()
                      print(f"Model loaded on {self.device}")

                  def predict(self, X, features_names=None, meta=None):
                      # Convert input to tensor
                      X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

                      # Inference
                      with torch.no_grad():
                          outputs = self.model(X_tensor)
                          predictions = torch.softmax(outputs, dim=1)

                      return predictions.cpu().numpy()

              # Start model server
              from seldon_core.microservice import (
                  run_microservice_class,
                  seldon_microservice,
              )

              if __name__ == "__main__":
                  model = PyTorchModel()
                  model.load()
                  run_microservice_class(model)

          resources:
            requests:
              memory: "4Gi"
              cpu: "2"
              nvidia.com/gpu: 1
            limits:
              memory: "8Gi"
              cpu: "4"
              nvidia.com/gpu: 1

          volumeMounts:
          - name: model-storage
            mountPath: /mnt/models

        volumes:
        - name: model-storage
          persistentVolumeClaim:
            claimName: models-pvc

        nodeSelector:
          accelerator: nvidia-tesla-v100

        tolerations:
        - key: nvidia.com/gpu
          operator: Exists
          effect: NoSchedule

    graph:
      name: classifier
      type: MODEL
      endpoint:
        type: REST

    # Auto-scaling
    hpaSpec:
      minReplicas: 1
      maxReplicas: 10
      metrics:
      - type: Resource
        resource:
          name: cpu
          target:
            type: Utilization
            averageUtilization: 70
```

**5.2.2 A/B Testing with Seldon**:

**`seldon-ab-testing.yaml`**:

```yaml
apiVersion: machinelearning.seldon.io/v1
kind: SeldonDeployment
metadata:
  name: ab-test-deployment
  namespace: ml-workloads
spec:
  name: ab-test
  predictors:
  # Model A (90% traffic)
  - name: model-a
    replicas: 3
    traffic: 90

    graph:
      name: model-a-classifier
      type: MODEL
      endpoint:
        type: REST

    componentSpecs:
    - spec:
        containers:
        - name: model-a
          image: pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
          env:
          - name: MODEL_VERSION
            value: "v1.0"
          - name: MODEL_PATH
            value: "/models/v1.0/model.pth"

          volumeMounts:
          - name: models
            mountPath: /models

        volumes:
        - name: models
          persistentVolumeClaim:
            claimName: models-pvc

  # Model B (10% traffic - canary)
  - name: model-b
    replicas: 1
    traffic: 10

    graph:
      name: model-b-classifier
      type: MODEL
      endpoint:
        type: REST

    componentSpecs:
    - spec:
        containers:
        - name: model-b
          image: pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
          env:
          - name: MODEL_VERSION
            value: "v2.0"
          - name: MODEL_PATH
            value: "/models/v2.0/model.pth"

          volumeMounts:
          - name: models
            mountPath: /models

        volumes:
        - name: models
          persistentVolumeClaim:
            claimName: models-pvc
```

**Testing Model Serving**:

```bash
# KServe inference
curl -X POST \
  https://pytorch-model.ml-workloads.example.com/v1/models/pytorch-model:predict \
  -H 'Content-Type: application/json' \
  -d '{
    "instances": [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]]
  }'

# Seldon inference
curl -X POST \
  http://pytorch-classifier-default.ml-workloads.svc.cluster.local:8000/api/v1.0/predictions \
  -H 'Content-Type: application/json' \
  -d '{
    "data": {
      "ndarray": [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]]
    }
  }'
```

---

## 6. Resource Quotas and Priorities

### 6.1 Resource Quotas for ML Workloads

**`ml-resource-quotas.yaml`**:

```yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: ml-training-quota
  namespace: ml-workloads
spec:
  hard:
    # Compute resources
    requests.cpu: "100"
    requests.memory: "200Gi"
    requests.nvidia.com/gpu: "16"
    limits.cpu: "200"
    limits.memory: "400Gi"
    limits.nvidia.com/gpu: "16"

    # Storage
    requests.storage: "5Ti"
    persistentvolumeclaims: "20"

    # Object counts
    pods: "100"
    services: "20"
    count/jobs.batch: "50"
    count/cronjobs.batch: "10"

  # Scope to specific priority classes
  scopeSelector:
    matchExpressions:
    - operator: In
      scopeName: PriorityClass
      values: ["high-priority", "medium-priority"]

---
apiVersion: v1
kind: ResourceQuota
metadata:
  name: ml-serving-quota
  namespace: ml-workloads
spec:
  hard:
    requests.cpu: "50"
    requests.memory: "100Gi"
    requests.nvidia.com/gpu: "8"
    limits.cpu: "100"
    limits.memory: "200Gi"
    limits.nvidia.com/gpu: "8"

    pods: "50"
    services: "10"
```

### 6.2 Priority Classes

**`ml-priority-classes.yaml`**:

```yaml
# Critical production serving
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: ml-serving-critical
value: 1000000
globalDefault: false
description: "Critical ML model serving workloads"

---
# High priority training
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: ml-training-high
value: 100000
globalDefault: false
description: "High priority training jobs"
preemptionPolicy: PreemptLowerPriority

---
# Medium priority training
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: ml-training-medium
value: 10000
globalDefault: false
description: "Medium priority training jobs"

---
# Low priority (preemptible)
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: ml-training-low
value: 1000
globalDefault: false
description: "Low priority, can be preempted"
```

### 6.3 Using Priority Classes

**`priority-training-job.yaml`**:

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: high-priority-training
  namespace: ml-workloads
spec:
  template:
    spec:
      # Set priority class
      priorityClassName: ml-training-high

      containers:
      - name: trainer
        image: pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
        command: ["python", "train.py"]

        resources:
          requests:
            nvidia.com/gpu: 2
            memory: "32Gi"
            cpu: "8"

      # This job can preempt lower priority jobs
      preemptionPolicy: PreemptLowerPriority
```

### 6.4 Limit Ranges

**`ml-limit-ranges.yaml`**:

```yaml
apiVersion: v1
kind: LimitRange
metadata:
  name: ml-workload-limits
  namespace: ml-workloads
spec:
  limits:
  # Container limits
  - type: Container
    max:
      cpu: "16"
      memory: "64Gi"
      nvidia.com/gpu: "4"
    min:
      cpu: "100m"
      memory: "128Mi"
    default:
      cpu: "2"
      memory: "4Gi"
    defaultRequest:
      cpu: "1"
      memory: "2Gi"
    maxLimitRequestRatio:
      cpu: "2"  # Limit can be at most 2x request
      memory: "2"

  # PVC limits
  - type: PersistentVolumeClaim
    max:
      storage: "1Ti"
    min:
      storage: "1Gi"
```

---

## 7. Production ML Workflow Orchestration

### 7.1 Argo Workflows for ML Pipelines

**Install Argo Workflows**:

```bash
kubectl create namespace argo
kubectl apply -n argo -f https://github.com/argoproj/argo-workflows/releases/download/v3.5.0/install.yaml
```

**7.1.1 Complete ML Training Pipeline**:

**`argo-ml-pipeline.yaml`**:

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Workflow
metadata:
  generateName: ml-training-pipeline-
  namespace: ml-workloads
spec:
  entrypoint: ml-pipeline

  # Volume claims for sharing data between steps
  volumeClaimTemplates:
  - metadata:
      name: workspace
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: "50Gi"

  templates:
  # Main pipeline
  - name: ml-pipeline
    steps:
    - - name: data-validation
        template: validate-data

    - - name: data-preprocessing
        template: preprocess-data

    - - name: training
        template: train-model

    - - name: evaluation
        template: evaluate-model

    - - name: model-registration
        template: register-model
        when: "{{steps.evaluation.outputs.parameters.accuracy}} > 0.90"

  # Step 1: Data validation
  - name: validate-data
    container:
      image: python:3.11-slim
      command: ["bash"]
      args:
        - "-c"
        - |
          pip install pandas numpy
          python << 'EOF'
          import pandas as pd
          import json

          print("Validating training data...")

          # Simulate validation
          validation_report = {
              "total_samples": 10000,
              "missing_values": 50,
              "duplicates": 10,
              "quality_score": 0.95
          }

          with open('/workspace/validation_report.json', 'w') as f:
              json.dump(validation_report, f)

          print("Validation complete")
          print(f"Quality score: {validation_report['quality_score']}")
          EOF

      volumeMounts:
      - name: workspace
        mountPath: /workspace

  # Step 2: Data preprocessing
  - name: preprocess-data
    container:
      image: python:3.11-slim
      command: ["bash"]
      args:
        - "-c"
        - |
          pip install pandas scikit-learn
          python << 'EOF'
          import pandas as pd
          import pickle

          print("Preprocessing data...")

          # Simulate preprocessing
          # In production: load data, clean, transform, split

          preprocessing_config = {
              "scaler": "StandardScaler",
              "feature_selection": "SelectKBest",
              "train_test_split": 0.8
          }

          with open('/workspace/preprocessing_config.pkl', 'wb') as f:
              pickle.dump(preprocessing_config, f)

          print("Preprocessing complete")
          EOF

      volumeMounts:
      - name: workspace
        mountPath: /workspace

  # Step 3: Model training
  - name: train-model
    container:
      image: pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
      command: ["python"]
      args:
        - "-c"
        - |
          import torch
          import torch.nn as nn
          import json
          from datetime import datetime

          print("Training model...")

          class Model(nn.Module):
              def __init__(self):
                  super().__init__()
                  self.fc = nn.Linear(100, 10)
              def forward(self, x):
                  return self.fc(x)

          model = Model()

          # Training loop
          for epoch in range(20):
              loss = 1.0 / (epoch + 1)
              if (epoch + 1) % 5 == 0:
                  print(f"Epoch {epoch+1}/20 - Loss: {loss:.4f}")

          # Save model
          torch.save(model.state_dict(), '/workspace/model.pth')

          # Save training metadata
          metadata = {
              "version": datetime.now().strftime("%Y%m%d-%H%M%S"),
              "epochs": 20,
              "final_loss": float(loss)
          }

          with open('/workspace/training_metadata.json', 'w') as f:
              json.dump(metadata, f)

          print("Training complete")

      resources:
        requests:
          nvidia.com/gpu: 1
          memory: "8Gi"
          cpu: "4"

      volumeMounts:
      - name: workspace
        mountPath: /workspace

      nodeSelector:
        accelerator: nvidia-tesla-v100

  # Step 4: Model evaluation
  - name: evaluate-model
    outputs:
      parameters:
      - name: accuracy
        valueFrom:
          path: /tmp/accuracy.txt

    container:
      image: pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
      command: ["bash"]
      args:
        - "-c"
        - |
          python << 'EOF'
          import torch
          import json

          print("Evaluating model...")

          # Simulate evaluation
          accuracy = 0.94
          precision = 0.93
          recall = 0.92
          f1_score = 0.925

          metrics = {
              "accuracy": accuracy,
              "precision": precision,
              "recall": recall,
              "f1_score": f1_score
          }

          print("Evaluation metrics:")
          for key, value in metrics.items():
              print(f"  {key}: {value:.4f}")

          # Save metrics
          with open('/workspace/evaluation_metrics.json', 'w') as f:
              json.dump(metrics, f)

          # Output accuracy for workflow decision
          with open('/tmp/accuracy.txt', 'w') as f:
              f.write(str(accuracy))

          print("Evaluation complete")
          EOF

      volumeMounts:
      - name: workspace
        mountPath: /workspace

  # Step 5: Register model (conditional)
  - name: register-model
    container:
      image: python:3.11-slim
      command: ["bash"]
      args:
        - "-c"
        - |
          echo "Registering model in model registry..."

          # Copy to models PVC
          mkdir -p /models/production/
          cp /workspace/model.pth /models/production/
          cp /workspace/training_metadata.json /models/production/
          cp /workspace/evaluation_metrics.json /models/production/

          echo "Model registered successfully"

      volumeMounts:
      - name: workspace
        mountPath: /workspace
      - name: models-pvc
        mountPath: /models

  # PVC for model storage
  volumes:
  - name: models-pvc
    persistentVolumeClaim:
      claimName: models-pvc
```

### 7.2 Running the Pipeline

```bash
# Submit workflow
argo submit -n ml-workloads argo-ml-pipeline.yaml

# Watch workflow
argo watch -n ml-workloads @latest

# Get workflow logs
argo logs -n ml-workloads @latest

# List workflows
argo list -n ml-workloads

# Get workflow status
argo get -n ml-workloads @latest
```

### 7.3 Automated CI/CD Pipeline for ML

**`.github/workflows/ml-cicd.yaml`** (GitHub Actions example):

```yaml
name: ML Model CI/CD

on:
  push:
    branches: [main]
    paths:
      - 'models/**'
      - 'training/**'

jobs:
  train-and-deploy:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: |
        pip install torch torchvision pytest

    - name: Run tests
      run: |
        pytest tests/

    - name: Train model
      run: |
        python train.py --epochs=10 --output-dir=./models

    - name: Evaluate model
      id: evaluate
      run: |
        accuracy=$(python evaluate.py --model-dir=./models)
        echo "accuracy=$accuracy" >> $GITHUB_OUTPUT

    - name: Configure kubectl
      uses: azure/k8s-set-context@v3
      with:
        kubeconfig: ${{ secrets.KUBE_CONFIG }}

    - name: Deploy to Kubernetes
      if: steps.evaluate.outputs.accuracy > 0.90
      run: |
        kubectl apply -f k8s/model-serving.yaml
        kubectl rollout status deployment/model-server -n ml-workloads
```

---

## Conclusion

This implementation guide covered:

1. **Kubernetes Jobs** for ML training with checkpointing
2. **CronJobs** for scheduled retraining and drift detection
3. **GPU allocation** with node selection, MIG, and multi-GPU training
4. **Training patterns**: single-node, DDP, elastic, and parameter server
5. **Model serving** with KServe and Seldon Core
6. **Resource management** with quotas, limits, and priorities
7. **ML workflow orchestration** using Argo Workflows

For production deployments, consider:
- Implementing monitoring and alerting (Prometheus, Grafana)
- Adding model versioning and registry (MLflow, DVC)
- Setting up experiment tracking
- Implementing data versioning
- Adding security policies and RBAC
- Establishing model governance and compliance
- Creating disaster recovery procedures

**Next Steps**:
- Explore Kubeflow for end-to-end ML platforms
- Implement feature stores (Feast)
- Add model explainability (SHAP, LIME)
- Set up continuous training pipelines
- Implement shadow deployments
- Add online learning capabilities
