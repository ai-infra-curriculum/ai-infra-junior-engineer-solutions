# Exercise 06: End-to-End ML Pipeline

## Overview

Build a complete, production-ready ML pipeline integrating all CI/CD concepts from previous exercises. This capstone project demonstrates a full MLOps workflow from data preparation through model deployment.

## Learning Objectives

- Integrate Git workflows with ML pipelines
- Implement automated testing in ML workflows
- Containerize ML pipeline components
- Deploy pipelines to Kubernetes
- Version models and artifacts
- Implement monitoring and alerting
- Build reproducible ML workflows

## Integration Points

This exercise builds upon:

- **Exercise 01**: Git workflows, GitHub Actions, PR validation
- **Exercise 02**: Automated testing, pytest, coverage
- **Exercise 03**: Docker images, multi-stage builds, registries
- **Exercise 04**: Kubernetes deployments, Helm charts, GitOps
- **Exercise 05**: MLflow tracking, model registry, DVC

## Project Structure

```
exercise-06-end-to-end-pipeline/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── ingestion.py          # Data ingestion
│   │   ├── validation.py         # Data validation
│   │   └── preprocessing.py      # Data preprocessing
│   ├── features/
│   │   ├── __init__.py
│   │   └── engineering.py        # Feature engineering
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py              # Model training
│   │   └── evaluate.py           # Model evaluation
│   └── serve/
│       ├── __init__.py
│       └── api.py                # Prediction API
├── tests/
│   ├── test_data.py
│   ├── test_features.py
│   ├── test_models.py
│   └── test_api.py
├── pipelines/
│   ├── training_pipeline.py       # Complete training pipeline
│   ├── inference_pipeline.py      # Inference pipeline
│   └── monitoring_pipeline.py     # Monitoring pipeline
├── .github/
│   └── workflows/
│       ├── pipeline-ci.yml        # CI for pipeline code
│       ├── pipeline-training.yml  # Automated training
│       ├── pipeline-deploy.yml    # Automated deployment
│       └── pipeline-monitoring.yml # Monitoring checks
├── kubernetes/
│   ├── training-job.yaml          # K8s job for training
│   ├── inference-deployment.yaml  # Inference service
│   └── monitoring-deployment.yaml # Monitoring service
└── README.md
```

## Quick Start

### 1. Run Complete Pipeline Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run training pipeline
python pipelines/training_pipeline.py \
  --data-path data/dataset.csv \
  --experiment-name end-to-end-demo

# Run inference
python pipelines/inference_pipeline.py \
  --model-uri models:/RandomForestClassifier/Production
```

### 2. Run with Docker

```bash
# Build pipeline image
docker build -t ml-pipeline:latest .

# Run training
docker run ml-pipeline:latest python pipelines/training_pipeline.py

# Run inference service
docker run -p 8000:8000 ml-pipeline:latest python src/serve/api.py
```

### 3. Deploy to Kubernetes

```bash
# Deploy training job
kubectl apply -f kubernetes/training-job.yaml

# Deploy inference service
kubectl apply -f kubernetes/inference-deployment.yaml

# Check status
kubectl get pods -n ml-pipeline
```

## Pipeline Components

### Data Ingestion

```python
from src.data.ingestion import DataIngester

# Load data from various sources
ingester = DataIngester()
data = ingester.load_from_csv("data/raw.csv")
# data = ingester.load_from_s3("s3://bucket/data.csv")
# data = ingester.load_from_database("postgresql://...")
```

### Data Validation

```python
from src.data.validation import DataValidator

# Validate data quality
validator = DataValidator()
is_valid, issues = validator.validate(data)

if not is_valid:
    raise ValueError(f"Data validation failed: {issues}")
```

### Feature Engineering

```python
from src.features.engineering import FeatureEngineer

# Transform features
engineer = FeatureEngineer()
features = engineer.transform(data)
```

### Model Training

```python
from src.models.train import ModelTrainer

# Train model with MLflow tracking
trainer = ModelTrainer(experiment_name="production")
model, metrics = trainer.train(features, target)
```

### Model Evaluation

```python
from src.models.evaluate import ModelEvaluator

# Evaluate model performance
evaluator = ModelEvaluator()
metrics = evaluator.evaluate(model, X_test, y_test)

# Check if model meets production criteria
if metrics["accuracy"] > 0.85:
    print("Model ready for production!")
```

## End-to-End Training Pipeline

The training pipeline orchestrates all components:

```python
# pipelines/training_pipeline.py

import mlflow
from src.data.ingestion import DataIngester
from src.data.validation import DataValidator
from src.data.preprocessing import DataPreprocessor
from src.features.engineering import FeatureEngineer
from src.models.train import ModelTrainer
from src.models.evaluate import ModelEvaluator

def run_training_pipeline(config):
    with mlflow.start_run(run_name="end-to-end-training"):
        # 1. Ingest data
        ingester = DataIngester()
        raw_data = ingester.load(config.data_source)
        mlflow.log_param("data_source", config.data_source)

        # 2. Validate data
        validator = DataValidator()
        is_valid, issues = validator.validate(raw_data)
        if not is_valid:
            raise ValueError(f"Validation failed: {issues}")

        # 3. Preprocess data
        preprocessor = DataPreprocessor()
        clean_data = preprocessor.process(raw_data)

        # 4. Engineer features
        engineer = FeatureEngineer()
        features = engineer.transform(clean_data)

        # 5. Train model
        trainer = ModelTrainer()
        model, train_metrics = trainer.train(features)

        # 6. Evaluate model
        evaluator = ModelEvaluator()
        eval_metrics = evaluator.evaluate(model, features)

        # 7. Register model if meets criteria
        if eval_metrics["accuracy"] >= config.accuracy_threshold:
            mlflow.sklearn.log_model(model, "model")
            print("✓ Model registered successfully")

        return model, eval_metrics
```

## CI/CD Workflows

### Continuous Integration

```yaml
# .github/workflows/pipeline-ci.yml
name: Pipeline CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run tests
        run: |
          pytest tests/ --cov=src --cov-report=html
      - name: Lint code
        run: |
          flake8 src/ tests/
          black --check src/ tests/
```

### Automated Training

```yaml
# .github/workflows/pipeline-training.yml
name: Automated Training

on:
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM
  workflow_dispatch:

jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - name: Run training pipeline
        run: |
          python pipelines/training_pipeline.py \
            --experiment-name scheduled-training
```

### Automated Deployment

```yaml
# .github/workflows/pipeline-deploy.yml
name: Deploy Pipeline

on:
  release:
    types: [published]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - name: Build and push Docker image
        run: |
          docker build -t ml-pipeline:${{ github.ref_name }} .
          docker push ml-pipeline:${{ github.ref_name }}

      - name: Deploy to Kubernetes
        run: |
          kubectl set image deployment/ml-inference \
            ml-inference=ml-pipeline:${{ github.ref_name }}
```

## Kubernetes Deployment

### Training Job

```yaml
# kubernetes/training-job.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: ml-training
spec:
  template:
    spec:
      containers:
      - name: trainer
        image: ml-pipeline:latest
        command: ["python", "pipelines/training_pipeline.py"]
        env:
        - name: MLFLOW_TRACKING_URI
          valueFrom:
            secretKeyRef:
              name: mlflow-secrets
              key: tracking-uri
      restartPolicy: Never
```

### Inference Deployment

```yaml
# kubernetes/inference-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-inference
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ml-inference
  template:
    metadata:
      labels:
        app: ml-inference
    spec:
      containers:
      - name: api
        image: ml-pipeline:latest
        command: ["python", "src/serve/api.py"]
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
```

## Monitoring & Observability

### Model Performance Monitoring

```python
# pipelines/monitoring_pipeline.py

import mlflow
from prometheus_client import Gauge, Counter

# Metrics
prediction_latency = Gauge('prediction_latency_seconds', 'Prediction latency')
prediction_count = Counter('predictions_total', 'Total predictions')
model_accuracy = Gauge('model_accuracy', 'Current model accuracy')

def monitor_predictions(model, data):
    """Monitor model predictions in production."""
    with mlflow.start_run(run_name="monitoring"):
        predictions = model.predict(data)

        # Log metrics
        mlflow.log_metric("predictions_count", len(predictions))

        # Check for data drift
        drift_score = detect_drift(data)
        mlflow.log_metric("drift_score", drift_score)

        if drift_score > 0.1:
            send_alert("Data drift detected!")
```

### Alerts

```python
def send_alert(message):
    """Send alert to monitoring system."""
    # Slack, PagerDuty, email, etc.
    print(f"ALERT: {message}")
```

## Best Practices

### 1. Reproducibility

✅ Pin all dependencies
✅ Set random seeds
✅ Version data with DVC
✅ Log all parameters with MLflow

### 2. Testing

✅ Unit tests for all components
✅ Integration tests for pipeline
✅ Data validation tests
✅ Model performance tests

### 3. Monitoring

✅ Track data quality
✅ Monitor model performance
✅ Log prediction latency
✅ Detect data drift

### 4. Automation

✅ Automated training on schedule
✅ Automated testing on PR
✅ Automated deployment on release
✅ Automated monitoring alerts

## Resources

- [MLOps Principles](https://ml-ops.org/content/mlops-principles)
- [Kubernetes Best Practices](https://kubernetes.io/docs/concepts/configuration/overview/)
- [MLflow Production Guide](https://mlflow.org/docs/latest/production.html)

## Next Steps

After completing this exercise:

1. ✅ Understand end-to-end ML workflows
2. ✅ Build production-ready pipelines
3. ✅ Integrate CI/CD with MLOps
4. ✅ Deploy and monitor ML systems
5. ✅ Implement best practices

**Congratulations!** You've completed the CI/CD Basics module!
