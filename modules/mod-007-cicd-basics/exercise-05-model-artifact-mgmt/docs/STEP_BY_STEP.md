# Step-by-Step Implementation Guide: Model Artifact Management

## Overview

Manage ML model artifacts in CI/CD! Learn model versioning, registry integration, automated model testing, deployment automation, and artifact tracking.

**Time**: 2 hours | **Difficulty**: Intermediate to Advanced

---

## Learning Objectives

✅ Version and track model artifacts
✅ Integrate MLflow with CI/CD
✅ Automate model validation
✅ Deploy models from registry
✅ Implement model rollback
✅ Track model lineage
✅ Monitor model performance

---

## MLflow Integration

```.github/workflows/model-training.yml
name: Model Training & Registration

on:
  workflow_dispatch:
    inputs:
      experiment_name:
        description: 'MLflow experiment name'
        required: true
        default: 'production-model'

jobs:
  train:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install mlflow torch scikit-learn

      - name: Train model
        env:
          MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}
          MLFLOW_TRACKING_USERNAME: ${{ secrets.MLFLOW_USERNAME }}
          MLFLOW_TRACKING_PASSWORD: ${{ secrets.MLFLOW_PASSWORD }}
        run: |
          python scripts/train.py \
            --experiment-name ${{ github.event.inputs.experiment_name }} \
            --register-model true

      - name: Get best model
        id: model
        run: |
          MODEL_URI=$(python scripts/get_best_model.py)
          echo "model_uri=$MODEL_URI" >> $GITHUB_OUTPUT

      - name: Validate model
        run: |
          python scripts/validate_model.py --model-uri ${{ steps.model.outputs.model_uri }}

      - name: Promote to production
        if: success()
        run: |
          python scripts/promote_model.py \
            --model-uri ${{ steps.model.outputs.model_uri }} \
            --stage Production
```

---

## Training Script with MLflow

```python
# scripts/train.py
import mlflow
import mlflow.pytorch
import argparse

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-name', required=True)
    parser.add_argument('--register-model', type=bool, default=False)
    args = parser.parse_args()

    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run():
        # Log parameters
        mlflow.log_params({
            'epochs': 100,
            'batch_size': 32,
            'learning_rate': 0.001
        })

        # Train model
        model = train_model()

        # Log metrics
        mlflow.log_metrics({
            'accuracy': accuracy,
            'loss': loss,
            'f1_score': f1
        })

        # Log model
        mlflow.pytorch.log_model(
            model,
            "model",
            registered_model_name="production-classifier" if args.register_model else None
        )

if __name__ == "__main__":
    train()
```

---

## Model Validation

```python
# scripts/validate_model.py
import mlflow
import argparse

def validate_model(model_uri):
    # Load model
    model = mlflow.pytorch.load_model(model_uri)

    # Run validation tests
    test_data = load_test_data()

    # Performance tests
    accuracy = evaluate_accuracy(model, test_data)
    latency = measure_latency(model, test_data)

    # Quality gates
    assert accuracy > 0.85, f"Accuracy {accuracy} below threshold"
    assert latency < 100, f"Latency {latency}ms above threshold"

    # Bias/fairness tests
    bias_score = check_bias(model, test_data)
    assert bias_score < 0.1, f"Bias score {bias_score} too high"

    print("✅ All validation checks passed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-uri', required=True)
    args = parser.parse_args()

    validate_model(args.model_uri)
```

---

## Model Deployment

```.github/workflows/model-deploy.yml
name: Deploy Model

on:
  repository_dispatch:
    types: [model-promoted]

jobs:
  deploy:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Download model from MLflow
        env:
          MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}
        run: |
          python scripts/download_model.py \
            --model-name production-classifier \
            --stage Production \
            --output-dir ./models

      - name: Build Docker image
        run: |
          docker build -t ml-api:${{ github.sha }} .
          docker tag ml-api:${{ github.sha }} ml-api:latest

      - name: Push to registry
        run: |
          docker push ml-api:${{ github.sha }}
          docker push ml-api:latest

      - name: Deploy to Kubernetes
        run: |
          kubectl set image deployment/ml-api \
            api=ml-api:${{ github.sha }} \
            -n production

      - name: Verify deployment
        run: |
          kubectl rollout status deployment/ml-api -n production
```

---

## Model Monitoring

```python
# scripts/monitor_model.py
import mlflow
from datetime import datetime

def log_prediction_metrics(predictions, actuals):
    """Log real-time prediction metrics"""

    with mlflow.start_run():
        mlflow.log_metrics({
            'prediction_count': len(predictions),
            'mean_confidence': predictions.mean(),
            'timestamp': datetime.now().timestamp()
        })

        # Log drift metrics
        if actuals is not None:
            accuracy = compute_accuracy(predictions, actuals)
            mlflow.log_metric('online_accuracy', accuracy)

            # Alert if drift detected
            if accuracy < 0.80:
                send_alert("Model performance degraded")
```

---

## Best Practices

✅ Version all model artifacts
✅ Track model lineage (data, code, params)
✅ Validate models before deployment
✅ Implement quality gates
✅ Monitor model performance post-deployment
✅ Automate model rollback
✅ Use model registry (MLflow, Weights & Biases)
✅ Log all experiments
✅ Test for bias and fairness
✅ Implement A/B testing for models

---

**Model Artifact Management mastered!** 🤖
