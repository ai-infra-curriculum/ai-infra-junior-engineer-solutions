# Step-by-Step Guide: Model Artifact Management

## Overview
Implement comprehensive model versioning and artifact storage using MLflow and DVC for tracking experiments, managing model lifecycle, and ensuring reproducibility.

## Phase 1: MLflow Setup (15 minutes)

### Install and Initialize MLflow
```bash
# Create project structure
mkdir -p model-artifacts/{models,data,experiments}
cd model-artifacts

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install mlflow scikit-learn pandas numpy boto3
pip freeze > requirements.txt
```

### Start MLflow Tracking Server
```bash
# Start local MLflow server
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlruns \
    --host 0.0.0.0 \
    --port 5000 &

# Verify server is running
curl http://localhost:5000/health
```

### Create Training Script with MLflow
Create `train_model.py`:
```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

# Set tracking URI
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("model-training")

def train_model(n_estimators=100, max_depth=10, random_state=42):
    # Start MLflow run
    with mlflow.start_run(run_name=f"rf-{n_estimators}-{max_depth}"):
        # Log parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)

        # Generate dataset
        X, y = make_classification(n_samples=1000, n_features=20, random_state=random_state)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # Train model
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)

        # Log model
        mlflow.sklearn.log_model(model, "model")

        # Log additional artifacts
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        feature_importance = model.feature_importances_
        plt.bar(range(len(feature_importance)), feature_importance)
        plt.title("Feature Importance")
        plt.savefig("feature_importance.png")
        mlflow.log_artifact("feature_importance.png")

        print(f"Run ID: {mlflow.active_run().info.run_id}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")

        return mlflow.active_run().info.run_id

if __name__ == "__main__":
    # Train multiple models with different parameters
    for n_est in [50, 100, 200]:
        for depth in [5, 10, 15]:
            train_model(n_estimators=n_est, max_depth=depth)
```

**Validation**: Run `python train_model.py` and check MLflow UI at http://localhost:5000

## Phase 2: Model Registry (15 minutes)

### Register Best Model
Create `register_model.py`:
```python
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://localhost:5000")
client = MlflowClient()

def register_best_model(experiment_name="model-training", metric="accuracy"):
    # Get experiment
    experiment = client.get_experiment_by_name(experiment_name)

    # Find best run
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} DESC"],
        max_results=1
    )

    if not runs:
        print("No runs found")
        return

    best_run = runs[0]
    run_id = best_run.info.run_id

    print(f"Best run ID: {run_id}")
    print(f"Best {metric}: {best_run.data.metrics[metric]}")

    # Register model
    model_uri = f"runs:/{run_id}/model"
    model_details = mlflow.register_model(model_uri, "RandomForestClassifier")

    print(f"Model registered: {model_details.name}")
    print(f"Version: {model_details.version}")

    return model_details

def transition_model_stage(model_name, version, stage):
    """Transition model to different stage"""
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=stage,
        archive_existing_versions=True
    )
    print(f"Model {model_name} v{version} transitioned to {stage}")

if __name__ == "__main__":
    # Register best model
    model = register_best_model()

    # Transition to staging
    transition_model_stage("RandomForestClassifier", model.version, "Staging")

    # After validation, transition to production
    # transition_model_stage("RandomForestClassifier", model.version, "Production")
```

**Validation**: Run `python register_model.py` and verify model in MLflow UI under Models tab.

## Phase 3: DVC Integration (15 minutes)

### Initialize DVC
```bash
# Initialize git repo
git init
git config user.name "Your Name"
git config user.email "your.email@example.com"

# Initialize DVC
pip install dvc dvc-s3
dvc init

# Create data directory
mkdir -p data/raw data/processed
```

### Track Data with DVC
```bash
# Generate sample dataset
python << 'EOF'
import pandas as pd
import numpy as np

# Create sample dataset
df = pd.DataFrame({
    'feature1': np.random.rand(10000),
    'feature2': np.random.rand(10000),
    'label': np.random.randint(0, 2, 10000)
})
df.to_csv('data/raw/dataset.csv', index=False)
print("Dataset created: data/raw/dataset.csv")
EOF

# Track with DVC
dvc add data/raw/dataset.csv

# Commit to git
git add data/raw/dataset.csv.dvc data/raw/.gitignore
git commit -m "Track dataset with DVC"
```

### Create DVC Pipeline
Create `dvc.yaml`:
```yaml
stages:
  prepare:
    cmd: python scripts/prepare_data.py
    deps:
      - data/raw/dataset.csv
      - scripts/prepare_data.py
    outs:
      - data/processed/train.csv
      - data/processed/test.csv
    params:
      - prepare.test_size
      - prepare.random_state

  train:
    cmd: python scripts/train.py
    deps:
      - data/processed/train.csv
      - scripts/train.py
    outs:
      - models/model.pkl
    params:
      - train.n_estimators
      - train.max_depth
    metrics:
      - metrics.json:
          cache: false

  evaluate:
    cmd: python scripts/evaluate.py
    deps:
      - data/processed/test.csv
      - models/model.pkl
      - scripts/evaluate.py
    metrics:
      - evaluation.json:
          cache: false
```

Create `params.yaml`:
```yaml
prepare:
  test_size: 0.2
  random_state: 42

train:
  n_estimators: 100
  max_depth: 10
  random_state: 42
```

**Validation**: Run `dvc dag` to visualize pipeline dependencies.

## Phase 4: Remote Storage (15 minutes)

### Configure S3 Remote (or Local Remote)
```bash
# Option 1: Local remote storage for testing
mkdir -p /tmp/dvc-storage
dvc remote add -d local /tmp/dvc-storage

# Option 2: S3 remote (if you have AWS credentials)
# dvc remote add -d s3remote s3://my-bucket/dvc-storage
# dvc remote modify s3remote region us-east-1
```

### Push Artifacts to Remote
```bash
# Push data to remote
dvc push

# Verify .dvc/config
cat .dvc/config

# Remove local cache and pull
rm -rf .dvc/cache
dvc pull

# Verify data is restored
ls -lh data/raw/
```

### Version Data Changes
```bash
# Modify dataset
python << 'EOF'
import pandas as pd
df = pd.read_csv('data/raw/dataset.csv')
df['feature3'] = df['feature1'] * df['feature2']
df.to_csv('data/raw/dataset.csv', index=False)
EOF

# Track new version
dvc add data/raw/dataset.csv
git add data/raw/dataset.csv.dvc
git commit -m "Add feature3 to dataset"
dvc push

# Switch between versions
git checkout HEAD~1 data/raw/dataset.csv.dvc
dvc checkout
```

**Validation**: Verify ability to switch between dataset versions.

## Phase 5: CI/CD Integration (15 minutes)

### Create GitHub Actions Workflow
Create `.github/workflows/model-training.yml`:
```yaml
name: Model Training and Versioning

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

env:
  MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}

jobs:
  train-and-register:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: |
        pip install -r requirements.txt

    - name: Pull data from DVC
      env:
        AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
        AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
      run: |
        dvc pull

    - name: Run DVC pipeline
      run: |
        dvc repro

    - name: Train model with MLflow
      run: |
        python train_model.py

    - name: Register best model
      if: github.ref == 'refs/heads/main'
      run: |
        python register_model.py

    - name: Push DVC changes
      if: github.ref == 'refs/heads/main'
      env:
        AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
        AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
      run: |
        dvc push

    - name: Upload metrics
      uses: actions/upload-artifact@v3
      with:
        name: metrics
        path: |
          metrics.json
          evaluation.json
```

**Validation**: Push to GitHub and verify workflow executes successfully.

## Phase 6: Model Loading and Serving (10 minutes)

### Load Model from Registry
Create `serve_model.py`:
```python
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://localhost:5000")
client = MlflowClient()

def load_production_model(model_name="RandomForestClassifier"):
    # Get production model
    models = client.get_latest_versions(model_name, stages=["Production"])

    if not models:
        print("No production model found")
        return None

    model_version = models[0]
    model_uri = f"models:/{model_name}/{model_version.version}"

    # Load model
    model = mlflow.sklearn.load_model(model_uri)

    print(f"Loaded model: {model_name} v{model_version.version}")
    return model

def predict(model, features):
    prediction = model.predict([features])
    return prediction[0]

if __name__ == "__main__":
    # Load production model
    model = load_production_model()

    if model:
        # Make prediction
        sample_features = [0.5] * 20
        result = predict(model, sample_features)
        print(f"Prediction: {result}")
```

### Create Model Comparison Script
```python
def compare_models(model_name, metric="accuracy"):
    versions = client.search_model_versions(f"name='{model_name}'")

    comparison = []
    for v in versions:
        run = client.get_run(v.run_id)
        comparison.append({
            'version': v.version,
            'stage': v.current_stage,
            metric: run.data.metrics.get(metric, 0)
        })

    import pandas as pd
    df = pd.DataFrame(comparison).sort_values(metric, ascending=False)
    print(df)
    return df
```

**Validation**: Run `python serve_model.py` to load and use production model.

## Summary

You've built a complete model artifact management system featuring:
- **MLflow tracking** for experiments, parameters, and metrics logging
- **Model registry** with staging/production lifecycle management
- **DVC pipelines** for reproducible data and model versioning
- **Remote storage** for large artifacts with version control
- **CI/CD integration** automating training, registration, and deployment
- **Model comparison** tools for evaluating different versions

This infrastructure ensures full reproducibility, enables rollback to previous versions, and provides audit trails for all models and datasets in production.
