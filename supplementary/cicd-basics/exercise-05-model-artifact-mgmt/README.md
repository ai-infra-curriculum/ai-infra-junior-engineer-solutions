# Exercise 05: Model Versioning & Artifact Management

## Overview

Implement comprehensive model versioning and artifact management for ML systems using MLflow, DVC, and cloud storage. This exercise teaches how to track experiments, version models, manage artifacts, and integrate with CI/CD pipelines.

## Learning Objectives

- Implement model versioning strategies
- Set up MLflow for experiment tracking
- Use DVC for data and model versioning
- Create model registries
- Track model lineage and metadata
- Integrate artifact management with CI/CD
- Implement model promotion workflows
- Set up artifact storage backends

## Prerequisites

- Completed Exercise 03 (Docker CI/CD)
- Completed Exercise 04 (Kubernetes Deployment)
- Python 3.8+
- Docker installed
- Cloud storage account (AWS S3, GCS, or Azure Blob)
- Basic understanding of ML workflows

## Project Structure

```
exercise-05-model-artifact-mgmt/
├── mlflow/
│   ├── MLproject                     # MLflow project definition
│   ├── conda.yaml                    # Conda environment
│   ├── train.py                      # Training script with MLflow
│   ├── register_model.py             # Model registration script
│   ├── promote_model.py              # Model promotion workflow
│   └── docker-compose.yml            # MLflow server setup
├── dvc/
│   ├── .dvc/                         # DVC configuration
│   ├── .dvcignore                    # DVC ignore patterns
│   ├── data.dvc                      # Data version tracking
│   ├── models.dvc                    # Model version tracking
│   └── setup-dvc.sh                  # DVC initialization script
├── scripts/
│   ├── train-and-log.sh              # Training with logging
│   ├── promote-model.sh              # Model promotion
│   ├── download-model.sh             # Model download
│   └── compare-models.sh             # Model comparison
├── .github/
│   └── workflows/
│       ├── train-and-register.yml    # Train and register model
│       ├── model-promotion.yml       # Promote model workflow
│       └── artifact-sync.yml         # Sync artifacts
├── docs/
│   ├── VERSIONING_STRATEGY.md        # Versioning documentation
│   ├── MLFLOW_GUIDE.md               # MLflow usage guide
│   └── DVC_GUIDE.md                  # DVC usage guide
└── README.md
```

## Quick Start

### 1. Set Up MLflow

```bash
# Start MLflow server with Docker Compose
cd mlflow
docker-compose up -d

# Access MLflow UI
open http://localhost:5000
```

### 2. Set Up DVC

```bash
# Initialize DVC
cd dvc
dvc init

# Configure remote storage
dvc remote add -d myremote s3://my-bucket/dvc-storage
dvc remote modify myremote region us-west-2

# Track data
dvc add ../data/dataset.csv
git add data/dataset.csv.dvc .gitignore
git commit -m "Add dataset to DVC"
```

### 3. Train and Log Model

```bash
# Train model with MLflow tracking
python mlflow/train.py \
  --experiment-name my-experiment \
  --data-path data/dataset.csv \
  --model-type random_forest

# View in MLflow UI
open http://localhost:5000
```

## Model Versioning Strategies

### Semantic Versioning for Models

```
v{MAJOR}.{MINOR}.{PATCH}

MAJOR: Breaking changes (new features, API changes)
MINOR: New functionality (backwards compatible)
PATCH: Bug fixes, minor improvements
```

Examples:
- `v1.0.0` - Initial production model
- `v1.1.0` - Added new features to model
- `v1.1.1` - Fixed prediction bug
- `v2.0.0` - Complete model retraining with new architecture

### Model Stages

Models progress through stages:

1. **Development** - Experimental models
2. **Staging** - Models ready for testing
3. **Production** - Models serving live traffic
4. **Archived** - Deprecated models

## MLflow Integration

### Experiment Tracking

```python
import mlflow
import mlflow.sklearn

# Set experiment
mlflow.set_experiment("model-training")

# Start run
with mlflow.start_run(run_name="random_forest_v1"):
    # Log parameters
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)

    # Train model
    model = train_model(X_train, y_train)

    # Log metrics
    mlflow.log_metric("accuracy", 0.95)
    mlflow.log_metric("f1_score", 0.93)

    # Log model
    mlflow.sklearn.log_model(
        model,
        "model",
        registered_model_name="RandomForestClassifier"
    )

    # Log artifacts
    mlflow.log_artifact("feature_importance.png")
```

### Model Registry

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Register model
result = client.create_registered_model(
    name="RandomForestClassifier",
    tags={"team": "ml-platform"},
    description="Production classifier model"
)

# Create model version
model_version = client.create_model_version(
    name="RandomForestClassifier",
    source="runs:/abc123/model",
    run_id="abc123"
)

# Transition to staging
client.transition_model_version_stage(
    name="RandomForestClassifier",
    version=1,
    stage="Staging",
    archive_existing_versions=False
)

# Add version tags
client.set_model_version_tag(
    name="RandomForestClassifier",
    version=1,
    key="validation_status",
    value="passed"
)
```

### Model Loading

```python
import mlflow.pyfunc

# Load latest production model
model_name = "RandomForestClassifier"
stage = "Production"

model = mlflow.pyfunc.load_model(
    model_uri=f"models:/{model_name}/{stage}"
)

# Make predictions
predictions = model.predict(data)

# Load specific version
model_version = 5
model = mlflow.pyfunc.load_model(
    model_uri=f"models:/{model_name}/{model_version}"
)
```

## DVC Integration

### Initialize DVC

```bash
# Initialize DVC in project
dvc init

# Configure remote storage (S3)
dvc remote add -d myremote s3://my-bucket/dvc-storage
dvc remote modify myremote region us-west-2

# Configure remote storage (GCS)
dvc remote add -d myremote gs://my-bucket/dvc-storage

# Configure remote storage (Azure)
dvc remote add -d myremote azure://my-container/dvc-storage
```

### Track Data

```bash
# Add dataset to DVC
dvc add data/raw/dataset.csv
git add data/raw/dataset.csv.dvc data/raw/.gitignore
git commit -m "Track dataset with DVC"

# Push to remote
dvc push

# Pull from remote
dvc pull
```

### Version Models

```bash
# Add trained model to DVC
dvc add models/random_forest_v1.pkl
git add models/random_forest_v1.pkl.dvc .gitignore
git commit -m "Add model v1"
git tag -a v1.0.0 -m "Model version 1.0.0"

# Push model to remote
dvc push

# Checkout specific version
git checkout v1.0.0
dvc pull
```

### DVC Pipelines

```yaml
# dvc.yaml
stages:
  prepare:
    cmd: python scripts/prepare.py
    deps:
      - data/raw/dataset.csv
    outs:
      - data/processed/train.csv
      - data/processed/test.csv

  train:
    cmd: python scripts/train.py
    deps:
      - data/processed/train.csv
      - scripts/train.py
    params:
      - train.n_estimators
      - train.max_depth
    outs:
      - models/model.pkl
    metrics:
      - metrics/train_metrics.json:
          cache: false

  evaluate:
    cmd: python scripts/evaluate.py
    deps:
      - models/model.pkl
      - data/processed/test.csv
    metrics:
      - metrics/eval_metrics.json:
          cache: false
```

Run pipeline:
```bash
# Run entire pipeline
dvc repro

# Run specific stage
dvc repro train

# Show pipeline
dvc dag

# Compare experiments
dvc metrics diff
dvc params diff
```

## Artifact Storage Backends

### AWS S3

```python
# MLflow with S3
import mlflow

mlflow.set_tracking_uri("http://mlflow-server:5000")

# Configure S3 artifact store
mlflow.create_experiment(
    "my-experiment",
    artifact_location="s3://my-bucket/mlflow-artifacts"
)
```

```bash
# Environment variables
export AWS_ACCESS_KEY_ID=your-key
export AWS_SECRET_ACCESS_KEY=your-secret
export MLFLOW_S3_ENDPOINT_URL=https://s3.amazonaws.com
```

### Google Cloud Storage

```python
# MLflow with GCS
mlflow.create_experiment(
    "my-experiment",
    artifact_location="gs://my-bucket/mlflow-artifacts"
)
```

```bash
# Environment variables
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
```

### Azure Blob Storage

```python
# MLflow with Azure
mlflow.create_experiment(
    "my-experiment",
    artifact_location="wasbs://container@account.blob.core.windows.net/mlflow"
)
```

```bash
# Environment variables
export AZURE_STORAGE_CONNECTION_STRING="..."
export AZURE_STORAGE_ACCESS_KEY="..."
```

## Model Promotion Workflow

### Promotion Strategy

```
Development → Staging → Production
     ↓            ↓           ↓
   Testing    Validation   Monitoring
```

### Automated Promotion

```python
# promote_model.py
from mlflow.tracking import MlflowClient
import mlflow

def promote_model(
    model_name: str,
    version: int,
    from_stage: str,
    to_stage: str,
    validation_metrics: dict
):
    """Promote model to next stage after validation."""
    client = MlflowClient()

    # Validate metrics meet threshold
    if validation_metrics["accuracy"] < 0.85:
        raise ValueError("Model accuracy below threshold")

    # Transition model
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=to_stage,
        archive_existing_versions=True
    )

    # Add promotion metadata
    client.set_model_version_tag(
        name=model_name,
        version=version,
        key="promoted_from",
        value=from_stage
    )

    print(f"✓ Promoted {model_name} v{version}: {from_stage} → {to_stage}")

# Usage
promote_model(
    model_name="RandomForestClassifier",
    version=5,
    from_stage="Staging",
    to_stage="Production",
    validation_metrics={"accuracy": 0.92, "f1_score": 0.90}
)
```

## CI/CD Integration

### Training Pipeline

```yaml
# .github/workflows/train-and-register.yml
name: Train and Register Model

on:
  push:
    branches: [main]
    paths:
      - 'models/**'
      - 'data/**'

jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          pip install mlflow scikit-learn pandas

      - name: Configure MLflow
        env:
          MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        run: |
          mlflow experiments list

      - name: Train model
        env:
          MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}
        run: |
          python mlflow/train.py \
            --experiment-name production-training \
            --model-type random_forest

      - name: Register model
        env:
          MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}
        run: |
          python mlflow/register_model.py \
            --run-id $RUN_ID \
            --model-name RandomForestClassifier
```

### Model Promotion Pipeline

```yaml
# .github/workflows/model-promotion.yml
name: Model Promotion

on:
  workflow_dispatch:
    inputs:
      model_name:
        description: 'Model name'
        required: true
      version:
        description: 'Model version'
        required: true
      target_stage:
        description: 'Target stage'
        required: true
        type: choice
        options:
          - Staging
          - Production

jobs:
  promote:
    runs-on: ubuntu-latest
    environment: ${{ github.event.inputs.target_stage }}

    steps:
      - uses: actions/checkout@v4

      - name: Validate model
        env:
          MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}
        run: |
          python scripts/validate_model.py \
            --model-name ${{ github.event.inputs.model_name }} \
            --version ${{ github.event.inputs.version }}

      - name: Promote model
        env:
          MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}
        run: |
          python mlflow/promote_model.py \
            --model-name ${{ github.event.inputs.model_name }} \
            --version ${{ github.event.inputs.version }} \
            --stage ${{ github.event.inputs.target_stage }}

      - name: Create release tag
        if: github.event.inputs.target_stage == 'Production'
        run: |
          git tag -a "model-${{ github.event.inputs.model_name }}-v${{ github.event.inputs.version }}" \
            -m "Promote model to production"
          git push origin --tags
```

## Model Lineage & Metadata

### Track Model Lineage

```python
import mlflow

with mlflow.start_run() as run:
    # Log parent run ID for lineage
    mlflow.set_tag("parent_run_id", parent_run_id)

    # Log data version
    mlflow.set_tag("data_version", "v1.2.3")
    mlflow.set_tag("data_hash", data_hash)

    # Log code version
    mlflow.set_tag("git_commit", git_commit_sha)
    mlflow.set_tag("git_branch", git_branch)

    # Log model info
    mlflow.set_tag("model_type", "random_forest")
    mlflow.set_tag("framework", "scikit-learn")
    mlflow.set_tag("framework_version", sklearn.__version__)

    # Log training metadata
    mlflow.set_tag("training_duration", training_time)
    mlflow.set_tag("training_date", datetime.now().isoformat())
    mlflow.set_tag("trained_by", "ci-pipeline")

    # Log environment
    mlflow.set_tag("python_version", sys.version)
    mlflow.log_param("random_seed", 42)
```

### Query Model Lineage

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Get model version details
version = client.get_model_version(
    name="RandomForestClassifier",
    version=5
)

# Get run details
run = client.get_run(version.run_id)

# Extract lineage
lineage = {
    "model_name": version.name,
    "version": version.version,
    "stage": version.current_stage,
    "run_id": version.run_id,
    "data_version": run.data.tags.get("data_version"),
    "git_commit": run.data.tags.get("git_commit"),
    "training_date": run.data.tags.get("training_date"),
    "parent_run": run.data.tags.get("parent_run_id"),
}

print(lineage)
```

## Best Practices

### 1. Version Everything

✅ Version data with DVC
✅ Version models with MLflow
✅ Version code with Git
✅ Version dependencies (requirements.txt, conda.yaml)
✅ Version configurations

### 2. Use Semantic Versioning

✅ Follow semantic versioning for models
✅ Tag releases in Git
✅ Document breaking changes
✅ Maintain changelog

### 3. Track Experiments Thoroughly

✅ Log all hyperparameters
✅ Log all metrics
✅ Log artifacts (plots, confusion matrices)
✅ Log system info (Python version, dependencies)
✅ Log data statistics

### 4. Implement Model Governance

✅ Require approval for production promotion
✅ Run validation tests before promotion
✅ Document model limitations
✅ Track model performance in production

### 5. Automate Workflows

✅ Automate training pipelines
✅ Automate model registration
✅ Automate promotion workflows
✅ Automate artifact sync

## Common Commands

### MLflow

```bash
# Start MLflow server
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root s3://my-bucket/mlflow \
  --host 0.0.0.0 \
  --port 5000

# Run MLflow project
mlflow run . -P alpha=0.5

# List experiments
mlflow experiments list

# Search runs
mlflow runs list --experiment-id 1

# Download artifacts
mlflow artifacts download --run-id abc123 --dst-path ./artifacts
```

### DVC

```bash
# Initialize DVC
dvc init

# Track file
dvc add data/large_file.csv

# Push to remote
dvc push

# Pull from remote
dvc pull

# Checkout version
git checkout v1.0.0
dvc checkout

# Run pipeline
dvc repro

# Show differences
dvc diff
dvc metrics diff
```

## Troubleshooting

### MLflow Connection Issues

```bash
# Check MLflow server is running
curl http://localhost:5000/health

# Verify environment variables
echo $MLFLOW_TRACKING_URI

# Test connection
python -c "import mlflow; print(mlflow.get_tracking_uri())"
```

### DVC Remote Issues

```bash
# Test remote connection
dvc remote list
dvc push --remote myremote -v

# Check credentials
aws s3 ls s3://my-bucket/dvc-storage
```

### Model Loading Errors

```python
# Debug model loading
import mlflow
mlflow.set_tracking_uri("http://localhost:5000")

# List registered models
from mlflow.tracking import MlflowClient
client = MlflowClient()
for rm in client.list_registered_models():
    print(f"name: {rm.name}")
    for mv in rm.latest_versions:
        print(f"  version {mv.version}, stage: {mv.current_stage}")
```

## Resources

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [DVC Documentation](https://dvc.org/doc)
- [Semantic Versioning](https://semver.org/)
- [MLOps Best Practices](https://ml-ops.org/)

## Next Steps

After completing this exercise:

1. ✅ Understand model versioning strategies
2. ✅ Track experiments with MLflow
3. ✅ Version data and models with DVC
4. ✅ Implement model registries
5. ✅ Automate model promotion
6. ✅ Integrate with CI/CD pipelines

**Move on to**: Exercise 06 - End-to-End ML Pipeline
