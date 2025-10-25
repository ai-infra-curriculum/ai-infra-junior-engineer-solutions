# Project 03: ML Pipeline with Experiment Tracking - Solution Guide

## Overview

This solution implements an end-to-end ML pipeline with automated experiment tracking using MLflow, data versioning with DVC, and workflow orchestration using Apache Airflow.

## Architecture

```
Data Sources → Airflow DAG → [Ingest → Validate → Preprocess → Train → Evaluate → Register]
                                ↓          ↓          ↓         ↓        ↓           ↓
                              DVC      GE Checks    DVC     MLflow   MLflow    MLflow Registry
```

## Key Components

### 1. Airflow DAG (`dags/ml_pipeline_dag.py`)

**Pipeline Tasks:**
1. **Data Ingestion**: Fetch data from source
2. **Data Validation**: Great Expectations checks
3. **Preprocessing**: Clean and transform data
4. **Feature Engineering**: Create features
5. **Model Training**: Train with MLflow tracking
6. **Model Evaluation**: Compute metrics
7. **Model Registration**: Register in MLflow

**Implementation:**
```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'ml_training_pipeline',
    default_args=default_args,
    description='End-to-end ML training pipeline',
    schedule_interval='0 2 * * *',  # Daily at 2 AM
    catchup=False,
    tags=['ml', 'training'],
)

# Task definitions
ingest_task = PythonOperator(
    task_id='ingest_data',
    python_callable=ingest_data,
    dag=dag,
)

validate_task = PythonOperator(
    task_id='validate_data',
    python_callable=validate_data,
    dag=dag,
)

preprocess_task = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data,
    dag=dag,
)

train_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag,
)

evaluate_task = PythonOperator(
    task_id='evaluate_model',
    python_callable=evaluate_model,
    dag=dag,
)

register_task = PythonOperator(
    task_id='register_model',
    python_callable=register_model,
    dag=dag,
)

# Task dependencies
ingest_task >> validate_task >> preprocess_task >> train_task >> evaluate_task >> register_task
```

### 2. MLflow Integration (`src/training.py`)

**Experiment Tracking:**
```python
import mlflow
import mlflow.pytorch

def train_model(data_path, params):
    # Start MLflow run
    with mlflow.start_run(run_name=f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Log parameters
        mlflow.log_params(params)

        # Log git commit
        mlflow.log_param("git_commit", get_git_commit())

        # Train model
        model = Model(**params)
        for epoch in range(params['epochs']):
            train_loss = model.train_epoch(data)
            val_loss = model.validate(val_data)

            # Log metrics
            mlflow.log_metrics({
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, step=epoch)

        # Log model
        mlflow.pytorch.log_model(model, "model")

        # Log artifacts
        mlflow.log_artifact("plots/loss_curve.png")
        mlflow.log_artifact("metrics/confusion_matrix.png")

        # Register model
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
        mlflow.register_model(model_uri, "ImageClassifier")
```

**Key MLflow Features:**
- Parameter logging (hyperparameters, config)
- Metric logging (accuracy, loss, F1)
- Artifact logging (models, plots, data)
- Model registry (versioning, staging, production)
- Run comparison (UI and API)

### 3. DVC Integration

**Data Versioning:**
```bash
# Initialize DVC
dvc init

# Track dataset
dvc add data/raw/dataset.csv
git add data/raw/dataset.csv.dvc .gitignore
git commit -m "Track dataset v1.0"

# Configure remote storage
dvc remote add -d storage s3://my-bucket/dvc-cache
dvc push

# Later: retrieve specific version
git checkout <commit>
dvc pull
```

**Benefits:**
- Version large datasets efficiently
- Git-like workflow for data
- Track data lineage
- Collaborate on datasets
- Remote storage support (S3, GCS, Azure)

### 4. Data Validation (`src/validation.py`)

**Great Expectations:**
```python
import great_expectations as ge

def validate_data(data_path):
    # Load data
    df = ge.read_csv(data_path)

    # Define expectations
    df.expect_column_values_to_not_be_null("image_path")
    df.expect_column_values_to_be_in_set("label", valid_labels)
    df.expect_column_values_to_be_between("confidence", 0, 1)
    df.expect_table_row_count_to_be_between(min_value=1000, max_value=100000)

    # Validate
    result = df.validate()

    if not result['success']:
        raise ValueError(f"Data validation failed: {result}")

    return result
```

## Docker Compose Stack

```yaml
version: '3.8'

services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: mlflow
      POSTGRES_USER: mlflow
      POSTGRES_PASSWORD: mlflow
    volumes:
      - postgres_data:/var/lib/postgresql/data

  minio:
    image: minio/minio:latest
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    volumes:
      - minio_data:/data
    ports:
      - "9000:9000"
      - "9001:9001"

  mlflow:
    image: ghcr.io/mlflow/mlflow:latest
    environment:
      MLFLOW_BACKEND_STORE_URI: postgresql://mlflow:mlflow@postgres:5432/mlflow
      MLFLOW_S3_ENDPOINT_URL: http://minio:9000
      AWS_ACCESS_KEY_ID: minioadmin
      AWS_SECRET_ACCESS_KEY: minioadmin
    ports:
      - "5000:5000"
    depends_on:
      - postgres
      - minio
    command: >
      mlflow server
      --backend-store-uri postgresql://mlflow:mlflow@postgres:5432/mlflow
      --default-artifact-root s3://mlflow/
      --host 0.0.0.0

  airflow-webserver:
    image: apache/airflow:2.7.0
    environment:
      AIRFLOW__CORE__EXECUTOR: LocalExecutor
      AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: postgresql+psycopg2://mlflow:mlflow@postgres:5432/airflow
      AIRFLOW__CORE__LOAD_EXAMPLES: 'false'
    ports:
      - "8080:8080"
    depends_on:
      - postgres
    volumes:
      - ./dags:/opt/airflow/dags
      - ./logs:/opt/airflow/logs
      - ./plugins:/opt/airflow/plugins

volumes:
  postgres_data:
  minio_data:
```

## Deployment Instructions

### Local Setup

```bash
# Start services
docker-compose up -d

# Wait for services to be ready
sleep 30

# Verify services
curl http://localhost:5000/health  # MLflow
curl http://localhost:8080/health  # Airflow
curl http://localhost:9001         # MinIO

# Access UIs
# MLflow: http://localhost:5000
# Airflow: http://localhost:8080 (admin/admin)
# MinIO: http://localhost:9001 (minioadmin/minioadmin)
```

### Run Pipeline

```bash
# Trigger DAG via CLI
airflow dags trigger ml_training_pipeline

# Or via UI
# Navigate to http://localhost:8080
# Click on DAG → Trigger

# Monitor execution
airflow dags list-runs -d ml_training_pipeline

# Check logs
airflow tasks logs ml_training_pipeline ingest_data <run_id>
```

### MLflow Usage

```bash
# List experiments
mlflow experiments list

# List runs
mlflow runs list --experiment-id 0

# Compare runs
mlflow ui  # Then navigate to UI

# Serve model
mlflow models serve -m models:/ImageClassifier/Production -p 5001
```

## Key Features

### 1. Experiment Tracking

Track all experiments automatically:
- Hyperparameters
- Metrics (train/val accuracy, loss)
- Model artifacts
- Code version (git commit)
- Environment (dependencies)
- Execution time
- Hardware (CPU/GPU)

### 2. Model Registry

Manage model lifecycle:
- **None**: New models
- **Staging**: Models being tested
- **Production**: Deployed models
- **Archived**: Old models

### 3. Data Lineage

Track data through pipeline:
- Source data version
- Transformations applied
- Feature engineering steps
- Train/val/test splits
- Model trained on which data

### 4. Reproducibility

Reproduce any experiment:
1. Get git commit
2. Get data version (DVC)
3. Get hyperparameters (MLflow)
4. Recreate environment
5. Re-run training

## Best Practices

### Experiment Organization

```python
# Use nested runs
with mlflow.start_run(run_name="hp_tuning"):
    for lr in [0.001, 0.01, 0.1]:
        with mlflow.start_run(run_name=f"lr_{lr}", nested=True):
            model = train(lr=lr)
            mlflow.log_metric("accuracy", accuracy)

# Tag runs
mlflow.set_tag("model_type", "resnet50")
mlflow.set_tag("purpose", "production")
mlflow.set_tag("dataset_version", "v2.1")
```

### Parameter Logging

Log everything needed to reproduce:
```python
mlflow.log_params({
    # Model
    "model_type": "resnet50",
    "num_layers": 50,

    # Training
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 100,
    "optimizer": "adam",

    # Data
    "dataset_version": "v2.1",
    "train_size": 50000,
    "val_size": 10000,

    # Environment
    "git_commit": get_git_commit(),
    "python_version": sys.version,
    "cuda_version": torch.version.cuda,
})
```

### Metric Logging

Log metrics at appropriate frequency:
```python
# Per epoch
for epoch in range(epochs):
    mlflow.log_metrics({
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
    }, step=epoch)

# Final metrics
mlflow.log_metrics({
    "final_train_accuracy": final_train_acc,
    "final_val_accuracy": final_val_acc,
    "best_val_accuracy": best_val_acc,
    "test_accuracy": test_acc,
})
```

## Troubleshooting

### MLflow Connection Issues
```bash
# Check MLflow server
curl http://localhost:5000/api/2.0/mlflow/experiments/list

# Set tracking URI
export MLFLOW_TRACKING_URI=http://localhost:5000
```

### Airflow DAG Not Appearing
```bash
# Check DAG file for syntax errors
python dags/ml_pipeline_dag.py

# Refresh Airflow
airflow dags list

# Check logs
tail -f logs/scheduler/latest/*.log
```

### DVC Push Failing
```bash
# Check remote configuration
dvc remote list
dvc remote modify storage access_key_id YOUR_KEY

# Verify credentials
dvc push --verbose
```

## Testing Checklist

- [ ] MLflow tracks experiments
- [ ] Parameters logged correctly
- [ ] Metrics logged correctly
- [ ] Artifacts uploaded
- [ ] Model registered
- [ ] DVC tracks data
- [ ] Data validation passes
- [ ] Airflow DAG runs successfully
- [ ] All tasks complete
- [ ] Logs are accessible

## Conclusion

This ML pipeline provides:
- **Automated training** with Airflow
- **Experiment tracking** with MLflow
- **Data versioning** with DVC
- **Data validation** with Great Expectations
- **Model registry** for lifecycle management
- **Complete reproducibility**

The pipeline demonstrates production MLOps practices and can be extended with additional features like model monitoring, A/B testing, and automated retraining.
