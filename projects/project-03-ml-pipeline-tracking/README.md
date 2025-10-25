# Project 03: ML Pipeline with Experiment Tracking - Solution

Complete ML pipeline with Apache Airflow orchestration, MLflow experiment tracking, and DVC data versioning.

## Quick Start

```bash
# Start all services
docker-compose up -d

# Wait for services to be ready
sleep 30

# Access UIs
# MLflow: http://localhost:5000
# Airflow: http://localhost:8080 (admin/admin)
# MinIO: http://localhost:9001 (minioadmin/minioadmin)

# Trigger pipeline
docker-compose exec airflow-webserver airflow dags trigger ml_training_pipeline
```

## Features

- **Workflow Orchestration**: Airflow DAGs for pipeline automation
- **Experiment Tracking**: MLflow logs parameters, metrics, artifacts
- **Data Versioning**: DVC tracks datasets like Git
- **Data Validation**: Great Expectations ensures data quality
- **Model Registry**: MLflow manages model lifecycle
- **Artifact Storage**: MinIO (S3-compatible) for models and data
- **Reproducibility**: Complete experiment reproduction

## Architecture

```
Data Sources → Airflow DAG → [Ingest → Validate → Train → Evaluate → Register]
                                ↓         ↓         ↓        ↓          ↓
                              DVC    Great Exp   MLflow  MLflow   ML Registry
```

## Pipeline Stages

1. **Data Ingestion**: Fetch data from source
2. **Data Validation**: Quality checks with Great Expectations
3. **Preprocessing**: Clean and transform data
4. **Training**: Train model with MLflow tracking
5. **Evaluation**: Compute metrics and compare models
6. **Registration**: Register best model in MLflow

## Project Structure

```
project-03-ml-pipeline-tracking/
├── src/
│   ├── data_ingestion.py    # Data loading
│   ├── preprocessing.py     # Data transforms
│   ├── training.py          # Model training
│   └── evaluation.py        # Model evaluation
├── dags/
│   └── ml_pipeline_dag.py   # Airflow DAG
├── docker-compose.yml       # Full stack
├── README.md               # This file
└── SOLUTION_GUIDE.md       # Detailed guide
```

## MLflow Usage

```python
import mlflow

# Log experiment
with mlflow.start_run():
    mlflow.log_params({"lr": 0.001, "epochs": 100})
    mlflow.log_metrics({"accuracy": 0.95, "loss": 0.05})
    mlflow.pytorch.log_model(model, "model")

# Register model
mlflow.register_model("runs:/RUN_ID/model", "ImageClassifier")

# Load model
model = mlflow.pytorch.load_model("models:/ImageClassifier/Production")
```

## DVC Usage

```bash
# Track dataset
dvc add data/train.csv
git add data/train.csv.dvc
git commit -m "Add training data v1"

# Configure remote
dvc remote add -d storage s3://bucket/dvc-cache
dvc push

# Retrieve data
dvc pull
```

## Services

| Service | Port | Purpose |
|---------|------|---------|
| MLflow | 5000 | Experiment tracking UI |
| Airflow | 8080 | Pipeline orchestration UI |
| MinIO | 9000/9001 | Artifact storage |
| PostgreSQL | 5432 | Metadata database |

## Testing

```bash
# Test pipeline components
pytest tests/test_pipeline.py -v

# Test data validation
pytest tests/test_validation.py -v

# End-to-end test
pytest tests/test_e2e.py -v
```

## Key Features

### Experiment Tracking
- All hyperparameters logged
- Metrics tracked over time
- Models versioned automatically
- Artifacts stored (plots, models, data)

### Data Versioning
- Git-like workflow for data
- Efficient storage of large files
- Remote storage support (S3, GCS, Azure)
- Complete data lineage

### Reproducibility
Every experiment can be reproduced with:
- Git commit (code)
- DVC version (data)
- MLflow run (parameters)

## Documentation

See [SOLUTION_GUIDE.md](SOLUTION_GUIDE.md) for:
- Architecture details
- Component explanations
- Best practices
- Troubleshooting
- Advanced usage

## Requirements

- Docker and Docker Compose
- Python 3.11+
- 8GB RAM minimum
- 20GB disk space

## License

Educational use only - AI Infrastructure Curriculum
