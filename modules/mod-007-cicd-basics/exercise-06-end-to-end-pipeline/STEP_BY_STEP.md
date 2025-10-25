# Step-by-Step Guide: End-to-End ML Pipeline

## Overview
Build a complete automated ML pipeline integrating training, testing, containerization, versioning, and deployment with comprehensive CI/CD orchestration.

## Phase 1: Pipeline Architecture Setup (15 minutes)

### Create Project Structure
```bash
# Create comprehensive directory structure
mkdir -p ml-pipeline/{src,tests,data,models,k8s,scripts,.github/workflows}
cd ml-pipeline

# Initialize git
git init
git config user.name "Your Name"
git config user.email "your.email@example.com"

# Create virtual environment
python3 -m venv venv
source venv/bin/activate
```

### Install All Dependencies
Create `requirements.txt`:
```
# ML Libraries
scikit-learn==1.3.2
pandas==2.1.3
numpy==1.26.2

# API Framework
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0

# MLOps Tools
mlflow==2.8.1
dvc==3.30.1

# Testing
pytest==7.4.3
pytest-cov==4.1.0
requests==2.31.0

# Monitoring
prometheus-client==0.19.0
```

```bash
pip install -r requirements.txt
```

### Create Configuration
Create `config.yaml`:
```yaml
model:
  type: RandomForestClassifier
  params:
    n_estimators: 100
    max_depth: 10
    random_state: 42

data:
  train_split: 0.8
  validation_split: 0.1
  test_split: 0.1

training:
  batch_size: 32
  epochs: 10

mlflow:
  experiment_name: "end-to-end-pipeline"
  tracking_uri: "http://localhost:5000"

api:
  host: "0.0.0.0"
  port: 8000
  workers: 4
```

**Validation**: Verify all dependencies install successfully.

## Phase 2: Data Pipeline (15 minutes)

### Create Data Preparation Script
Create `src/data_pipeline.py`:
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPipeline:
    def __init__(self, config_path='config.yaml'):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

    def generate_dataset(self, n_samples=10000):
        """Generate synthetic dataset"""
        logger.info(f"Generating {n_samples} samples")
        np.random.seed(42)

        data = {
            'feature1': np.random.randn(n_samples),
            'feature2': np.random.randn(n_samples),
            'feature3': np.random.randn(n_samples),
            'feature4': np.random.randn(n_samples),
        }

        df = pd.DataFrame(data)
        df['target'] = (df['feature1'] + df['feature2'] > 0).astype(int)

        return df

    def split_data(self, df):
        """Split data into train/val/test"""
        train_size = self.config['data']['train_split']
        val_size = self.config['data']['validation_split']

        # First split: train and temp
        train_df, temp_df = train_test_split(df, train_size=train_size, random_state=42)

        # Second split: validation and test
        val_ratio = val_size / (1 - train_size)
        val_df, test_df = train_test_split(temp_df, train_size=val_ratio, random_state=42)

        logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

        return train_df, val_df, test_df

    def preprocess(self, train_df, val_df, test_df):
        """Preprocess and scale features"""
        feature_cols = [c for c in train_df.columns if c != 'target']

        scaler = StandardScaler()
        train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
        val_df[feature_cols] = scaler.transform(val_df[feature_cols])
        test_df[feature_cols] = scaler.transform(test_df[feature_cols])

        return train_df, val_df, test_df, scaler

    def run(self):
        """Execute full data pipeline"""
        df = self.generate_dataset()
        train_df, val_df, test_df = self.split_data(df)
        train_df, val_df, test_df, scaler = self.preprocess(train_df, val_df, test_df)

        # Save datasets
        train_df.to_csv('data/train.csv', index=False)
        val_df.to_csv('data/validation.csv', index=False)
        test_df.to_csv('data/test.csv', index=False)

        logger.info("Data pipeline completed successfully")

if __name__ == "__main__":
    pipeline = DataPipeline()
    pipeline.run()
```

**Validation**: Run `python src/data_pipeline.py` and verify CSV files created.

## Phase 3: Training Pipeline (15 minutes)

### Create Training Script
Create `src/train.py`:
```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, config_path='config.yaml'):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        mlflow.set_tracking_uri(self.config['mlflow']['tracking_uri'])
        mlflow.set_experiment(self.config['mlflow']['experiment_name'])

    def load_data(self):
        """Load preprocessed data"""
        train_df = pd.read_csv('data/train.csv')
        val_df = pd.read_csv('data/validation.csv')

        X_train = train_df.drop('target', axis=1)
        y_train = train_df['target']
        X_val = val_df.drop('target', axis=1)
        y_val = val_df['target']

        return X_train, y_train, X_val, y_val

    def train(self):
        """Train model with MLflow tracking"""
        X_train, y_train, X_val, y_val = self.load_data()

        with mlflow.start_run():
            # Log parameters
            params = self.config['model']['params']
            mlflow.log_params(params)

            # Train model
            logger.info("Training model...")
            model = RandomForestClassifier(**params)
            model.fit(X_train, y_train)

            # Evaluate on training set
            train_pred = model.predict(X_train)
            train_acc = accuracy_score(y_train, train_pred)
            mlflow.log_metric("train_accuracy", train_acc)

            # Evaluate on validation set
            val_pred = model.predict(X_val)
            val_metrics = {
                'val_accuracy': accuracy_score(y_val, val_pred),
                'val_precision': precision_score(y_val, val_pred),
                'val_recall': recall_score(y_val, val_pred),
                'val_f1': f1_score(y_val, val_pred)
            }

            for metric_name, metric_value in val_metrics.items():
                mlflow.log_metric(metric_name, metric_value)
                logger.info(f"{metric_name}: {metric_value:.4f}")

            # Log model
            mlflow.sklearn.log_model(model, "model")

            # Save model locally
            import joblib
            joblib.dump(model, 'models/model.pkl')

            logger.info("Training completed successfully")
            return mlflow.active_run().info.run_id

if __name__ == "__main__":
    trainer = ModelTrainer()
    run_id = trainer.train()
    print(f"MLflow Run ID: {run_id}")
```

**Validation**: Run `python src/train.py` and verify model saved.

## Phase 4: API Development (15 minutes)

### Create FastAPI Application
Create `src/api.py`:
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import logging
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import Response

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
prediction_counter = Counter('predictions_total', 'Total predictions made')
prediction_latency = Histogram('prediction_latency_seconds', 'Prediction latency')

app = FastAPI(title="ML Prediction API", version="1.0.0")

# Load model at startup
model = None

@app.on_event("startup")
async def load_model():
    global model
    try:
        model = joblib.load('models/model.pkl')
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")

class PredictionRequest(BaseModel):
    features: list[float]

class PredictionResponse(BaseModel):
    prediction: int
    probability: float

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type="text/plain")

@app.post("/predict", response_model=PredictionResponse)
@prediction_latency.time()
async def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        features = np.array([request.features])
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0].max()

        prediction_counter.inc()

        return PredictionResponse(
            prediction=int(prediction),
            probability=float(probability)
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Create API Tests
Create `tests/test_api.py`:
```python
from fastapi.testclient import TestClient
from src.api import app
import pytest

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()

def test_prediction():
    response = client.post(
        "/predict",
        json={"features": [0.5, 0.3, 0.2, 0.1]}
    )
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert "probability" in response.json()

def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
```

**Validation**: Run `pytest tests/test_api.py -v`

## Phase 5: Containerization (15 minutes)

### Create Dockerfile
Create `Dockerfile`:
```dockerfile
FROM python:3.11-slim as builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

FROM python:3.11-slim

WORKDIR /app

COPY --from=builder /root/.local /root/.local
COPY src/ ./src/
COPY models/ ./models/
COPY config.yaml .

ENV PATH=/root/.local/bin:$PATH

RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Create Docker Compose
Create `docker-compose.yml`:
```yaml
version: '3.8'

services:
  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.8.1
    ports:
      - "5000:5000"
    command: mlflow server --host 0.0.0.0 --port 5000

  api:
    build: .
    ports:
      - "8000:8000"
    depends_on:
      - mlflow
    environment:
      - MLFLOW_TRACKING_URI=http://mlflow:5000
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 10s
      timeout: 5s
      retries: 3
```

**Validation**: Run `docker-compose up --build` and test endpoints.

## Phase 6: Complete CI/CD Pipeline (15 minutes)

### Create End-to-End Workflow
Create `.github/workflows/ml-pipeline.yml`:
```yaml
name: End-to-End ML Pipeline

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  data-preparation:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    - name: Install dependencies
      run: pip install -r requirements.txt
    - name: Run data pipeline
      run: python src/data_pipeline.py
    - name: Upload data artifacts
      uses: actions/upload-artifact@v3
      with:
        name: datasets
        path: data/

  train-model:
    needs: data-preparation
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    - name: Download data
      uses: actions/download-artifact@v3
      with:
        name: datasets
        path: data/
    - name: Install dependencies
      run: pip install -r requirements.txt
    - name: Train model
      run: python src/train.py
    - name: Upload model
      uses: actions/upload-artifact@v3
      with:
        name: model
        path: models/

  test-api:
    needs: train-model
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    - name: Download model
      uses: actions/download-artifact@v3
      with:
        name: model
        path: models/
    - name: Install dependencies
      run: pip install -r requirements.txt
    - name: Run tests
      run: pytest tests/ -v --cov=src

  build-and-push:
    needs: test-api
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
    - uses: actions/checkout@v3
    - name: Download model
      uses: actions/download-artifact@v3
      with:
        name: model
        path: models/
    - name: Build and push Docker image
      uses: docker/build-push-action@v4
      with:
        context: .
        push: true
        tags: ghcr.io/${{ github.repository }}:latest

  deploy:
    needs: build-and-push
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
    - uses: actions/checkout@v3
    - name: Deploy to Kubernetes
      run: |
        kubectl set image deployment/ml-api ml-api=ghcr.io/${{ github.repository }}:latest
        kubectl rollout status deployment/ml-api
```

**Validation**: Push to GitHub and verify entire pipeline executes.

## Summary

You've built a production-ready end-to-end ML pipeline featuring:
- **Data pipeline** with preprocessing, splitting, and versioning
- **Training pipeline** with MLflow experiment tracking and model versioning
- **FastAPI application** with health checks and Prometheus metrics
- **Comprehensive testing** covering API endpoints and model functionality
- **Docker containerization** with multi-stage builds and health checks
- **Complete CI/CD** automating data prep, training, testing, building, and deployment

This integrated pipeline enables fully automated ML workflows from data ingestion through production deployment with complete observability and reproducibility.
