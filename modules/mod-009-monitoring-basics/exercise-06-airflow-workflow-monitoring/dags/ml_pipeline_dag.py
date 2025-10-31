"""
ML Pipeline DAG

A complete machine learning pipeline that:
1. Downloads training data
2. Validates data quality
3. Preprocesses and cleans data
4. Engineers features
5. Trains model
6. Evaluates model performance
7. Deploys model to production
8. Sends notification

Schedule: Daily at 2 AM
Retries: 2 times with 1-minute delay
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_processing import download_data, validate_data, preprocess_data
from src.model_training import feature_engineering, train_model, evaluate_model
from src.model_deployment import deploy_model


# Default arguments applied to all tasks
default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'email': ['ml-alerts@example.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=1),
    'retry_exponential_backoff': True,
    'max_retry_delay': timedelta(minutes=10),
}


# Define the DAG
with DAG(
    'ml_pipeline_dag',
    default_args=default_args,
    description='End-to-end ML model training and deployment pipeline',
    schedule_interval='0 2 * * *',  # Daily at 2 AM
    start_date=datetime(2024, 1, 1),
    catchup=False,  # Don't backfill
    tags=['ml', 'production', 'training'],
    max_active_runs=1,  # Only one run at a time
) as dag:

    # Task 1: Download dataset
    download_task = PythonOperator(
        task_id='download_data',
        python_callable=download_data,
        doc_md="""
        ### Download Data Task

        Downloads training dataset from cloud storage.

        **Returns:**
        - dataset_size_gb: Size of downloaded dataset
        - num_samples: Number of training samples
        - download_time_seconds: Time taken to download
        """,
    )

    # Task 2: Validate data quality
    validate_task = PythonOperator(
        task_id='validate_data',
        python_callable=validate_data,
        doc_md="""
        ### Validate Data Task

        Validates data quality metrics:
        - Checks for missing values
        - Validates schema
        - Checks for duplicates

        **Fails if:**
        - More than 5% null values
        - Schema mismatch
        - More than 1% duplicates
        """,
    )

    # Task 3: Preprocess data
    preprocess_task = PythonOperator(
        task_id='preprocess_data',
        python_callable=preprocess_data,
        doc_md="""
        ### Preprocess Data Task

        Cleans and preprocesses data:
        - Handles missing values
        - Normalizes numerical features
        - Encodes categorical features
        - Splits into train/validation sets
        """,
    )

    # Task 4: Feature engineering
    feature_task = PythonOperator(
        task_id='feature_engineering',
        python_callable=feature_engineering,
        execution_timeout=timedelta(minutes=30),
        doc_md="""
        ### Feature Engineering Task

        Creates additional features:
        - Aggregations
        - Interactions
        - Temporal features
        - Domain-specific transformations
        """,
    )

    # Task 5: Train model
    train_task = PythonOperator(
        task_id='train_model',
        python_callable=train_model,
        execution_timeout=timedelta(hours=2),
        pool='gpu_pool',  # Use GPU pool if available
        doc_md="""
        ### Train Model Task

        Trains ML model on processed data.

        **Resource requirements:**
        - Pool: gpu_pool
        - Timeout: 2 hours

        **Returns:**
        - accuracy: Validation accuracy
        - model_size_mb: Model size
        - training_time_minutes: Training duration
        """,
    )

    # Task 6: Evaluate model
    evaluate_task = PythonOperator(
        task_id='evaluate_model',
        python_callable=evaluate_model,
        doc_md="""
        ### Evaluate Model Task

        Evaluates model performance on test set.

        **Metrics:**
        - Accuracy
        - Precision/Recall
        - F1 Score
        - AUC-ROC

        **Fails if:**
        - Accuracy < 0.88
        - F1 Score < 0.85
        """,
    )

    # Task 7: Deploy model
    deploy_task = PythonOperator(
        task_id='deploy_model',
        python_callable=deploy_model,
        doc_md="""
        ### Deploy Model Task

        Deploys model to production:
        - Uploads model to model registry
        - Updates serving endpoint
        - Performs health check

        **Returns:**
        - deployment_url: Model API endpoint
        - version: Model version
        """,
    )

    # Task 8: Send success notification
    notify_task = BashOperator(
        task_id='send_notification',
        bash_command="""
        echo "=================================================="
        echo "ML Pipeline completed successfully!"
        echo "Execution date: {{ ds }}"
        echo "DAG run ID: {{ run_id }}"
        echo "=================================================="
        """,
    )

    # Define task dependencies
    # Linear flow with quality gates
    download_task >> validate_task >> preprocess_task
    preprocess_task >> feature_task >> train_task
    train_task >> evaluate_task >> deploy_task >> notify_task


# DAG documentation
dag.doc_md = """
# ML Training Pipeline

## Overview

This DAG orchestrates the complete ML model training and deployment pipeline.

## Schedule

- **Frequency**: Daily at 2 AM UTC
- **Catchup**: Disabled
- **Max Active Runs**: 1

## Pipeline Stages

```
download_data → validate_data → preprocess_data → feature_engineering
                                                          ↓
                    deploy_model ← evaluate_model ← train_model
                          ↓
                  send_notification
```

## SLAs

- Total pipeline: < 3 hours
- Data download: < 15 minutes
- Training: < 2 hours
- Deployment: < 10 minutes

## Error Handling

- **Retries**: 2 attempts with exponential backoff
- **Alerts**: Email sent on task failure
- **Validation**: Quality gates at validation and evaluation stages

## Monitoring

- View metrics in Prometheus
- Dashboard available in Grafana
- Logs available in Airflow UI

## Dependencies

- Python 3.8+
- scikit-learn
- pandas
- numpy

## Contact

For issues or questions, contact: ml-team@example.com
"""
