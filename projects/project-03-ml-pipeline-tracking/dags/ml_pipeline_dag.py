"""Airflow DAG for the end-to-end ML training pipeline.

Pipeline stages (in execution order):
    ingest_data  -> validate_data  -> preprocess_data  -> version_data_dvc
                  -> train_model   -> evaluate_model    -> register_model
                  -> send_success_email

Task callables push their useful outputs to XCom so downstream tasks can
pick them up without re-reading from disk.

The DAG is written to fail fast on data-quality issues (validation has
``retries=0``) but tolerate transient ingestion / training problems with
exponential backoff.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict

from airflow import DAG
from airflow.operators.email import EmailOperator
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

# Allow the DAG to import from the project's ``src`` directory regardless
# of which directory Airflow loads DAGs from.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PIPELINE_CONFIG: Dict[str, Any] = {
    "raw_data_path": "/opt/airflow/data/raw",
    "processed_data_path": "/opt/airflow/data/processed",
    "model_save_path": "/opt/airflow/models",
    "artifacts_path": "/opt/airflow/artifacts",
    "source_csv": "/opt/airflow/data/source/dataset.csv",
    "mlflow_tracking_uri": os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"),
    "experiment_name": "image_classification_pipeline",
    "registered_model_name": "image_classifier",
    "accuracy_threshold": 0.85,
    "class_names": ["cat", "dog", "bird", "fish"],
    "notify_email": os.getenv("ML_NOTIFY_EMAIL", "mlops@example.com"),
}

DEFAULT_ARGS: Dict[str, Any] = {
    "owner": "ml-team",
    "depends_on_past": False,
    "email": [PIPELINE_CONFIG["notify_email"]],
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
    "retry_exponential_backoff": True,
    "max_retry_delay": timedelta(minutes=30),
    "execution_timeout": timedelta(hours=2),
}


# ---------------------------------------------------------------------------
# Task functions
# ---------------------------------------------------------------------------


def ingest_data(**context: Any) -> str:
    """Ingest the source CSV into the raw-data lake."""
    from src.data_ingestion import DataIngestion  # local import = fast DAG load

    logger.info("Starting data ingestion...")
    ingestion = DataIngestion(PIPELINE_CONFIG)
    df = ingestion.ingest_from_csv(PIPELINE_CONFIG["source_csv"])
    output_path = ingestion.save_raw_data(df, "raw_dataset.csv")
    context["task_instance"].xcom_push(key="raw_data_path", value=str(output_path))
    context["task_instance"].xcom_push(key="row_count", value=len(df))
    logger.info("Data ingestion complete: %d rows -> %s", len(df), output_path)
    return f"Ingested {len(df)} rows"


def validate_data(**context: Any) -> str:
    """Run Great Expectations against the freshly-ingested data."""
    import pandas as pd

    from src.data_validation import DataValidator

    raw_data_path = context["task_instance"].xcom_pull(
        task_ids="ingest_data", key="raw_data_path"
    )
    if not raw_data_path:
        raise ValueError("ingest_data did not push raw_data_path; cannot validate.")

    logger.info("Validating %s", raw_data_path)
    df = pd.read_csv(raw_data_path)
    validator = DataValidator()
    suite_name = "data_quality_suite"
    validator.create_expectation_suite(suite_name)
    validation_passed = validator.validate_data(df, suite_name)
    if not validation_passed:
        # Validation failure is a fast-fail signal: training on bad data
        # wastes GPU budget. Don't retry — surface the issue.
        raise ValueError("Data validation failed — check Great Expectations report.")
    context["task_instance"].xcom_push(key="validation_passed", value=True)
    return "Validation passed"


def preprocess_data(**context: Any) -> str:
    """Clean + split the raw dataset into train/val/test."""
    import pandas as pd

    from src.preprocessing import DataPreprocessor

    raw_data_path = context["task_instance"].xcom_pull(
        task_ids="ingest_data", key="raw_data_path"
    )
    if not raw_data_path:
        raise ValueError("ingest_data did not push raw_data_path; cannot preprocess.")

    df = pd.read_csv(raw_data_path)
    preprocessor = DataPreprocessor(PIPELINE_CONFIG)
    train, val, test = preprocessor.run_pipeline(df, label_column="label")
    context["task_instance"].xcom_push(
        key="split_sizes",
        value={"train": len(train), "val": len(val), "test": len(test)},
    )
    return (
        f"Preprocessing complete: train={len(train)} val={len(val)} test={len(test)}"
    )


def version_data_dvc(**context: Any) -> str:
    """Track the processed dataset with DVC + Git."""
    commands = [
        ["dvc", "add", "data/processed"],
        ["dvc", "push"],
        ["git", "add", "data/processed.dvc", ".gitignore"],
        [
            "git",
            "commit",
            "-m",
            f"Pipeline data version {datetime.utcnow().isoformat()}",
        ],
    ]
    cwd = str(_PROJECT_ROOT)
    for cmd in commands:
        logger.info("Running: %s", " ".join(cmd))
        try:
            subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            # ``git commit`` with no changes returns non-zero. Treat that
            # case as a soft success because it means the dataset is
            # unchanged from the previous run.
            if cmd[0] == "git" and cmd[1] == "commit" and "nothing to commit" in (
                exc.stdout or ""
            ):
                logger.info("No data changes since previous run; skipping commit.")
                break
            logger.error("Command failed: %s\nstderr: %s", cmd, exc.stderr)
            raise
    return "DVC versioning complete"


def train_model(**context: Any) -> str:
    """Train a fresh model and log everything to MLflow."""
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader

    from src.training import ImageDataset, MLflowTracker, ModelTrainer

    processed = Path(PIPELINE_CONFIG["processed_data_path"])
    train_df = pd.read_csv(processed / "train.csv")
    val_df = pd.read_csv(processed / "val.csv")

    train_loader = DataLoader(
        ImageDataset(train_df),
        batch_size=32,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        ImageDataset(val_df),
        batch_size=64,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    tracker = MLflowTracker(
        tracking_uri=PIPELINE_CONFIG["mlflow_tracking_uri"],
        experiment_name=PIPELINE_CONFIG["experiment_name"],
    )
    params = {
        "model_name": "resnet18",
        "num_epochs": 10,
        "batch_size": 32,
        "learning_rate": 0.001,
        "optimizer": "adam",
        "lr_step_size": 5,
        "lr_gamma": 0.1,
        "early_stopping_patience": 3,
    }
    trainer = ModelTrainer(PIPELINE_CONFIG, tracker)
    _, best_val_acc, run_id = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=len(PIPELINE_CONFIG["class_names"]),
        params=params,
    )
    context["task_instance"].xcom_push(key="best_val_acc", value=best_val_acc)
    context["task_instance"].xcom_push(key="mlflow_run_id", value=run_id)
    return f"Training complete: best_val_acc={best_val_acc:.4f}"


def evaluate_model(**context: Any) -> str:
    """Evaluate the latest model against the held-out test set."""
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader

    from src.evaluation import ModelEvaluator
    from src.training import ImageDataset

    processed = Path(PIPELINE_CONFIG["processed_data_path"])
    test_df = pd.read_csv(processed / "test.csv")
    test_loader = DataLoader(
        ImageDataset(test_df), batch_size=64, shuffle=False
    )

    model_path = Path(PIPELINE_CONFIG["model_save_path"]) / "best_model.pth"
    model = torch.load(model_path, map_location="cpu")
    evaluator = ModelEvaluator(PIPELINE_CONFIG, PIPELINE_CONFIG["class_names"])
    metrics = evaluator.evaluate(model, test_loader)
    context["task_instance"].xcom_push(key="test_metrics", value=metrics)
    return f"Evaluation complete: test_accuracy={metrics['test_accuracy']:.4f}"


def register_model(**context: Any) -> str:
    """Register the model with MLflow if it clears the accuracy bar."""
    import mlflow
    from mlflow.tracking import MlflowClient

    test_metrics = context["task_instance"].xcom_pull(
        task_ids="evaluate_model", key="test_metrics"
    )
    if not test_metrics:
        raise ValueError("evaluate_model did not push test_metrics.")

    accuracy = test_metrics.get("test_accuracy", 0.0)
    threshold = PIPELINE_CONFIG["accuracy_threshold"]
    if accuracy < threshold:
        logger.info(
            "Skipping registration: accuracy %.4f below threshold %.4f",
            accuracy,
            threshold,
        )
        return f"Not registered — accuracy {accuracy:.4f} < {threshold:.4f}"

    run_id = context["task_instance"].xcom_pull(
        task_ids="train_model", key="mlflow_run_id"
    )
    if not run_id:
        # Fallback: look up the latest run in the experiment.
        mlflow.set_tracking_uri(PIPELINE_CONFIG["mlflow_tracking_uri"])
        experiment = mlflow.get_experiment_by_name(PIPELINE_CONFIG["experiment_name"])
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=1,
        )
        run_id = runs.iloc[0]["run_id"]

    model_uri = f"runs:/{run_id}/model"
    registration = mlflow.register_model(
        model_uri=model_uri, name=PIPELINE_CONFIG["registered_model_name"]
    )

    client = MlflowClient(tracking_uri=PIPELINE_CONFIG["mlflow_tracking_uri"])
    client.transition_model_version_stage(
        name=PIPELINE_CONFIG["registered_model_name"],
        version=registration.version,
        stage="Staging",
        archive_existing_versions=False,
    )
    logger.info(
        "Registered %s version %s -> Staging",
        PIPELINE_CONFIG["registered_model_name"],
        registration.version,
    )
    return f"Registered version {registration.version} (Staging)"


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------

dag = DAG(
    dag_id="ml_training_pipeline",
    default_args=DEFAULT_ARGS,
    description="End-to-end ML training pipeline with MLflow tracking",
    schedule_interval="0 0 * * 0",  # Sunday 00:00 UTC.
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,
    tags=["ml", "training", "production"],
    doc_md=__doc__,
)


with dag:
    task_ingest = PythonOperator(
        task_id="ingest_data",
        python_callable=ingest_data,
    )

    task_validate = PythonOperator(
        task_id="validate_data",
        python_callable=validate_data,
        # Validation failures should fail fast, not retry against the
        # same broken dataset.
        retries=0,
    )

    task_preprocess = PythonOperator(
        task_id="preprocess_data",
        python_callable=preprocess_data,
    )

    task_dvc = PythonOperator(
        task_id="version_data_dvc",
        python_callable=version_data_dvc,
        # DVC operations involve network I/O — keep the default retries.
    )

    task_train = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
        execution_timeout=timedelta(hours=6),
        # Training is expensive; only retry on truly transient failures.
        retries=1,
        retry_delay=timedelta(minutes=15),
    )

    task_evaluate = PythonOperator(
        task_id="evaluate_model",
        python_callable=evaluate_model,
    )

    task_register = PythonOperator(
        task_id="register_model",
        python_callable=register_model,
    )

    task_notify = EmailOperator(
        task_id="send_success_email",
        to=PIPELINE_CONFIG["notify_email"],
        subject="[SUCCESS] ML Training Pipeline - {{ ds }}",
        html_content="""
        <h3>ML Training Pipeline Completed Successfully</h3>
        <p><strong>Execution Date:</strong> {{ ds }}</p>
        <p><strong>Status:</strong> SUCCESS</p>
        <p>View results in MLflow: <a href="http://mlflow:5000">MLflow UI</a></p>
        <p>View pipeline: <a href="http://airflow:8080/dags/ml_training_pipeline/grid">Airflow DAG</a></p>
        """,
    )

    (
        task_ingest
        >> task_validate
        >> task_preprocess
        >> task_dvc
        >> task_train
        >> task_evaluate
        >> task_register
        >> task_notify
    )


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    print(f"DAG: {dag.dag_id}")
    print(f"Schedule: {dag.schedule_interval}")
    print(f"Tasks: {len(dag.tasks)}")
    print("\nTask dependencies:")
    for task in dag.tasks:
        print(
            f"  {task.task_id}: "
            f"upstream={sorted(task.upstream_task_ids)} "
            f"downstream={sorted(task.downstream_task_ids)}"
        )
