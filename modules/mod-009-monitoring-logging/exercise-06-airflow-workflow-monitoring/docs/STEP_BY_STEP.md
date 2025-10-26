# Step-by-Step Implementation Guide: Airflow Workflow Monitoring

## Overview

Monitor Apache Airflow ML pipelines! Learn DAG monitoring, task tracking, SLA monitoring, and Airflow observability best practices.

**Time**: 2 hours | **Difficulty**: Intermediate

---

## Learning Objectives

✅ Deploy Airflow on Kubernetes
✅ Monitor DAG execution
✅ Track task failures
✅ Set up SLA monitoring
✅ Integrate with Prometheus
✅ Create Airflow alerts
✅ Implement data quality checks

---

## Install Airflow

```bash
# Add Helm repo
helm repo add apache-airflow https://airflow.apache.org
helm repo update

# Install Airflow
helm install airflow apache-airflow/airflow \
  --namespace airflow \
  --create-namespace \
  --set executor=KubernetesExecutor \
  --set prometheus.enabled=true \
  --set workers.replicas=3

# Access UI
kubectl port-forward svc/airflow-webserver 8080:8080 -n airflow
```

---

## DAG with Monitoring

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.slack.operators.slack_webhook import SlackWebhookOperator
from datetime import datetime, timedelta
import logging

default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'email': ['alerts@example.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'sla': timedelta(hours=2)
}

dag = DAG(
    'ml_training_pipeline',
    default_args=default_args,
    description='ML training pipeline with monitoring',
    schedule_interval='0 2 * * *',
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['ml', 'training'],
)

def log_metrics(**context):
    """Log custom metrics to Prometheus"""
    from airflow.metrics import statsd
    statsd.incr('ml_training.started')
    statsd.timing('ml_training.duration', duration)

def train_model(**context):
    """Train ML model with monitoring"""
    import mlflow
    from prometheus_client import Counter, Histogram, push_to_gateway

    # Prometheus metrics
    training_runs = Counter('airflow_ml_training_runs', 'ML training runs')
    training_duration = Histogram('airflow_ml_training_duration_seconds', 'Training duration')

    with training_duration.time():
        # Train model
        with mlflow.start_run():
            model = train()

            # Log metrics
            mlflow.log_metrics({
                'accuracy': accuracy,
                'loss': loss
            })

    training_runs.inc()

    # Push to Prometheus Pushgateway
    push_to_gateway('pushgateway:9091', job='airflow_training', registry=registry)

# Tasks
extract = PythonOperator(
    task_id='extract_data',
    python_callable=extract_data,
    dag=dag,
)

validate = PythonOperator(
    task_id='validate_data',
    python_callable=validate_data,
    dag=dag,
)

train = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag,
)

evaluate = PythonOperator(
    task_id='evaluate_model',
    python_callable=evaluate_model,
    dag=dag,
)

# Alert on failure
alert = SlackWebhookOperator(
    task_id='alert_on_failure',
    http_conn_id='slack_webhook',
    message='ML training failed: {{ ti.task_id }}',
    trigger_rule='one_failed',
    dag=dag
)

extract >> validate >> train >> evaluate >> alert
```

---

## Airflow Metrics in Prometheus

```yaml
# servicemonitor.yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: airflow-metrics
  namespace: airflow
spec:
  selector:
    matchLabels:
      app: airflow
  endpoints:
  - port: metrics
    interval: 30s
```

### PromQL Queries

```promql
# DAG run duration
airflow_dag_run_duration_seconds

# Task failures
sum(rate(airflow_task_failures_total[5m])) by (dag_id, task_id)

# Running tasks
airflow_task_status{status="running"}

# SLA misses
airflow_sla_miss

# Scheduler heartbeat
time() - airflow_scheduler_heartbeat < 60
```

---

## SLA Monitoring

```python
def sla_miss_callback(dag, task_list, blocking_task_list, slas, blocking_tis):
    """Called when SLA is missed"""
    import requests

    # Send alert
    requests.post('https://hooks.slack.com/services/YOUR/WEBHOOK', json={
        'text': f'SLA missed for DAG {dag.dag_id}',
        'attachments': [{
            'fields': [
                {'title': 'Task', 'value': task.task_id}
                for task in task_list
            ]
        }]
    })

dag = DAG(
    'ml_pipeline',
    default_args={'sla': timedelta(hours=2)},
    sla_miss_callback=sla_miss_callback,
)
```

---

## Data Quality Checks

```python
from airflow.operators.python import BranchPythonOperator

def check_data_quality(**context):
    """Check data quality before training"""
    import pandas as pd

    data = pd.read_csv('/data/training_data.csv')

    # Quality checks
    checks = {
        'row_count': len(data) > 1000,
        'null_values': data.isnull().sum().sum() == 0,
        'duplicates': data.duplicated().sum() == 0,
        'schema': set(data.columns) == expected_columns
    }

    if all(checks.values()):
        return 'train_model'
    else:
        # Log failures
        context['ti'].xcom_push(key='failed_checks', value=[k for k, v in checks.items() if not v])
        return 'alert_data_quality'

quality_check = BranchPythonOperator(
    task_id='check_quality',
    python_callable=check_data_quality,
    dag=dag
)
```

---

## Alerting Rules

```yaml
# prometheus-rules.yaml
groups:
- name: airflow_alerts
  rules:
  - alert: AirflowDAGFailed
    expr: |
      airflow_dag_status{status="failed"} > 0
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "Airflow DAG failed"
      description: "DAG {{ $labels.dag_id }} has failed"

  - alert: AirflowTaskStuck
    expr: |
      time() - airflow_task_start_date{state="running"} > 3600
    for: 10m
    labels:
      severity: warning
    annotations:
      summary: "Airflow task running for too long"
```

---

## Best Practices

✅ Set appropriate SLAs for DAGs
✅ Implement data quality checks
✅ Monitor task duration trends
✅ Use task callbacks for alerts
✅ Log to structured format
✅ Track lineage with XComs
✅ Implement retry logic
✅ Monitor scheduler health
✅ Use pools to limit concurrency
✅ Regular DAG cleanup

---

**Airflow Workflow Monitoring mastered!** 🔄

**Congratulations!** You've completed the entire Monitoring & Logging module!
