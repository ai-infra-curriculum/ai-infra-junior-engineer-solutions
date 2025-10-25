# Step-by-Step Implementation Guide

This guide walks through the complete implementation of the Airflow workflow monitoring solution.

## Table of Contents

1. [Environment Setup](#step-1-environment-setup)
2. [Docker Configuration](#step-2-docker-configuration)
3. [Airflow Configuration](#step-3-airflow-configuration)
4. [Creating the ML Pipeline DAG](#step-4-creating-the-ml-pipeline-dag)
5. [Implementing Task Functions](#step-5-implementing-task-functions)
6. [Creating the Monitoring DAG](#step-6-creating-the-monitoring-dag)
7. [Setting Up Tests](#step-7-setting-up-tests)
8. [Prometheus Integration](#step-8-prometheus-integration)
9. [Grafana Dashboards](#step-9-grafana-dashboards)
10. [Running and Validating](#step-10-running-and-validating)

---

## Step 1: Environment Setup

### Create Project Structure

```bash
# Create all directories
mkdir -p exercise-06-airflow-workflow-monitoring/{dags,src,tests,docker,config,scripts,logs,plugins}

cd exercise-06-airflow-workflow-monitoring
```

### Initialize Python Packages

```bash
# Create __init__.py files
touch dags/__init__.py
touch src/__init__.py
touch tests/__init__.py
```

### Set Environment Variables

Create `.env` file:

```bash
# Get your user ID
echo "AIRFLOW_UID=$(id -u)" > .env
cat .env
```

---

## Step 2: Docker Configuration

### Understanding the Docker Stack

The stack includes:
- **PostgreSQL**: Airflow metadata database
- **Airflow Webserver**: Web UI
- **Airflow Scheduler**: Task scheduling
- **StatsD Exporter**: Convert Airflow metrics to Prometheus format
- **Prometheus**: Metrics storage
- **Grafana**: Visualization

### Create docker-compose.yml

The `docker-compose.yml` file uses:
- **x-airflow-common**: Shared configuration for Airflow services
- **LocalExecutor**: Runs tasks in the same process
- **PostgreSQL backend**: Production-ready metadata DB
- **StatsD integration**: Export metrics to Prometheus

Key environment variables:
```yaml
AIRFLOW__CORE__EXECUTOR: LocalExecutor  # Executor type
AIRFLOW__METRICS__STATSD_ON: 'true'    # Enable metrics
AIRFLOW__METRICS__STATSD_HOST: statsd-exporter
```

### Create Custom Dockerfile

The custom Dockerfile extends the base Airflow image to:
- Install additional Python packages
- Add custom dependencies
- Configure permissions

---

## Step 3: Airflow Configuration

### Prometheus Configuration

The `config/prometheus.yml` tells Prometheus:
- Where to find metrics (statsd-exporter)
- How often to scrape (15s intervals)
- What to monitor

### StatsD Mapping

The `config/statsd_mapping.yml` maps Airflow's StatsD metrics to Prometheus format:

```yaml
airflow.dag_processing.* → airflow_dag_processing
airflow.scheduler.* → airflow_scheduler
airflow.task.* → airflow_task
```

This standardizes metric names for easier querying.

---

## Step 4: Creating the ML Pipeline DAG

### DAG Structure Overview

```python
with DAG(
    'ml_pipeline_dag',
    default_args=default_args,
    description='ML model training pipeline',
    schedule_interval='0 2 * * *',  # 2 AM daily
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['ml', 'production'],
) as dag:
    # Define tasks here
```

### Key DAG Parameters

- **dag_id**: Unique identifier
- **schedule_interval**: Cron expression or timedelta
- **start_date**: When DAG becomes active
- **catchup**: Whether to backfill missed runs
- **tags**: Organization and filtering
- **default_args**: Applied to all tasks

### Default Arguments

```python
default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,      # Don't wait for previous runs
    'email_on_failure': True,      # Alert on failures
    'email': ['alerts@example.com'],
    'retries': 2,                  # Retry failed tasks
    'retry_delay': timedelta(minutes=1),
}
```

### Task Dependency Patterns

**Linear flow:**
```python
task1 >> task2 >> task3
```

**Fan-out (parallel execution):**
```python
task1 >> [task2, task3, task4]
```

**Fan-in (join):**
```python
[task1, task2, task3] >> task4
```

**Complex dependencies:**
```python
task1 >> task2
task1 >> task3
[task2, task3] >> task4
```

---

## Step 5: Implementing Task Functions

### Task Function Structure

All task functions receive `**context`:

```python
def my_task(**context):
    # Access context
    ti = context['ti']  # Task Instance
    ds = context['ds']  # Execution date as string

    # Do work
    result = process_data()

    # Return data (stored in XCom)
    return result
```

### Using XCom for Data Passing

**Push data (automatic):**
```python
def task1(**context):
    return {"accuracy": 0.95, "samples": 10000}
```

**Pull data:**
```python
def task2(**context):
    ti = context['ti']
    data = ti.xcom_pull(task_ids='task1')
    accuracy = data['accuracy']
```

### Data Processing Task Example

```python
def download_data(**context):
    """Download dataset from storage"""
    import time

    print("Downloading dataset...")
    # Simulate download
    time.sleep(2)

    # Return metadata
    return {
        'dataset_size_gb': 1.5,
        'num_samples': 10000,
        'download_time': time.time()
    }
```

### Model Training Task Example

```python
def train_model(**context):
    """Train ML model"""
    ti = context['ti']

    # Get data from previous task
    data_info = ti.xcom_pull(task_ids='preprocess_data')
    num_samples = data_info['processed_samples']

    print(f"Training on {num_samples} samples...")

    # Simulate training
    import time
    import random
    time.sleep(5)

    # Generate metrics
    accuracy = random.uniform(0.85, 0.95)

    return {
        'accuracy': round(accuracy, 4),
        'model_size_mb': 450,
        'training_time_minutes': 45
    }
```

### Error Handling in Tasks

```python
def validate_data(**context):
    """Validate data quality"""
    ti = context['ti']
    data = ti.xcom_pull(task_ids='download_data')

    # Validate
    if data['num_samples'] < 1000:
        raise ValueError(
            f"Insufficient samples: {data['num_samples']} < 1000"
        )

    # Check for nulls
    null_pct = check_nulls(data)
    if null_pct > 0.05:
        raise ValueError(
            f"Too many nulls: {null_pct:.2%} > 5%"
        )

    print("Data validation passed!")
    return True
```

---

## Step 6: Creating the Monitoring DAG

### Purpose of Monitoring DAG

The monitoring DAG checks:
- DAG execution metrics
- Task failure rates
- Scheduler health
- Resource usage

### Monitoring Tasks

**Check DAG Runs:**
```python
def check_dag_runs(**context):
    """Monitor recent DAG executions"""
    from airflow.models import DagRun
    from datetime import timedelta

    # Get recent runs
    recent_runs = DagRun.find(
        execution_start_date=datetime.now() - timedelta(hours=24)
    )

    # Calculate metrics
    total_runs = len(recent_runs)
    failed_runs = len([r for r in recent_runs if r.state == 'failed'])

    failure_rate = failed_runs / total_runs if total_runs > 0 else 0

    return {
        'total_runs': total_runs,
        'failed_runs': failed_runs,
        'failure_rate': failure_rate
    }
```

**Check Task Failures:**
```python
def check_task_failures(**context):
    """Detect task failure patterns"""
    from airflow.models import TaskInstance

    # Get recent task instances
    failed_tasks = TaskInstance.find(
        state='failed',
        execution_date=datetime.now() - timedelta(hours=24)
    )

    # Group by task
    task_failures = {}
    for ti in failed_tasks:
        key = f"{ti.dag_id}.{ti.task_id}"
        task_failures[key] = task_failures.get(key, 0) + 1

    return task_failures
```

### Alerting Logic

```python
def send_alerts(**context):
    """Send alerts for anomalies"""
    ti = context['ti']

    # Get monitoring data
    dag_metrics = ti.xcom_pull(task_ids='check_dag_runs')

    # Check thresholds
    if dag_metrics['failure_rate'] > 0.2:  # 20% threshold
        print(f"ALERT: High failure rate: {dag_metrics['failure_rate']:.2%}")
        # Send actual alert (email, Slack, PagerDuty, etc.)
        send_slack_alert(
            f"Airflow Alert: {dag_metrics['failure_rate']:.0%} DAG failure rate"
        )
```

---

## Step 7: Setting Up Tests

### Why Test DAGs?

- Catch syntax errors before deployment
- Verify DAG structure
- Validate task logic
- Ensure dependencies are correct

### DAG Integrity Tests

```python
def test_dag_loaded():
    """Test that DAGs load without errors"""
    from airflow.models import DagBag

    dagbag = DagBag()
    assert len(dagbag.import_errors) == 0, "DAG import failures"
```

### Task Tests

```python
def test_download_data_task():
    """Test download_data function"""
    from src.data_processing import download_data

    # Mock context
    context = {'ti': MockTaskInstance()}

    # Run task
    result = download_data(**context)

    # Validate
    assert 'dataset_size_gb' in result
    assert result['num_samples'] > 0
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_dags.py -v

# Run with coverage
pytest tests/ --cov=src --cov=dags
```

---

## Step 8: Prometheus Integration

### How Metrics Flow

```
Airflow → StatsD → StatsD Exporter → Prometheus → Grafana
```

1. **Airflow** emits metrics via StatsD protocol
2. **StatsD Exporter** converts to Prometheus format
3. **Prometheus** scrapes and stores metrics
4. **Grafana** visualizes metrics

### Key Airflow Metrics

**DAG Metrics:**
- `airflow_dagrun_duration_seconds`: How long DAG runs take
- `airflow_dagrun_success`: Successful runs
- `airflow_dagrun_failed`: Failed runs

**Task Metrics:**
- `airflow_task_duration`: Task execution time
- `airflow_task_success`: Successful tasks
- `airflow_task_failed`: Failed tasks

**Scheduler Metrics:**
- `airflow_scheduler_heartbeat`: Scheduler health
- `airflow_dag_processing_total_parse_time`: DAG parse time

### Useful Prometheus Queries

**Task failure rate:**
```promql
rate(airflow_task_failed[5m]) / rate(airflow_task_start[5m])
```

**Average DAG duration:**
```promql
avg(airflow_dagrun_duration_seconds) by (dag_id)
```

**Tasks in queue:**
```promql
airflow_executor_queued_tasks
```

---

## Step 9: Grafana Dashboards

### Setting Up Grafana

1. Access Grafana at http://localhost:3000
2. Login (admin/admin)
3. Add Prometheus data source:
   - Configuration → Data Sources → Add
   - Type: Prometheus
   - URL: `http://prometheus:9090`
   - Save & Test

### Creating Dashboards

**Panel 1: DAG Success Rate**
- Visualization: Gauge
- Query:
```promql
sum(rate(airflow_dagrun_success[5m])) /
sum(rate(airflow_dagrun_total[5m]))
```

**Panel 2: Task Duration**
- Visualization: Time series
- Query:
```promql
avg(airflow_task_duration) by (task_id)
```

**Panel 3: Failed Tasks**
- Visualization: Bar chart
- Query:
```promql
sum(rate(airflow_task_failed[1h])) by (dag_id)
```

### Dashboard JSON

Export and share dashboards as JSON:
- Dashboard Settings → JSON Model
- Copy and save to version control
- Import on other Grafana instances

---

## Step 10: Running and Validating

### Start Services

```bash
# Run setup script
./scripts/setup.sh

# Start all services
./scripts/start.sh

# Check service health
docker-compose ps
```

All services should show "Up" and "healthy".

### Validate DAGs

```bash
# List DAGs
docker-compose exec airflow-webserver airflow dags list

# Test DAG
docker-compose exec airflow-webserver python /opt/airflow/dags/ml_pipeline_dag.py

# Show DAG structure
docker-compose exec airflow-webserver airflow dags show ml_pipeline_dag
```

### Trigger DAG

**Via UI:**
1. Go to http://localhost:8080
2. Login (airflow/airflow)
3. Enable DAG by toggling switch
4. Click play button to trigger

**Via CLI:**
```bash
docker-compose exec airflow-webserver airflow dags trigger ml_pipeline_dag
```

### Monitor Execution

1. **Graph View**: See task states in real-time
2. **Tree View**: Historical runs
3. **Gantt View**: Task timing
4. **Task Logs**: Click task → View Log

### Verify Metrics

1. Go to Prometheus (http://localhost:9090)
2. Check metrics are being collected:
   - Enter `airflow_` in query box
   - See autocomplete suggestions
   - Execute query

3. Go to Grafana (http://localhost:3000)
4. Create panels with queries
5. Save dashboard

---

## Common Issues and Solutions

### DAG Not Appearing

**Problem**: DAG doesn't show in UI

**Solutions**:
1. Check for syntax errors:
   ```bash
   python dags/ml_pipeline_dag.py
   ```

2. Check scheduler logs:
   ```bash
   docker-compose logs airflow-scheduler
   ```

3. Verify DAG file location:
   ```bash
   docker-compose exec airflow-webserver ls /opt/airflow/dags/
   ```

### Tasks Not Starting

**Problem**: Tasks stuck in queued state

**Solutions**:
1. Check executor is running
2. Verify task parallelism settings
3. Check resource pools
4. Review scheduler logs

### Metrics Not Appearing

**Problem**: No metrics in Prometheus

**Solutions**:
1. Verify StatsD is enabled in Airflow config
2. Check statsd-exporter is running
3. Test StatsD connection:
   ```bash
   docker-compose logs statsd-exporter
   ```
4. Verify Prometheus is scraping:
   - Go to http://localhost:9090/targets

---

## Best Practices

### 1. Idempotent Tasks

Tasks should produce the same result when re-run:

```python
def idempotent_task(**context):
    ds = context['ds']  # Execution date

    output_file = f"/data/output_{ds}.csv"

    # Check if already done
    if file_exists(output_file):
        print("Already processed, skipping...")
        return

    # Process data
    result = process_data()
    save(result, output_file)
```

### 2. Use Sensors for External Dependencies

Wait for external events:

```python
from airflow.sensors.filesystem import FileSensor

wait_for_file = FileSensor(
    task_id='wait_for_data',
    filepath='/data/input.csv',
    timeout=3600,
    poke_interval=60,
)
```

### 3. Task Groups for Organization

Group related tasks:

```python
from airflow.utils.task_group import TaskGroup

with TaskGroup("data_preparation") as data_prep:
    download = PythonOperator(...)
    validate = PythonOperator(...)
    transform = PythonOperator(...)

    download >> validate >> transform

# Use in DAG
start >> data_prep >> train_model
```

### 4. Connection Management

Store credentials securely:

```python
from airflow.hooks.base import BaseHook

def use_connection(**context):
    # Get connection from Airflow UI
    conn = BaseHook.get_connection('my_database')

    db_url = f"postgresql://{conn.login}:{conn.password}@{conn.host}/{conn.schema}"
```

### 5. Variables for Configuration

Store configuration values:

```python
from airflow.models import Variable

def configurable_task(**context):
    # Get variable (set in UI)
    batch_size = Variable.get("batch_size", default_var=100)

    process_data(batch_size=batch_size)
```

---

## Production Considerations

### High Availability

- Run multiple schedulers (Airflow 2.0+)
- Use CeleryExecutor for distributed execution
- External database (RDS, Cloud SQL)
- Redis for Celery backend

### Security

- Enable authentication (LDAP, OAuth)
- Use RBAC for access control
- Encrypt connections
- Secure secret management (AWS Secrets Manager, Vault)

### Scaling

- Use KubernetesExecutor for dynamic scaling
- Separate DAG parsing from execution
- Use task pools to limit concurrency
- Implement proper resource management

### CI/CD

- Test DAGs in staging environment
- Automated testing on commit
- DAG deployment via GitOps
- Version control for all DAG code

---

## Summary

You've now implemented:

1. Complete Airflow setup with Docker Compose
2. ML pipeline DAG with proper dependencies
3. Monitoring DAG for system health
4. Task implementations with error handling
5. XCom for inter-task communication
6. Prometheus metrics integration
7. Grafana dashboards
8. Comprehensive tests
9. Production-ready scripts

This solution demonstrates real-world workflow orchestration patterns used in production ML systems.

## Next Steps

1. Add more complex workflows (branching, loops)
2. Integrate with cloud services (S3, BigQuery)
3. Implement custom operators
4. Add data quality checks
5. Set up alerting (PagerDuty, Slack)
6. Deploy to Kubernetes
7. Implement DAG versioning
8. Add cost monitoring
