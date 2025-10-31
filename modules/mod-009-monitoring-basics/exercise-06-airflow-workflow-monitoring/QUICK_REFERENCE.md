# Quick Reference Guide

Essential commands and concepts for the Airflow Workflow Monitoring exercise.

## Quick Start Commands

```bash
# Setup (first time only)
./scripts/setup.sh

# Start all services
./scripts/start.sh

# Run tests
./scripts/test.sh

# Stop services
cd docker && docker-compose down

# Stop and remove all data
cd docker && docker-compose down -v
```

## Service URLs

| Service | URL | Credentials |
|---------|-----|-------------|
| Airflow UI | http://localhost:8080 | airflow / airflow |
| Prometheus | http://localhost:9090 | None |
| Grafana | http://localhost:3000 | admin / admin |

## Common Airflow CLI Commands

```bash
# Note: All commands should be run from docker directory or prefix with docker-compose -f docker/docker-compose.yml

# List all DAGs
docker-compose exec airflow-webserver airflow dags list

# Trigger a DAG
docker-compose exec airflow-webserver airflow dags trigger ml_pipeline_dag

# Pause/Unpause a DAG
docker-compose exec airflow-webserver airflow dags pause ml_pipeline_dag
docker-compose exec airflow-webserver airflow dags unpause ml_pipeline_dag

# Test a specific task
docker-compose exec airflow-webserver airflow tasks test ml_pipeline_dag download_data 2024-01-15

# View task logs
docker-compose exec airflow-webserver airflow tasks logs ml_pipeline_dag download_data 2024-01-15

# List DAG runs
docker-compose exec airflow-webserver airflow dags list-runs -d ml_pipeline_dag

# Check for import errors
docker-compose exec airflow-webserver airflow dags list-import-errors

# Show DAG structure
docker-compose exec airflow-webserver airflow dags show ml_pipeline_dag

# Backfill a DAG
docker-compose exec airflow-webserver airflow dags backfill ml_pipeline_dag \
  --start-date 2024-01-01 --end-date 2024-01-07
```

## Docker Commands

```bash
# View running containers
docker-compose ps

# View logs (all services)
docker-compose logs -f

# View logs (specific service)
docker-compose logs -f airflow-scheduler
docker-compose logs -f airflow-webserver

# Restart a service
docker-compose restart airflow-scheduler

# Execute command in container
docker-compose exec airflow-webserver bash

# Check container resource usage
docker stats
```

## DAG Development

### Basic DAG Structure

```python
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

default_args = {
    'owner': 'data-team',
    'retries': 2,
    'retry_delay': timedelta(minutes=1),
}

with DAG(
    'my_dag',
    default_args=default_args,
    description='My DAG description',
    schedule_interval='0 2 * * *',  # Daily at 2 AM
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['example'],
) as dag:

    task1 = PythonOperator(
        task_id='task1',
        python_callable=my_function,
    )
```

### Task Dependencies

```python
# Linear
task1 >> task2 >> task3

# Parallel
task1 >> [task2, task3, task4]

# Fan-in
[task1, task2, task3] >> task4

# Multiple dependencies
task1 >> task2
task1 >> task3
[task2, task3] >> task4
```

### XCom Usage

```python
# Push data (automatic from return value)
def task1(**context):
    return {"key": "value"}

# Pull data
def task2(**context):
    ti = context['ti']
    data = ti.xcom_pull(task_ids='task1')
    print(data['key'])
```

## Prometheus Queries

Access Prometheus at http://localhost:9090 and try these queries:

```promql
# Task duration by DAG
avg(airflow_task_duration) by (dag_id)

# Task failure rate
rate(airflow_task_failed[5m]) / rate(airflow_task_start[5m])

# DAG run duration
avg(airflow_dagrun_duration_success) by (dag_id)

# Tasks running
airflow_scheduler_tasks_running

# Executor queue size
airflow_executor_queued_tasks

# Scheduler heartbeat
airflow_scheduler_heartbeat

# Tasks by state
sum(airflow_task_finish) by (state)
```

## Grafana Dashboard Setup

1. **Add Prometheus Data Source:**
   - Configuration → Data Sources → Add data source
   - Type: Prometheus
   - URL: `http://prometheus:9090`
   - Save & Test

2. **Create Dashboard:**
   - Create → Dashboard → Add new panel
   - Add queries from above
   - Customize visualization

3. **Useful Panels:**
   - Task Success Rate (Gauge)
   - DAG Duration (Time series)
   - Failed Tasks (Bar chart)
   - Task States (Pie chart)

## Testing

```bash
# Run all tests
./scripts/test.sh

# Run specific test file
docker-compose exec airflow-webserver pytest tests/test_dags.py -v

# Run specific test
docker-compose exec airflow-webserver pytest tests/test_dags.py::TestMLPipelineDAG::test_dag_exists -v

# Run with coverage
docker-compose exec airflow-webserver pytest tests/ --cov=src --cov=dags

# Test DAG syntax
python dags/ml_pipeline_dag.py
```

## Debugging

```bash
# Check DAG file for syntax errors
python dags/my_dag.py

# Test task in isolation
airflow tasks test dag_id task_id 2024-01-15

# View XCom data
# Via UI: Click task → XCom tab

# Check task instance state
airflow tasks state dag_id task_id 2024-01-15

# Clear task state (for rerun)
airflow tasks clear dag_id -t task_id --start-date 2024-01-15 --end-date 2024-01-15

# Check database
docker-compose exec postgres psql -U airflow -d airflow
```

## Environment Variables

Key variables in `.env`:

```bash
AIRFLOW_UID=50000                          # Your user ID
_AIRFLOW_WWW_USER_USERNAME=airflow        # Admin username
_AIRFLOW_WWW_USER_PASSWORD=airflow        # Admin password
GF_SECURITY_ADMIN_USER=admin              # Grafana admin
GF_SECURITY_ADMIN_PASSWORD=admin          # Grafana password
```

## Schedule Expressions

Common cron expressions:

```python
# Every minute
'* * * * *'

# Every hour at minute 0
'0 * * * *'

# Daily at 2 AM
'0 2 * * *'

# Every Monday at 9 AM
'0 9 * * 1'

# First day of month at midnight
'0 0 1 * *'

# Or use timedelta
schedule_interval=timedelta(hours=1)
```

## Task Operators

Common operators:

```python
# Python function
from airflow.operators.python import PythonOperator
task = PythonOperator(task_id='task', python_callable=func)

# Bash command
from airflow.operators.bash import BashOperator
task = BashOperator(task_id='task', bash_command='echo "hello"')

# Branch (conditional)
from airflow.operators.python import BranchPythonOperator
task = BranchPythonOperator(task_id='task', python_callable=choose_branch)

# Dummy (no-op)
from airflow.operators.empty import EmptyOperator
task = EmptyOperator(task_id='task')
```

## File Locations (in container)

```
/opt/airflow/
├── dags/              # DAG definitions
├── logs/              # Task logs
├── plugins/           # Custom plugins
├── src/               # Source code
└── airflow.cfg        # Configuration
```

## Useful Airflow UI Features

- **Graph View:** Visual DAG structure and task states
- **Tree View:** Historical runs over time
- **Gantt View:** Task timing and duration
- **Code View:** View DAG source code
- **Task Instance Details:** Logs, duration, XCom, etc.
- **Admin → Connections:** Manage external connections
- **Admin → Variables:** Store configuration values

## Monitoring Checklist

Daily:
- [ ] Check DAG success rates
- [ ] Review failed tasks
- [ ] Monitor queue sizes
- [ ] Check scheduler health

Weekly:
- [ ] Review DAG performance trends
- [ ] Clean old task instances
- [ ] Update dashboards
- [ ] Review alerts

## Common Issues & Quick Fixes

| Issue | Quick Fix |
|-------|-----------|
| DAG not appearing | Check syntax: `python dags/my_dag.py` |
| Task stuck in queue | Restart scheduler |
| Import errors | Check logs: `docker-compose logs airflow-scheduler` |
| Slow UI | Clean DB: `airflow db clean` |
| No metrics | Check StatsD: `docker-compose logs statsd-exporter` |

## Best Practices

1. **Always test DAGs before deployment:**
   ```bash
   python dags/my_dag.py
   airflow dags test my_dag 2024-01-15
   ```

2. **Use tags for organization:**
   ```python
   tags=['ml', 'production', 'daily']
   ```

3. **Set appropriate retries:**
   ```python
   default_args = {'retries': 2, 'retry_delay': timedelta(minutes=1)}
   ```

4. **Add task documentation:**
   ```python
   task = PythonOperator(
       task_id='task',
       python_callable=func,
       doc_md="## Task Documentation\nDetailed description..."
   )
   ```

5. **Use execution_timeout:**
   ```python
   task = PythonOperator(
       task_id='task',
       python_callable=func,
       execution_timeout=timedelta(hours=2)
   )
   ```

## Resources

- [Apache Airflow Docs](https://airflow.apache.org/docs/)
- [Airflow Best Practices](https://airflow.apache.org/docs/apache-airflow/stable/best-practices.html)
- [Prometheus Query Examples](https://prometheus.io/docs/prometheus/latest/querying/examples/)
- [Grafana Documentation](https://grafana.com/docs/)

## Quick Health Check

Run this one-liner to check if everything is working:

```bash
docker-compose ps && \
echo "=== DAGs ===" && \
docker-compose exec airflow-webserver airflow dags list && \
echo "=== Import Errors ===" && \
docker-compose exec airflow-webserver airflow dags list-import-errors
```
