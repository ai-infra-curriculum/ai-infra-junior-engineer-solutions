# Exercise 06: Airflow Workflow Monitoring - Solution

Complete solution for orchestrating and monitoring ML workflows using Apache Airflow with integrated monitoring using Prometheus and Grafana.

## Overview

This solution demonstrates:
- Complete ML pipeline orchestration with Airflow
- Data processing, model training, and deployment workflows
- Monitoring and alerting integration with Prometheus/Grafana
- Proper task dependency management
- Error handling and retry logic
- XCom for inter-task communication
- Docker Compose setup for local development

## Quick Start

### Prerequisites

- Docker and Docker Compose installed
- 8GB RAM minimum
- Ports available: 8080 (Airflow), 9090 (Prometheus), 3000 (Grafana)

### Setup and Run

```bash
# 1. Navigate to the exercise directory
cd exercise-06-airflow-workflow-monitoring

# 2. Run setup script
./scripts/setup.sh

# 3. Start all services
./scripts/start.sh

# 4. Wait for services to be ready (1-2 minutes)
# Check status
docker-compose ps
```

### Access Services

- **Airflow UI**: http://localhost:8080
  - Username: `airflow`
  - Password: `airflow`

- **Prometheus**: http://localhost:9090

- **Grafana**: http://localhost:3000
  - Username: `admin`
  - Password: `admin`

### Trigger Your First DAG

**Via UI:**
1. Go to http://localhost:8080
2. Toggle the `ml_pipeline_dag` to enable it
3. Click the play button to trigger manually

**Via CLI:**
```bash
docker-compose exec airflow-webserver airflow dags trigger ml_pipeline_dag
```

## Project Structure

```
exercise-06-airflow-workflow-monitoring/
├── README.md                   # This file
├── STEP_BY_STEP.md            # Detailed implementation guide
├── dags/
│   ├── __init__.py            # DAG package init
│   ├── ml_pipeline_dag.py     # Main ML pipeline
│   └── monitoring_dag.py      # System monitoring DAG
├── src/
│   ├── __init__.py            # Source package init
│   ├── data_processing.py     # Data processing tasks
│   ├── model_training.py      # Model training tasks
│   └── model_deployment.py    # Deployment tasks
├── tests/
│   ├── __init__.py            # Test package init
│   ├── test_dags.py           # DAG validation tests
│   └── test_tasks.py          # Task unit tests
├── docker/
│   ├── Dockerfile             # Custom Airflow image
│   └── docker-compose.yml     # Full stack definition
├── config/
│   ├── prometheus.yml         # Prometheus config
│   └── statsd_mapping.yml     # StatsD exporter config
├── scripts/
│   ├── setup.sh               # Initial setup
│   ├── start.sh               # Start services
│   └── test.sh                # Run tests
├── requirements.txt           # Python dependencies
└── .gitignore                 # Git ignore patterns
```

## DAG Workflows

### 1. ML Pipeline DAG (`ml_pipeline_dag`)

Main ML workflow that runs daily:

```
download_data → validate_data → preprocess_data → feature_engineering
                                                          ↓
                    deploy_model ← evaluate_model ← train_model
                          ↓
                  send_notification
```

**Tasks:**
- `download_data`: Download dataset from storage
- `validate_data`: Validate data quality
- `preprocess_data`: Clean and normalize data
- `feature_engineering`: Generate features
- `train_model`: Train ML model
- `evaluate_model`: Evaluate model performance
- `deploy_model`: Deploy to production
- `send_notification`: Notify team of completion

**Schedule**: Daily at 2 AM (`0 2 * * *`)

### 2. Monitoring DAG (`monitoring_dag`)

System health monitoring workflow:

```
check_dag_runs → check_task_failures → check_scheduler_health
                                              ↓
                                    generate_health_report
                                              ↓
                                    send_alerts (if needed)
```

**Tasks:**
- `check_dag_runs`: Monitor DAG execution metrics
- `check_task_failures`: Detect task failure patterns
- `check_scheduler_health`: Verify scheduler status
- `generate_health_report`: Create health summary
- `send_alerts`: Alert on anomalies

**Schedule**: Hourly (`0 * * * *`)

## Key Features

### 1. Task Dependencies

Tasks are properly ordered with dependencies:

```python
# Linear dependencies
task1 >> task2 >> task3

# Fan-out
task1 >> [task2, task3, task4]

# Fan-in
[task1, task2] >> task3
```

### 2. XCom for Data Passing

Tasks communicate via XCom:

```python
# Push data
def task1(**context):
    return {"accuracy": 0.95, "size": 100}

# Pull data
def task2(**context):
    ti = context['ti']
    data = ti.xcom_pull(task_ids='task1')
    print(f"Accuracy: {data['accuracy']}")
```

### 3. Error Handling

Automatic retries and error handling:

```python
default_args = {
    'retries': 2,
    'retry_delay': timedelta(minutes=1),
    'retry_exponential_backoff': True,
}
```

### 4. Monitoring Integration

- StatsD metrics exported to Prometheus
- Grafana dashboards for visualization
- Custom metrics from tasks
- SLA monitoring

## Testing

Run all tests:

```bash
./scripts/test.sh
```

Or run specific tests:

```bash
# Test DAG integrity
docker-compose exec airflow-webserver pytest tests/test_dags.py -v

# Test individual tasks
docker-compose exec airflow-webserver pytest tests/test_tasks.py -v
```

## Monitoring and Metrics

### Prometheus Queries

Access Prometheus at http://localhost:9090 and try:

```promql
# Task duration
airflow_task_duration{dag_id="ml_pipeline_dag"}

# Task failures
rate(airflow_task_failed[5m])

# DAG run success rate
airflow_dagrun_success / airflow_dagrun_total

# Scheduler heartbeat
airflow_scheduler_heartbeat
```

### Grafana Dashboards

1. Add Prometheus data source in Grafana
2. Import pre-built Airflow dashboards or create custom ones
3. Monitor:
   - DAG run times
   - Task success/failure rates
   - Queue sizes
   - Executor metrics

## Common Operations

### View Logs

```bash
# View task logs
docker-compose exec airflow-webserver airflow tasks logs ml_pipeline_dag download_data 2024-01-15

# Follow scheduler logs
docker-compose logs -f airflow-scheduler

# Follow all Airflow logs
docker-compose logs -f airflow-webserver airflow-scheduler
```

### List DAGs

```bash
docker-compose exec airflow-webserver airflow dags list
```

### Pause/Unpause DAG

```bash
# Pause
docker-compose exec airflow-webserver airflow dags pause ml_pipeline_dag

# Unpause
docker-compose exec airflow-webserver airflow dags unpause ml_pipeline_dag
```

### Backfill DAG

```bash
docker-compose exec airflow-webserver airflow dags backfill \
  ml_pipeline_dag \
  --start-date 2024-01-01 \
  --end-date 2024-01-07
```

## Best Practices Demonstrated

1. **Idempotent Tasks**: All tasks can be safely re-run
2. **Task Timeouts**: Prevent hanging tasks
3. **Resource Pools**: Limit concurrent resource usage
4. **Proper Error Handling**: Graceful failure handling
5. **Code Organization**: Separation of concerns
6. **Testing**: Comprehensive test coverage
7. **Documentation**: Clear docstrings and comments
8. **Configuration Management**: Environment-based config

## Troubleshooting

### DAG Not Appearing

```bash
# Check DAG for errors
docker-compose exec airflow-webserver python /opt/airflow/dags/ml_pipeline_dag.py

# Check scheduler logs
docker-compose logs airflow-scheduler
```

### Tasks Failing

1. Check task logs in UI
2. Verify XCom data is being passed correctly
3. Check resource availability
4. Review retry configuration

### Slow Performance

1. Check executor queue size
2. Verify database performance
3. Review parallelism settings
4. Check resource constraints

## Cleanup

```bash
# Stop services (keep data)
docker-compose down

# Stop and remove all data
docker-compose down -v

# Remove all containers and images
docker-compose down --rmi all -v
```

## Learning Outcomes

After completing this solution, you will understand:

- Workflow orchestration with Apache Airflow
- DAG creation and task dependencies
- Monitoring integration with Prometheus/Grafana
- Error handling and retry logic
- Testing Airflow workflows
- Production-ready Airflow setup
- Best practices for ML pipeline orchestration

## Next Steps

1. Add more complex workflows (branching, conditional execution)
2. Integrate with cloud services (AWS S3, GCP BigQuery)
3. Implement custom operators
4. Add data quality checks with Great Expectations
5. Set up production Kubernetes deployment
6. Implement CI/CD for DAG deployment

## Resources

- [Apache Airflow Documentation](https://airflow.apache.org/docs/)
- [Airflow Best Practices](https://airflow.apache.org/docs/apache-airflow/stable/best-practices.html)
- [Prometheus Metrics](https://prometheus.io/docs/concepts/metric_types/)
- [Grafana Dashboards](https://grafana.com/docs/grafana/latest/dashboards/)

## Support

For issues or questions:
1. Check the STEP_BY_STEP.md guide
2. Review DAG logs in Airflow UI
3. Check container logs: `docker-compose logs`
4. Verify all services are running: `docker-compose ps`
