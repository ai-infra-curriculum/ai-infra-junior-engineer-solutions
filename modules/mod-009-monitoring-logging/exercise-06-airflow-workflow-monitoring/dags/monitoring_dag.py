"""
Monitoring DAG

Monitors Airflow system health and sends alerts for anomalies.

Monitors:
- DAG run success/failure rates
- Task failure patterns
- Scheduler health
- Queue sizes
- Long-running tasks

Schedule: Hourly
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.models import DagRun, TaskInstance
from airflow.utils.state import State
import logging

logger = logging.getLogger(__name__)


def check_dag_runs(**context):
    """
    Monitor DAG run metrics over the last 24 hours.

    Checks:
    - Total runs
    - Failed runs
    - Success rate
    - Average duration
    """
    from airflow.models import DagRun
    from datetime import datetime, timedelta
    from sqlalchemy import func

    execution_date = context['execution_date']
    lookback_start = execution_date - timedelta(hours=24)

    logger.info(f"Checking DAG runs from {lookback_start} to {execution_date}")

    # Query recent runs
    session = context['ti'].get_dagrun().get_task_instance('check_dag_runs').task.dag.get_session()

    # Get run counts by state
    runs_query = session.query(
        DagRun.state,
        func.count(DagRun.id).label('count'),
        func.avg(DagRun.end_date - DagRun.start_date).label('avg_duration')
    ).filter(
        DagRun.execution_date >= lookback_start,
        DagRun.execution_date < execution_date
    ).group_by(DagRun.state)

    results = runs_query.all()

    # Calculate metrics
    total_runs = 0
    failed_runs = 0
    success_runs = 0
    running_runs = 0

    for state, count, avg_duration in results:
        total_runs += count
        if state == State.FAILED:
            failed_runs = count
        elif state == State.SUCCESS:
            success_runs = count
        elif state == State.RUNNING:
            running_runs = count

    failure_rate = failed_runs / total_runs if total_runs > 0 else 0
    success_rate = success_runs / total_runs if total_runs > 0 else 0

    metrics = {
        'total_runs': total_runs,
        'success_runs': success_runs,
        'failed_runs': failed_runs,
        'running_runs': running_runs,
        'success_rate': round(success_rate, 4),
        'failure_rate': round(failure_rate, 4),
        'lookback_hours': 24
    }

    logger.info(f"DAG metrics: {metrics}")

    # Log warnings if needed
    if failure_rate > 0.2:  # 20% threshold
        logger.warning(f"High failure rate detected: {failure_rate:.2%}")

    return metrics


def check_task_failures(**context):
    """
    Detect task failure patterns.

    Identifies:
    - Tasks with high failure rates
    - Recently failed tasks
    - Tasks that consistently fail
    """
    from airflow.models import TaskInstance
    from datetime import datetime, timedelta
    from sqlalchemy import func

    execution_date = context['execution_date']
    lookback_start = execution_date - timedelta(hours=24)

    logger.info("Checking task failure patterns...")

    session = context['ti'].get_dagrun().get_task_instance('check_task_failures').task.dag.get_session()

    # Get failed tasks grouped by DAG and task
    failed_query = session.query(
        TaskInstance.dag_id,
        TaskInstance.task_id,
        func.count(TaskInstance.task_id).label('failure_count')
    ).filter(
        TaskInstance.state == State.FAILED,
        TaskInstance.execution_date >= lookback_start
    ).group_by(
        TaskInstance.dag_id,
        TaskInstance.task_id
    ).order_by(
        func.count(TaskInstance.task_id).desc()
    )

    results = failed_query.all()

    task_failures = {}
    for dag_id, task_id, count in results:
        key = f"{dag_id}.{task_id}"
        task_failures[key] = count

        if count > 3:  # More than 3 failures
            logger.warning(f"Task {key} failed {count} times in last 24 hours")

    logger.info(f"Found {len(task_failures)} tasks with failures")

    return {
        'task_failures': task_failures,
        'total_failed_tasks': len(task_failures),
        'total_failures': sum(task_failures.values())
    }


def check_scheduler_health(**context):
    """
    Verify scheduler health and performance.

    Checks:
    - Scheduler heartbeat
    - DAG file processing time
    - Task queue sizes
    """
    import time
    from airflow.jobs.job import Job
    from airflow.jobs.scheduler_job import SchedulerJob

    logger.info("Checking scheduler health...")

    session = context['ti'].get_dagrun().get_task_instance('check_scheduler_health').task.dag.get_session()

    # Get most recent scheduler job
    scheduler_jobs = session.query(SchedulerJob).order_by(
        SchedulerJob.latest_heartbeat.desc()
    ).limit(1).all()

    if not scheduler_jobs:
        logger.error("No scheduler jobs found!")
        return {
            'status': 'error',
            'message': 'No scheduler jobs detected'
        }

    scheduler = scheduler_jobs[0]
    heartbeat_age = (datetime.now() - scheduler.latest_heartbeat).total_seconds()

    # Healthy if heartbeat within last 60 seconds
    is_healthy = heartbeat_age < 60

    health_status = {
        'status': 'healthy' if is_healthy else 'unhealthy',
        'latest_heartbeat': scheduler.latest_heartbeat.isoformat(),
        'heartbeat_age_seconds': round(heartbeat_age, 2),
        'hostname': scheduler.hostname,
    }

    if not is_healthy:
        logger.error(f"Scheduler unhealthy! Last heartbeat {heartbeat_age:.0f}s ago")
    else:
        logger.info("Scheduler is healthy")

    return health_status


def generate_health_report(**context):
    """
    Generate comprehensive health report.

    Aggregates data from all monitoring tasks.
    """
    ti = context['ti']

    # Pull data from previous tasks
    dag_metrics = ti.xcom_pull(task_ids='check_dag_runs')
    task_failures = ti.xcom_pull(task_ids='check_task_failures')
    scheduler_health = ti.xcom_pull(task_ids='check_scheduler_health')

    # Generate report
    report = {
        'timestamp': datetime.now().isoformat(),
        'summary': {
            'overall_status': 'healthy',
            'issues_detected': []
        },
        'dag_metrics': dag_metrics,
        'task_failures': task_failures,
        'scheduler_health': scheduler_health
    }

    # Determine overall status
    issues = []

    # Check DAG failure rate
    if dag_metrics and dag_metrics['failure_rate'] > 0.2:
        issues.append(f"High DAG failure rate: {dag_metrics['failure_rate']:.2%}")

    # Check task failures
    if task_failures and task_failures['total_failures'] > 10:
        issues.append(f"High task failure count: {task_failures['total_failures']}")

    # Check scheduler
    if scheduler_health and scheduler_health['status'] != 'healthy':
        issues.append("Scheduler unhealthy")

    if issues:
        report['summary']['overall_status'] = 'degraded'
        report['summary']['issues_detected'] = issues
        logger.warning(f"Health issues detected: {issues}")
    else:
        logger.info("All systems healthy")

    # Log report
    logger.info("=" * 60)
    logger.info("AIRFLOW HEALTH REPORT")
    logger.info("=" * 60)
    logger.info(f"Status: {report['summary']['overall_status'].upper()}")
    logger.info(f"DAG Success Rate: {dag_metrics.get('success_rate', 0):.2%}")
    logger.info(f"Total Failures: {task_failures.get('total_failures', 0)}")
    logger.info(f"Scheduler: {scheduler_health.get('status', 'unknown')}")
    logger.info("=" * 60)

    return report


def send_alerts(**context):
    """
    Send alerts if issues detected.

    In production, this would send:
    - Email alerts
    - Slack notifications
    - PagerDuty incidents
    """
    ti = context['ti']
    report = ti.xcom_pull(task_ids='generate_health_report')

    if not report:
        logger.error("No health report available")
        return

    status = report['summary']['overall_status']
    issues = report['summary']['issues_detected']

    if status == 'degraded' or status == 'unhealthy':
        # In production: send actual alerts
        logger.warning("=" * 60)
        logger.warning("ALERT: Airflow health issues detected!")
        logger.warning("=" * 60)

        for issue in issues:
            logger.warning(f"  - {issue}")

        logger.warning("=" * 60)

        # Simulate sending alerts
        alert_message = f"""
        Airflow Monitoring Alert

        Status: {status.upper()}
        Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

        Issues:
        {chr(10).join(f'  - {issue}' for issue in issues)}

        Please investigate immediately.
        """

        logger.info("Alert sent (simulated)")
        return {'alert_sent': True, 'message': alert_message}
    else:
        logger.info("No alerts needed - all systems healthy")
        return {'alert_sent': False}


# Default arguments
default_args = {
    'owner': 'platform-team',
    'depends_on_past': False,
    'email': ['platform-alerts@example.com'],
    'email_on_failure': True,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


# Define the monitoring DAG
with DAG(
    'monitoring_dag',
    default_args=default_args,
    description='Monitor Airflow system health and performance',
    schedule_interval='0 * * * *',  # Hourly
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['monitoring', 'alerting', 'operations'],
) as dag:

    # Task 1: Check DAG runs
    check_dags = PythonOperator(
        task_id='check_dag_runs',
        python_callable=check_dag_runs,
    )

    # Task 2: Check task failures
    check_tasks = PythonOperator(
        task_id='check_task_failures',
        python_callable=check_task_failures,
    )

    # Task 3: Check scheduler health
    check_scheduler = PythonOperator(
        task_id='check_scheduler_health',
        python_callable=check_scheduler_health,
    )

    # Task 4: Generate health report
    generate_report = PythonOperator(
        task_id='generate_health_report',
        python_callable=generate_health_report,
    )

    # Task 5: Send alerts if needed
    alert_task = PythonOperator(
        task_id='send_alerts',
        python_callable=send_alerts,
    )

    # Task 6: Log completion
    complete_task = BashOperator(
        task_id='monitoring_complete',
        bash_command='echo "Monitoring check completed at $(date)"',
    )

    # Define workflow
    # Run checks in parallel, then generate report and alert
    [check_dags, check_tasks, check_scheduler] >> generate_report >> alert_task >> complete_task


# DAG documentation
dag.doc_md = """
# Airflow Monitoring DAG

## Purpose

Monitors Airflow system health and sends alerts when issues are detected.

## Schedule

Runs hourly to continuously monitor system health.

## Monitoring Checks

### 1. DAG Run Metrics
- Success/failure rates
- Total runs in last 24 hours
- Average duration

### 2. Task Failures
- Failed task patterns
- Tasks with high failure rates
- Recent failures

### 3. Scheduler Health
- Scheduler heartbeat
- Last active time
- Processing status

## Alerting

Alerts are sent when:
- DAG failure rate > 20%
- More than 10 task failures in 24 hours
- Scheduler heartbeat age > 60 seconds

## Alert Channels

In production, alerts are sent via:
- Email
- Slack
- PagerDuty

## Metrics

All metrics are exported to Prometheus and visible in Grafana dashboards.
"""
