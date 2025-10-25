"""
DAG Tests

Tests for DAG integrity, structure, and configuration.
"""

import pytest
from datetime import datetime, timedelta
from airflow.models import DagBag, Variable
from airflow.utils.dag_cycle_tester import check_cycle
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestDAGIntegrity:
    """Test DAG loading and basic integrity."""

    @pytest.fixture(scope="class")
    def dagbag(self):
        """Load all DAGs."""
        return DagBag(dag_folder='dags/', include_examples=False)

    def test_dag_bag_import(self, dagbag):
        """Test that all DAGs load without errors."""
        assert len(dagbag.import_errors) == 0, \
            f"DAG import failures: {dagbag.import_errors}"

    def test_dags_loaded(self, dagbag):
        """Test that expected DAGs are loaded."""
        assert len(dagbag.dags) >= 2, "Expected at least 2 DAGs"

        expected_dags = ['ml_pipeline_dag', 'monitoring_dag']
        for dag_id in expected_dags:
            assert dag_id in dagbag.dags, f"DAG {dag_id} not found"

    def test_no_cycles(self, dagbag):
        """Test that DAGs have no cycles."""
        for dag_id, dag in dagbag.dags.items():
            check_cycle(dag)  # Raises if cycle detected


class TestMLPipelineDAG:
    """Test ML Pipeline DAG structure and configuration."""

    @pytest.fixture(scope="class")
    def dag(self):
        """Get ML Pipeline DAG."""
        dagbag = DagBag(dag_folder='dags/', include_examples=False)
        return dagbag.get_dag('ml_pipeline_dag')

    def test_dag_exists(self, dag):
        """Test DAG is loaded."""
        assert dag is not None

    def test_dag_tags(self, dag):
        """Test DAG has correct tags."""
        expected_tags = ['ml', 'production', 'training']
        for tag in expected_tags:
            assert tag in dag.tags, f"Missing tag: {tag}"

    def test_dag_schedule(self, dag):
        """Test DAG schedule is correct."""
        assert dag.schedule_interval == '0 2 * * *', \
            "Expected daily schedule at 2 AM"

    def test_dag_catchup(self, dag):
        """Test catchup is disabled."""
        assert dag.catchup is False

    def test_dag_default_args(self, dag):
        """Test DAG default arguments."""
        default_args = dag.default_args

        assert 'owner' in default_args
        assert 'retries' in default_args
        assert default_args['retries'] >= 1

    def test_dag_tasks(self, dag):
        """Test all expected tasks exist."""
        expected_tasks = [
            'download_data',
            'validate_data',
            'preprocess_data',
            'feature_engineering',
            'train_model',
            'evaluate_model',
            'deploy_model',
            'send_notification'
        ]

        task_ids = [task.task_id for task in dag.tasks]

        for task_id in expected_tasks:
            assert task_id in task_ids, f"Missing task: {task_id}"

    def test_task_count(self, dag):
        """Test correct number of tasks."""
        assert len(dag.tasks) == 8, f"Expected 8 tasks, got {len(dag.tasks)}"

    def test_task_dependencies(self, dag):
        """Test task dependencies are correct."""
        # Get tasks
        download = dag.get_task('download_data')
        validate = dag.get_task('validate_data')
        preprocess = dag.get_task('preprocess_data')
        feature = dag.get_task('feature_engineering')
        train = dag.get_task('train_model')
        evaluate = dag.get_task('evaluate_model')
        deploy = dag.get_task('deploy_model')
        notify = dag.get_task('send_notification')

        # Test downstream dependencies
        assert validate in download.downstream_list
        assert preprocess in validate.downstream_list
        assert feature in preprocess.downstream_list
        assert train in feature.downstream_list
        assert evaluate in train.downstream_list
        assert deploy in evaluate.downstream_list
        assert notify in deploy.downstream_list

    def test_max_active_runs(self, dag):
        """Test max active runs is set."""
        assert dag.max_active_runs == 1

    def test_task_retries(self, dag):
        """Test tasks have retry configuration."""
        for task in dag.tasks:
            # Check retries are configured
            assert hasattr(task, 'retries')
            if task.retries is not None:
                assert task.retries >= 0


class TestMonitoringDAG:
    """Test Monitoring DAG structure and configuration."""

    @pytest.fixture(scope="class")
    def dag(self):
        """Get Monitoring DAG."""
        dagbag = DagBag(dag_folder='dags/', include_examples=False)
        return dagbag.get_dag('monitoring_dag')

    def test_dag_exists(self, dag):
        """Test DAG is loaded."""
        assert dag is not None

    def test_dag_tags(self, dag):
        """Test DAG has correct tags."""
        expected_tags = ['monitoring', 'alerting', 'operations']
        for tag in expected_tags:
            assert tag in dag.tags, f"Missing tag: {tag}"

    def test_dag_schedule(self, dag):
        """Test DAG schedule is hourly."""
        assert dag.schedule_interval == '0 * * * *', \
            "Expected hourly schedule"

    def test_dag_tasks(self, dag):
        """Test all expected monitoring tasks exist."""
        expected_tasks = [
            'check_dag_runs',
            'check_task_failures',
            'check_scheduler_health',
            'generate_health_report',
            'send_alerts',
            'monitoring_complete'
        ]

        task_ids = [task.task_id for task in dag.tasks]

        for task_id in expected_tasks:
            assert task_id in task_ids, f"Missing task: {task_id}"

    def test_parallel_checks(self, dag):
        """Test that checks run in parallel."""
        check_dags = dag.get_task('check_dag_runs')
        check_tasks = dag.get_task('check_task_failures')
        check_scheduler = dag.get_task('check_scheduler_health')
        generate_report = dag.get_task('generate_health_report')

        # All checks should feed into report generation
        assert generate_report in check_dags.downstream_list
        assert generate_report in check_tasks.downstream_list
        assert generate_report in check_scheduler.downstream_list


class TestTaskConfiguration:
    """Test task-specific configurations."""

    @pytest.fixture(scope="class")
    def ml_dag(self):
        """Get ML Pipeline DAG."""
        dagbag = DagBag(dag_folder='dags/', include_examples=False)
        return dagbag.get_dag('ml_pipeline_dag')

    def test_train_task_timeout(self, ml_dag):
        """Test training task has execution timeout."""
        train_task = ml_dag.get_task('train_model')
        assert train_task.execution_timeout is not None
        assert train_task.execution_timeout.total_seconds() > 0

    def test_feature_task_timeout(self, ml_dag):
        """Test feature engineering has timeout."""
        feature_task = ml_dag.get_task('feature_engineering')
        assert feature_task.execution_timeout is not None

    def test_task_owners(self, ml_dag):
        """Test tasks have owners."""
        for task in ml_dag.tasks:
            assert task.owner is not None
            assert len(task.owner) > 0


class TestDAGDocumentation:
    """Test DAG documentation."""

    @pytest.fixture(scope="class")
    def dagbag(self):
        """Load all DAGs."""
        return DagBag(dag_folder='dags/', include_examples=False)

    def test_dag_has_description(self, dagbag):
        """Test all DAGs have descriptions."""
        for dag_id, dag in dagbag.dags.items():
            assert dag.description is not None
            assert len(dag.description) > 0

    def test_tasks_have_docs(self, dagbag):
        """Test critical tasks have documentation."""
        ml_dag = dagbag.get_dag('ml_pipeline_dag')

        critical_tasks = ['train_model', 'evaluate_model', 'deploy_model']

        for task_id in critical_tasks:
            task = ml_dag.get_task(task_id)
            # Check for doc_md or doc
            has_docs = (
                (hasattr(task, 'doc_md') and task.doc_md) or
                (hasattr(task, 'doc') and task.doc)
            )
            assert has_docs, f"Task {task_id} should have documentation"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
