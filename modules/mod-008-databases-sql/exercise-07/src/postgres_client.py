"""
PostgreSQL Client for ML Platform

Handles structured data requiring complex queries, JOINs, and ACID transactions.
Use cases:
- Training run metadata with dataset relationships
- Complex analytical queries
- Data requiring referential integrity
"""

from sqlalchemy import create_engine, text, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from contextlib import contextmanager
from typing import Generator, List, Dict, Any, Optional
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration
PG_URL = "postgresql://mluser:mlpass123@localhost:5432/ml_platform"

# Create engine with connection pooling
engine = create_engine(
    PG_URL,
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20,
    pool_timeout=30,
    pool_recycle=3600,
    pool_pre_ping=True,
    echo=False
)

# Session factory
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """
    Context manager for database sessions.
    Automatically commits on success and rolls back on exception.
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
        logger.debug("Session committed successfully")
    except Exception as e:
        session.rollback()
        logger.error(f"Session rolled back due to error: {e}")
        raise
    finally:
        session.close()


def test_connection() -> bool:
    """Test PostgreSQL connection."""
    try:
        with get_session() as session:
            result = session.execute(text("SELECT version()"))
            version = result.fetchone()[0]
            logger.info(f"✓ PostgreSQL connected: {version.split(',')[0]}")
            return True
    except Exception as e:
        logger.error(f"✗ PostgreSQL connection failed: {e}")
        return False


# ============================================================================
# DATA INSERTION
# ============================================================================

def create_dataset(
    name: str,
    description: str,
    total_rows: int,
    total_features: int,
    storage_path: str,
    created_by: str,
    tags: List[str] = None
) -> int:
    """Create a new dataset entry."""
    with get_session() as session:
        result = session.execute(
            text("""
                INSERT INTO datasets (name, description, total_rows, total_features, storage_path, created_by, tags)
                VALUES (:name, :desc, :rows, :features, :path, :created_by, :tags)
                ON CONFLICT (name) DO NOTHING
                RETURNING id
            """),
            {
                "name": name,
                "desc": description,
                "rows": total_rows,
                "features": total_features,
                "path": storage_path,
                "created_by": created_by,
                "tags": tags or []
            }
        )
        row = result.fetchone()
        if row:
            dataset_id = row[0]
            logger.info(f"✓ Created dataset: {name} (ID: {dataset_id})")
            return dataset_id
        else:
            logger.warning(f"Dataset '{name}' already exists")
            return None


def create_training_run(
    dataset_id: int,
    model_name: str,
    framework: str,
    status: str = 'running',
    accuracy: Optional[float] = None,
    loss: Optional[float] = None,
    training_time_seconds: Optional[int] = None,
    batch_size: Optional[int] = None,
    learning_rate: Optional[float] = None
) -> int:
    """Create a new training run entry."""
    with get_session() as session:
        result = session.execute(
            text("""
                INSERT INTO training_runs
                (dataset_id, model_name, framework, status, accuracy, loss, training_time_seconds, batch_size, learning_rate)
                VALUES (:did, :model, :fw, :status, :acc, :loss, :time, :batch, :lr)
                RETURNING id
            """),
            {
                "did": dataset_id,
                "model": model_name,
                "fw": framework,
                "status": status,
                "acc": accuracy,
                "loss": loss,
                "time": training_time_seconds,
                "batch": batch_size,
                "lr": learning_rate
            }
        )
        training_run_id = result.fetchone()[0]
        logger.info(f"✓ Created training run: {model_name} (ID: {training_run_id})")
        return training_run_id


def update_training_run_completion(
    training_run_id: int,
    accuracy: float,
    loss: float,
    training_time_seconds: int,
    precision_score: Optional[float] = None,
    recall: Optional[float] = None,
    f1_score: Optional[float] = None
):
    """Mark training run as completed with final metrics."""
    with get_session() as session:
        session.execute(
            text("""
                UPDATE training_runs
                SET status = 'completed',
                    accuracy = :acc,
                    loss = :loss,
                    training_time_seconds = :time,
                    precision_score = :prec,
                    recall = :recall,
                    f1_score = :f1,
                    completed_at = CURRENT_TIMESTAMP,
                    accuracy_per_hour = ROUND((:acc / (:time / 3600.0))::numeric, 4)
                WHERE id = :id
            """),
            {
                "id": training_run_id,
                "acc": accuracy,
                "loss": loss,
                "time": training_time_seconds,
                "prec": precision_score,
                "recall": recall,
                "f1": f1_score
            }
        )
        logger.info(f"✓ Updated training run {training_run_id} to completed")


# ============================================================================
# COMPLEX QUERIES
# ============================================================================

def get_top_models_by_accuracy(top_n: int = 5) -> List[Dict[str, Any]]:
    """
    SQL excels at aggregations and complex filtering.
    Get top N models by accuracy across all datasets.
    """
    with get_session() as session:
        result = session.execute(
            text("""
                SELECT
                    tr.model_name,
                    d.name as dataset_name,
                    tr.framework,
                    tr.accuracy,
                    tr.training_time_seconds,
                    tr.accuracy_per_hour,
                    tr.started_at,
                    tr.completed_at
                FROM training_runs tr
                JOIN datasets d ON tr.dataset_id = d.id
                WHERE tr.status = 'completed' AND tr.accuracy IS NOT NULL
                ORDER BY tr.accuracy DESC
                LIMIT :top_n
            """),
            {"top_n": top_n}
        )

        models = []
        for row in result:
            models.append({
                "model_name": row[0],
                "dataset_name": row[1],
                "framework": row[2],
                "accuracy": row[3],
                "training_time_seconds": row[4],
                "accuracy_per_hour": row[5],
                "started_at": row[6],
                "completed_at": row[7]
            })

        return models


def get_framework_comparison() -> List[Dict[str, Any]]:
    """
    Aggregation query: Compare frameworks by average performance.
    Demonstrates SQL's strength in analytical queries.
    """
    with get_session() as session:
        result = session.execute(
            text("""
                SELECT
                    framework,
                    COUNT(*) as total_runs,
                    ROUND(AVG(accuracy)::numeric, 4) as avg_accuracy,
                    ROUND(AVG(training_time_seconds)::numeric, 2) as avg_training_time,
                    MAX(accuracy) as best_accuracy,
                    MIN(accuracy) as worst_accuracy
                FROM training_runs
                WHERE status = 'completed'
                GROUP BY framework
                ORDER BY avg_accuracy DESC
            """)
        )

        frameworks = []
        for row in result:
            frameworks.append({
                "framework": row[0],
                "total_runs": row[1],
                "avg_accuracy": float(row[2]),
                "avg_training_time": float(row[3]),
                "best_accuracy": row[4],
                "worst_accuracy": row[5]
            })

        return frameworks


def get_dataset_training_stats(dataset_name: str) -> Dict[str, Any]:
    """
    Complex query with JOINs and aggregations.
    Get comprehensive statistics for a specific dataset.
    """
    with get_session() as session:
        result = session.execute(
            text("""
                SELECT
                    d.name,
                    d.total_rows,
                    d.total_features,
                    COUNT(tr.id) as total_experiments,
                    COUNT(CASE WHEN tr.status = 'completed' THEN 1 END) as completed_runs,
                    ROUND(AVG(CASE WHEN tr.status = 'completed' THEN tr.accuracy END)::numeric, 4) as avg_accuracy,
                    MAX(tr.accuracy) as best_accuracy,
                    ROUND(AVG(CASE WHEN tr.status = 'completed' THEN tr.training_time_seconds END)::numeric, 2) as avg_training_time
                FROM datasets d
                LEFT JOIN training_runs tr ON d.id = tr.dataset_id
                WHERE d.name = :name
                GROUP BY d.id, d.name, d.total_rows, d.total_features
            """),
            {"name": dataset_name}
        )

        row = result.fetchone()
        if row:
            return {
                "dataset_name": row[0],
                "total_rows": row[1],
                "total_features": row[2],
                "total_experiments": row[3],
                "completed_runs": row[4],
                "avg_accuracy": float(row[5]) if row[5] else None,
                "best_accuracy": row[6],
                "avg_training_time": float(row[7]) if row[7] else None
            }
        return None


def get_deployment_summary() -> List[Dict[str, Any]]:
    """
    Multi-table JOIN query.
    Get deployment information with model performance metrics.
    """
    with get_session() as session:
        result = session.execute(
            text("""
                SELECT
                    dep.deployment_name,
                    dep.environment,
                    dep.status,
                    dep.replicas,
                    dep.cpu_cores,
                    dep.memory_gb,
                    dep.gpu_count,
                    tr.model_name,
                    tr.accuracy,
                    dep.deployed_at
                FROM deployments dep
                LEFT JOIN training_runs tr ON dep.training_run_id = tr.id
                WHERE dep.status = 'active'
                ORDER BY dep.deployed_at DESC
            """)
        )

        deployments = []
        for row in result:
            deployments.append({
                "deployment_name": row[0],
                "environment": row[1],
                "status": row[2],
                "replicas": row[3],
                "cpu_cores": row[4],
                "memory_gb": row[5],
                "gpu_count": row[6],
                "model_name": row[7],
                "accuracy": row[8],
                "deployed_at": row[9]
            })

        return deployments


def search_models_by_criteria(
    min_accuracy: float = None,
    framework: str = None,
    dataset_tag: str = None
) -> List[Dict[str, Any]]:
    """
    Demonstrate complex WHERE clauses with multiple conditions.
    Shows SQL's power in filtering structured data.
    """
    filters = []
    params = {}

    if min_accuracy:
        filters.append("tr.accuracy >= :min_acc")
        params["min_acc"] = min_accuracy

    if framework:
        filters.append("tr.framework = :framework")
        params["framework"] = framework

    if dataset_tag:
        filters.append(":tag = ANY(d.tags)")
        params["tag"] = dataset_tag

    where_clause = " AND ".join(filters) if filters else "1=1"

    with get_session() as session:
        result = session.execute(
            text(f"""
                SELECT
                    tr.model_name,
                    tr.framework,
                    tr.accuracy,
                    tr.f1_score,
                    d.name as dataset_name,
                    d.tags,
                    tr.training_time_seconds,
                    tr.completed_at
                FROM training_runs tr
                JOIN datasets d ON tr.dataset_id = d.id
                WHERE tr.status = 'completed' AND {where_clause}
                ORDER BY tr.accuracy DESC
            """),
            params
        )

        models = []
        for row in result:
            models.append({
                "model_name": row[0],
                "framework": row[1],
                "accuracy": row[2],
                "f1_score": row[3],
                "dataset_name": row[4],
                "dataset_tags": row[5],
                "training_time_seconds": row[6],
                "completed_at": row[7]
            })

        return models


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_pool_status() -> Dict[str, Any]:
    """Return current connection pool statistics."""
    pool = engine.pool
    return {
        "size": pool.size(),
        "checked_in": pool.checkedin(),
        "checked_out": pool.checkedout(),
        "overflow": pool.overflow(),
        "total_connections": pool.size() + pool.overflow()
    }


if __name__ == "__main__":
    print("="*70)
    print("PostgreSQL Client - ML Platform")
    print("="*70)

    # Test connection
    test_connection()

    # Display connection pool status
    print(f"\nConnection Pool Status: {get_pool_status()}")

    # Demo: Top models
    print("\n=== Top 3 Models by Accuracy ===")
    top_models = get_top_models_by_accuracy(3)
    for model in top_models:
        print(f"  {model['model_name']:<30} | {model['framework']:<12} | Acc: {model['accuracy']:.4f} | Dataset: {model['dataset_name']}")

    # Demo: Framework comparison
    print("\n=== Framework Performance Comparison ===")
    frameworks = get_framework_comparison()
    for fw in frameworks:
        print(f"  {fw['framework']:<15} | Runs: {fw['total_runs']} | Avg Acc: {fw['avg_accuracy']:.4f} | Best: {fw['best_accuracy']:.4f}")

    # Demo: Search with criteria
    print("\n=== Models with Accuracy >= 0.95 ===")
    high_acc_models = search_models_by_criteria(min_accuracy=0.95)
    for model in high_acc_models:
        print(f"  {model['model_name']:<30} | Acc: {model['accuracy']:.4f} | {model['framework']}")

    print("\n" + "="*70)
    print("✓ PostgreSQL demonstration complete")
    print("="*70)
