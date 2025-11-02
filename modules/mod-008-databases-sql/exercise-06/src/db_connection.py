"""
Database Connection Management for ML Model Registry

Provides connection pooling, session management, and utility functions
for the ML model registry database.
"""

from sqlalchemy import create_engine, text, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from contextlib import contextmanager
from typing import Generator
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Database configuration
DATABASE_URL = "postgresql://mluser:mlregistry123@localhost:5432/model_registry"

# Create engine with connection pooling
engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=10,           # Normal pool size
    max_overflow=20,        # Extra connections under load
    pool_timeout=30,        # Wait time for connection
    pool_recycle=3600,      # Recycle connections after 1 hour
    pool_pre_ping=True,     # Verify connections before using
    echo=False,             # Set to True for SQL logging
    isolation_level="READ_COMMITTED"  # Default isolation level
)

# Session factory
SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False
)


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """
    Context manager for database sessions.

    Automatically commits on success and rolls back on exception.
    Always closes the session when done.

    Usage:
        with get_db_session() as session:
            session.execute(text("SELECT * FROM models"))
            # Auto-commits here if no exception
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
        logger.debug("Session closed")


def get_pool_status() -> dict:
    """
    Return current connection pool statistics.

    Useful for monitoring and debugging connection pool issues.

    Returns:
        dict: Pool statistics including size, checked in/out connections
    """
    pool = engine.pool
    return {
        "size": pool.size(),
        "checked_in": pool.checkedin(),
        "checked_out": pool.checkedout(),
        "overflow": pool.overflow(),
        "total_connections": pool.size() + pool.overflow()
    }


def log_pool_status():
    """Log current connection pool status."""
    status = get_pool_status()
    logger.info(f"Connection Pool Status: {status}")


def execute_with_isolation(
    query: str,
    params: dict = None,
    isolation_level: str = "READ_COMMITTED"
) -> list:
    """
    Execute a query with a specific isolation level.

    Args:
        query: SQL query to execute
        params: Query parameters
        isolation_level: Transaction isolation level

    Returns:
        list: Query results
    """
    with get_db_session() as session:
        session.execute(text(f"SET TRANSACTION ISOLATION LEVEL {isolation_level}"))
        result = session.execute(text(query), params or {})
        return result.fetchall()


def test_connection():
    """Test database connection."""
    try:
        with get_db_session() as session:
            result = session.execute(text("SELECT 1"))
            logger.info("✓ Database connection successful")
            return True
    except Exception as e:
        logger.error(f"✗ Database connection failed: {e}")
        return False


def init_database():
    """Initialize database schema."""
    import os
    schema_path = os.path.join(os.path.dirname(__file__), '..', 'sql', 'schema.sql')

    try:
        with open(schema_path, 'r') as f:
            schema_sql = f.read()

        with engine.begin() as conn:
            conn.execute(text(schema_sql))

        logger.info("✓ Database schema initialized")
        return True
    except Exception as e:
        logger.error(f"✗ Database initialization failed: {e}")
        return False


# Listen for connection events for debugging
@event.listens_for(engine, "connect")
def receive_connect(dbapi_conn, connection_record):
    """Log database connections."""
    logger.debug("New database connection established")


@event.listens_for(engine, "checkout")
def receive_checkout(dbapi_conn, connection_record, connection_proxy):
    """Log connection checkouts."""
    logger.debug("Connection checked out from pool")


if __name__ == "__main__":
    # Test the connection
    print("Testing database connection...")
    if test_connection():
        print("\nConnection pool status:")
        print(get_pool_status())
