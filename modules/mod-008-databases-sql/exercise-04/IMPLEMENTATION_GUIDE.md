# Exercise 04: SQLAlchemy ORM Integration - Implementation Guide

## Overview

This guide provides a complete implementation of Exercise 04, demonstrating how to build a production-ready Python package using SQLAlchemy ORM for the ML Model Registry. You'll create ORM models, implement repository patterns, manage migrations with Alembic, write comprehensive tests, and integrate with FastAPI.

**Technologies Used:**
- SQLAlchemy 2.0 (ORM)
- Alembic (database migrations)
- PostgreSQL (database)
- Pydantic (data validation)
- Poetry (dependency management)
- Pytest + Factory Boy (testing)
- FastAPI (API integration)
- Typer (CLI tools)

---

## Part 1: Project Setup and Configuration

### Step 1.1: Create Project Structure

```bash
# Create project directory structure
mkdir -p ml-registry-db/{src/ml_registry_db,alembic/versions,scripts,tests}
cd ml-registry-db

# Create all necessary files
touch src/ml_registry_db/{__init__.py,config.py,db.py,models.py,repositories.py,schemas.py,cli.py}
touch tests/{__init__.py,conftest.py,factories.py,test_models.py,test_repositories.py,test_integration.py}
touch scripts/{seed_data.py,create_tables.py,reset_db.py}
touch {pyproject.toml,README.md,.env.example,.gitignore}
```

**Complete Directory Structure:**
```
ml-registry-db/
├── pyproject.toml              # Poetry dependency management
├── README.md                   # Documentation
├── .env.example                # Environment template
├── .gitignore                  # Git ignore rules
├── src/
│   └── ml_registry_db/
│       ├── __init__.py         # Package initialization
│       ├── config.py           # Configuration management
│       ├── db.py               # Database session and engine
│       ├── models.py           # SQLAlchemy ORM models
│       ├── repositories.py     # Repository pattern
│       ├── schemas.py          # Pydantic models
│       └── cli.py              # CLI interface
├── alembic/                    # Database migrations
│   ├── env.py                  # Alembic environment
│   ├── alembic.ini             # Alembic config
│   └── versions/               # Migration files
├── scripts/
│   ├── seed_data.py            # Test data
│   ├── create_tables.py        # Initialize schema
│   └── reset_db.py             # Reset database
└── tests/
    ├── conftest.py             # Pytest fixtures
    ├── factories.py            # Test data generators
    ├── test_models.py          # Model tests
    ├── test_repositories.py    # Repository tests
    └── test_integration.py     # Integration tests
```

### Step 1.2: `pyproject.toml` - Dependency Management

```toml
[tool.poetry]
name = "ml-registry-db"
version = "0.1.0"
description = "ML Model Registry Database Package with SQLAlchemy ORM"
authors = ["Your Name <you@example.com>"]
readme = "README.md"
packages = [{include = "ml_registry_db", from = "src"}]

[tool.poetry.dependencies]
python = "^3.9"
SQLAlchemy = "^2.0"
alembic = "^1.12"
psycopg2-binary = "^2.9"
pydantic = "^2.5"
pydantic-settings = "^2.1"
python-dotenv = "^1.0"
typer = "^0.9"
rich = "^13.7"
fastapi = "^0.108"
uvicorn = {extras = ["standard"], version = "^0.25"}

[tool.poetry.group.dev.dependencies]
pytest = "^7.4"
pytest-asyncio = "^0.23"
pytest-cov = "^4.1"
factory-boy = "^3.3"
faker = "^21.0"
black = "^23.12"
ruff = "^0.1"
mypy = "^1.7"
httpx = "^0.26"

[tool.poetry.scripts]
ml-registry = "ml_registry_db.cli:app"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"

[tool.black]
line-length = 100
target-version = ['py39', 'py310', 'py311']

[tool.ruff]
line-length = 100
select = ["E", "F", "W", "I", "N", "UP"]

[tool.mypy]
python_version = "3.9"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_functions = ["test_*"]
addopts = "-v --cov=ml_registry_db --cov-report=term-missing"
```

### Step 1.3: `.env.example` - Environment Configuration

```bash
# Database Configuration
DATABASE_URL=postgresql://ml_user:ml_password@localhost:5432/ml_registry
TEST_DATABASE_URL=postgresql://ml_user:ml_password@localhost:5432/ml_registry_test

# Connection Pool Settings
DB_POOL_SIZE=5
DB_MAX_OVERFLOW=10
DB_POOL_TIMEOUT=30
DB_POOL_RECYCLE=3600

# Application Settings
ENVIRONMENT=development
LOG_LEVEL=INFO
DEBUG=true

# Optional: Enable query logging
SQLALCHEMY_ECHO=false
SQLALCHEMY_ECHO_POOL=false
```

### Step 1.4: `.gitignore` - Git Configuration

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environments
venv/
ENV/
env/
.venv

# Environment Variables
.env
.env.local

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# Testing
.pytest_cache/
.coverage
htmlcov/
.tox/

# Alembic
alembic.ini

# Database
*.db
*.sqlite3

# Logs
*.log
logs/

# OS
.DS_Store
Thumbs.db
```

### Step 1.5: Install Dependencies

```bash
# Install Poetry (if not already installed)
curl -sSL https://install.python-poetry.org | python3 -

# Install project dependencies
poetry install

# Activate virtual environment
poetry shell

# Copy environment template
cp .env.example .env

# Edit .env with your actual credentials
nano .env
```

---

## Part 2: Configuration Management

### Step 2.1: `src/ml_registry_db/config.py` - Settings Management

```python
"""Configuration management using Pydantic Settings."""

from functools import lru_cache
from typing import Optional

from pydantic import Field, PostgresDsn, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Database URLs
    database_url: PostgresDsn = Field(
        default="postgresql://ml_user:ml_password@localhost:5432/ml_registry",
        description="Primary database connection URL",
    )
    test_database_url: Optional[PostgresDsn] = Field(
        default=None,
        description="Test database connection URL",
    )

    # Connection Pool Settings
    db_pool_size: int = Field(default=5, ge=1, le=100)
    db_max_overflow: int = Field(default=10, ge=0, le=100)
    db_pool_timeout: int = Field(default=30, ge=1, le=300)
    db_pool_recycle: int = Field(default=3600, ge=60)

    # Application Settings
    environment: str = Field(default="development", pattern="^(development|staging|production)$")
    log_level: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    debug: bool = Field(default=False)

    # SQLAlchemy Settings
    sqlalchemy_echo: bool = Field(default=False, description="Log SQL queries")
    sqlalchemy_echo_pool: bool = Field(default=False, description="Log connection pool activity")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @field_validator("test_database_url", mode="before")
    @classmethod
    def default_test_url(cls, v: Optional[str], info) -> Optional[str]:
        """Generate test database URL if not provided."""
        if v is None and "database_url" in info.data:
            # Replace database name with test suffix
            url = str(info.data["database_url"])
            if "/" in url:
                base, db_name = url.rsplit("/", 1)
                return f"{base}/{db_name}_test"
        return v

    @property
    def sync_database_url(self) -> str:
        """Return synchronous database URL as string."""
        return str(self.database_url)

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == "development"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance.

    Returns:
        Settings: Application settings

    Example:
        >>> from ml_registry_db.config import get_settings
        >>> settings = get_settings()
        >>> print(settings.database_url)
    """
    return Settings()
```

**Key Features:**
- **Pydantic Settings**: Type-safe configuration with validation
- **Environment Variables**: Automatic loading from `.env` file
- **Connection Pool**: Configurable pool sizes and timeouts
- **Validation**: Pattern matching for environment and log level
- **Caching**: `@lru_cache` prevents repeated environment loading
- **Helper Properties**: `is_production`, `sync_database_url`

---

## Part 3: Database Connection and Session Management

### Step 3.1: `src/ml_registry_db/db.py` - Engine and Session Factory

```python
"""Database engine, session management, and connection utilities."""

import logging
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, event, pool
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from ml_registry_db.config import get_settings

logger = logging.getLogger(__name__)

# Get application settings
settings = get_settings()


# Declarative Base for ORM models
class Base(DeclarativeBase):
    """Base class for all ORM models."""

    pass


# Database Engine Configuration
def create_db_engine(echo: bool = False) -> Engine:
    """Create SQLAlchemy engine with connection pooling.

    Args:
        echo: Whether to log SQL queries (default: False)

    Returns:
        Engine: Configured SQLAlchemy engine

    Example:
        >>> engine = create_db_engine(echo=True)
        >>> print(engine.url)
    """
    engine = create_engine(
        settings.sync_database_url,
        echo=echo or settings.sqlalchemy_echo,
        echo_pool=settings.sqlalchemy_echo_pool,
        poolclass=pool.QueuePool,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        pool_timeout=settings.db_pool_timeout,
        pool_recycle=settings.db_pool_recycle,
        pool_pre_ping=True,  # Verify connections before using
        connect_args={
            "options": "-c timezone=utc",  # Force UTC timezone
        },
    )

    # Register event listeners
    _register_engine_events(engine)

    logger.info(
        f"Database engine created with pool_size={settings.db_pool_size}, "
        f"max_overflow={settings.db_max_overflow}"
    )

    return engine


def _register_engine_events(engine: Engine) -> None:
    """Register SQLAlchemy engine event listeners.

    Args:
        engine: SQLAlchemy engine instance
    """

    @event.listens_for(engine, "connect")
    def receive_connect(dbapi_conn, connection_record):
        """Log new database connections."""
        logger.debug(f"New database connection established: {id(dbapi_conn)}")

    @event.listens_for(engine, "checkout")
    def receive_checkout(dbapi_conn, connection_record, connection_proxy):
        """Log connection checkout from pool."""
        logger.debug(f"Connection checked out from pool: {id(dbapi_conn)}")

    @event.listens_for(engine, "checkin")
    def receive_checkin(dbapi_conn, connection_record):
        """Log connection return to pool."""
        logger.debug(f"Connection returned to pool: {id(dbapi_conn)}")


# Create global engine instance
engine = create_db_engine()

# Session Factory
SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,  # Prevent lazy loading errors after commit
)


def get_session() -> Generator[Session, None, None]:
    """Get database session with automatic cleanup.

    Yields:
        Session: SQLAlchemy session

    Example:
        >>> from ml_registry_db.db import get_session
        >>> with next(get_session()) as session:
        ...     models = session.query(Model).all()
    """
    session = SessionLocal()
    try:
        yield session
    except Exception as e:
        logger.error(f"Database session error: {e}")
        session.rollback()
        raise
    finally:
        session.close()


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """Context manager for database sessions.

    Yields:
        Session: SQLAlchemy session with automatic rollback on error

    Example:
        >>> from ml_registry_db.db import get_db_session
        >>> with get_db_session() as session:
        ...     model = session.query(Model).first()
        ...     print(model.name)
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as e:
        logger.error(f"Session error, rolling back: {e}")
        session.rollback()
        raise
    finally:
        session.close()


def init_db() -> None:
    """Initialize database by creating all tables.

    Example:
        >>> from ml_registry_db.db import init_db
        >>> init_db()  # Creates all tables
    """
    logger.info("Initializing database tables...")
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created successfully")


def drop_db() -> None:
    """Drop all database tables.

    WARNING: This will delete all data!

    Example:
        >>> from ml_registry_db.db import drop_db
        >>> drop_db()  # Drops all tables
    """
    logger.warning("Dropping all database tables...")
    Base.metadata.drop_all(bind=engine)
    logger.info("Database tables dropped successfully")


def reset_db() -> None:
    """Reset database by dropping and recreating all tables.

    WARNING: This will delete all data!

    Example:
        >>> from ml_registry_db.db import reset_db
        >>> reset_db()  # Drops and recreates all tables
    """
    logger.warning("Resetting database...")
    drop_db()
    init_db()
    logger.info("Database reset complete")
```

**Key Features:**
- **Connection Pooling**: QueuePool with configurable size and overflow
- **Pool Pre-Ping**: Verifies connections before use (handles stale connections)
- **Event Listeners**: Logs connection lifecycle for debugging
- **Session Management**: Multiple session access patterns (generator, context manager)
- **UTC Timezone**: Forces UTC for all database operations
- **Error Handling**: Automatic rollback on exceptions
- **Expire on Commit**: Disabled to prevent lazy loading issues

---

## Part 4: ORM Models

### Step 4.1: `src/ml_registry_db/models.py` - SQLAlchemy ORM Models

```python
"""SQLAlchemy ORM models for ML Model Registry."""

import uuid
from datetime import datetime
from typing import List, Optional

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from ml_registry_db.db import Base


# ============================================================================
# Core Models
# ============================================================================


class Team(Base):
    """Organization teams owning models."""

    __tablename__ = "teams"

    team_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    team_name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    department: Mapped[Optional[str]] = mapped_column(String(100))
    cost_center: Mapped[Optional[str]] = mapped_column(String(50))
    active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    models: Mapped[List["Model"]] = relationship(
        "Model", back_populates="team", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Team(team_id={self.team_id}, team_name='{self.team_name}')>"


class Tag(Base):
    """Tags for categorizing models."""

    __tablename__ = "tags"

    tag_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    tag_name: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Many-to-many relationship with models
    models: Mapped[List["Model"]] = relationship(
        "Model", secondary="model_tags", back_populates="tags"
    )

    def __repr__(self) -> str:
        return f"<Tag(tag_id={self.tag_id}, tag_name='{self.tag_name}')>"


class Model(Base):
    """ML models in the registry."""

    __tablename__ = "models"

    model_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    model_name: Mapped[str] = mapped_column(String(200), unique=True, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    use_case: Mapped[Optional[str]] = mapped_column(String(200))
    team_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("teams.team_id", ondelete="SET NULL")
    )
    risk_level: Mapped[str] = mapped_column(
        String(20), default="medium", nullable=False
    )
    compliance_required: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    # Relationships
    team: Mapped[Optional["Team"]] = relationship("Team", back_populates="models")
    versions: Mapped[List["ModelVersion"]] = relationship(
        "ModelVersion", back_populates="model", cascade="all, delete-orphan"
    )
    tags: Mapped[List["Tag"]] = relationship(
        "Tag", secondary="model_tags", back_populates="models"
    )

    # Constraints
    __table_args__ = (
        CheckConstraint(
            "risk_level IN ('low', 'medium', 'high', 'critical')",
            name="valid_risk_level",
        ),
        Index("idx_models_name", "model_name"),
        Index("idx_models_team", "team_id"),
        Index("idx_models_active", "is_active"),
    )

    def __repr__(self) -> str:
        return f"<Model(model_id={self.model_id}, model_name='{self.model_name}')>"


class ModelTag(Base):
    """Junction table for model-tag many-to-many relationship."""

    __tablename__ = "model_tags"

    model_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("models.model_id", ondelete="CASCADE"),
        primary_key=True,
    )
    tag_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("tags.tag_id", ondelete="CASCADE"),
        primary_key=True,
    )
    assigned_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    def __repr__(self) -> str:
        return f"<ModelTag(model_id={self.model_id}, tag_id={self.tag_id})>"


class ModelVersion(Base):
    """Versioned model artifacts."""

    __tablename__ = "model_versions"

    version_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    model_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("models.model_id", ondelete="CASCADE"),
        nullable=False,
    )
    semver: Mapped[str] = mapped_column(String(50), nullable=False)
    artifact_uri: Mapped[str] = mapped_column(String(500), nullable=False)
    framework: Mapped[str] = mapped_column(String(50), nullable=False)
    framework_version: Mapped[Optional[str]] = mapped_column(String(50))
    python_version: Mapped[Optional[str]] = mapped_column(String(20))
    status: Mapped[str] = mapped_column(String(20), default="registered", nullable=False)
    size_bytes: Mapped[Optional[int]] = mapped_column(Integer)
    checksum_sha256: Mapped[Optional[str]] = mapped_column(String(64))
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    model: Mapped["Model"] = relationship("Model", back_populates="versions")
    training_runs: Mapped[List["TrainingRun"]] = relationship(
        "TrainingRun", back_populates="version", cascade="all, delete-orphan"
    )
    deployments: Mapped[List["Deployment"]] = relationship(
        "Deployment", back_populates="version", cascade="all, delete-orphan"
    )
    metrics: Mapped[List["ModelMetric"]] = relationship(
        "ModelMetric", back_populates="version", cascade="all, delete-orphan"
    )

    # Constraints
    __table_args__ = (
        UniqueConstraint("model_id", "semver", name="unique_model_version"),
        CheckConstraint(
            "status IN ('registered', 'validated', 'production', 'archived', 'deprecated')",
            name="valid_status",
        ),
        CheckConstraint(
            "framework IN ('pytorch', 'tensorflow', 'sklearn', 'xgboost', 'jax', 'onnx', 'huggingface')",
            name="valid_framework",
        ),
        Index("idx_versions_model", "model_id"),
        Index("idx_versions_semver", "semver"),
        Index("idx_versions_status", "status"),
    )

    def __repr__(self) -> str:
        return f"<ModelVersion(version_id={self.version_id}, semver='{self.semver}')>"


# ============================================================================
# Training and Metrics Models
# ============================================================================


class TrainingRun(Base):
    """Training run metadata and results."""

    __tablename__ = "training_runs"

    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    version_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("model_versions.version_id", ondelete="CASCADE"),
        nullable=False,
    )
    run_name: Mapped[str] = mapped_column(String(200), nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="running", nullable=False)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    duration_seconds: Mapped[Optional[int]] = mapped_column(Integer)
    gpu_hours: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    cost_usd: Mapped[Optional[float]] = mapped_column(Float)
    dataset_name: Mapped[Optional[str]] = mapped_column(String(200))
    dataset_version: Mapped[Optional[str]] = mapped_column(String(50))
    hyperparameters: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    environment: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    created_by: Mapped[Optional[str]] = mapped_column(String(100))

    # Relationships
    version: Mapped["ModelVersion"] = relationship("ModelVersion", back_populates="training_runs")

    # Constraints
    __table_args__ = (
        CheckConstraint(
            "status IN ('running', 'succeeded', 'failed', 'cancelled', 'timeout')",
            name="valid_training_status",
        ),
        CheckConstraint("gpu_hours >= 0", name="positive_gpu_hours"),
        CheckConstraint("cost_usd IS NULL OR cost_usd >= 0", name="positive_cost"),
        Index("idx_training_runs_version", "version_id"),
        Index("idx_training_runs_status", "status"),
        Index("idx_training_runs_started", "started_at"),
    )

    def __repr__(self) -> str:
        return f"<TrainingRun(run_id={self.run_id}, run_name='{self.run_name}', status='{self.status}')>"


class ModelMetric(Base):
    """Model performance metrics."""

    __tablename__ = "model_metrics"

    metric_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    version_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("model_versions.version_id", ondelete="CASCADE"),
        nullable=False,
    )
    metric_name: Mapped[str] = mapped_column(String(100), nullable=False)
    metric_value: Mapped[float] = mapped_column(Float, nullable=False)
    dataset_split: Mapped[str] = mapped_column(String(20), nullable=False)
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    version: Mapped["ModelVersion"] = relationship("ModelVersion", back_populates="metrics")

    # Constraints
    __table_args__ = (
        CheckConstraint(
            "dataset_split IN ('train', 'validation', 'test', 'production')",
            name="valid_dataset_split",
        ),
        Index("idx_metrics_version", "version_id"),
        Index("idx_metrics_name", "metric_name"),
    )

    def __repr__(self) -> str:
        return f"<ModelMetric(metric_name='{self.metric_name}', value={self.metric_value})>"


# ============================================================================
# Deployment Models
# ============================================================================


class Environment(Base):
    """Deployment environments."""

    __tablename__ = "environments"

    environment_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    environment_name: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    environment_type: Mapped[str] = mapped_column(String(20), nullable=False)
    region: Mapped[Optional[str]] = mapped_column(String(50))
    kubernetes_namespace: Mapped[Optional[str]] = mapped_column(String(100))
    config: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    deployments: Mapped[List["Deployment"]] = relationship(
        "Deployment", back_populates="environment", cascade="all, delete-orphan"
    )

    # Constraints
    __table_args__ = (
        CheckConstraint(
            "environment_type IN ('development', 'staging', 'production', 'sandbox')",
            name="valid_environment_type",
        ),
    )

    def __repr__(self) -> str:
        return f"<Environment(name='{self.environment_name}', type='{self.environment_type}')>"


class Deployment(Base):
    """Model deployments to environments."""

    __tablename__ = "deployments"

    deployment_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    version_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("model_versions.version_id", ondelete="CASCADE"),
        nullable=False,
    )
    environment_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("environments.environment_id", ondelete="CASCADE"),
        nullable=False,
    )
    deployment_name: Mapped[str] = mapped_column(String(200), nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="deploying", nullable=False)
    endpoint_url: Mapped[Optional[str]] = mapped_column(String(500))
    replicas: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    cpu_request: Mapped[Optional[str]] = mapped_column(String(20))
    memory_request: Mapped[Optional[str]] = mapped_column(String(20))
    gpu_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    deployed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    deployed_by: Mapped[Optional[str]] = mapped_column(String(100))
    terminated_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    # Relationships
    version: Mapped["ModelVersion"] = relationship("ModelVersion", back_populates="deployments")
    environment: Mapped["Environment"] = relationship("Environment", back_populates="deployments")

    # Constraints
    __table_args__ = (
        CheckConstraint(
            "status IN ('deploying', 'healthy', 'degraded', 'failed', 'terminated')",
            name="valid_deployment_status",
        ),
        CheckConstraint("replicas >= 0", name="positive_replicas"),
        CheckConstraint("gpu_count >= 0", name="non_negative_gpu"),
        Index("idx_deployments_version", "version_id"),
        Index("idx_deployments_environment", "environment_id"),
        Index("idx_deployments_status", "status"),
    )

    def __repr__(self) -> str:
        return f"<Deployment(deployment_id={self.deployment_id}, name='{self.deployment_name}', status='{self.status}')>"


# ============================================================================
# Audit and History Models
# ============================================================================


class AuditLog(Base):
    """Audit trail for all registry changes."""

    __tablename__ = "audit_logs"

    log_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    entity_type: Mapped[str] = mapped_column(String(50), nullable=False)
    entity_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    action: Mapped[str] = mapped_column(String(20), nullable=False)
    actor: Mapped[str] = mapped_column(String(100), nullable=False)
    changes: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Constraints
    __table_args__ = (
        CheckConstraint(
            "action IN ('create', 'update', 'delete', 'deploy', 'rollback')",
            name="valid_action",
        ),
        Index("idx_audit_entity", "entity_type", "entity_id"),
        Index("idx_audit_timestamp", "timestamp"),
        Index("idx_audit_actor", "actor"),
    )

    def __repr__(self) -> str:
        return f"<AuditLog(entity_type='{self.entity_type}', action='{self.action}', actor='{self.actor}')>"
```

**Key Features:**
- **UUID Primary Keys**: Using `uuid.uuid4()` for distributed systems
- **Relationships**: One-to-many, many-to-many with proper cascade rules
- **Constraints**: CHECK constraints for enum-like values, UNIQUE constraints
- **Indexes**: Strategic indexes on foreign keys and frequently queried columns
- **JSONB Columns**: Flexible storage for metadata and hyperparameters
- **Timestamps**: Automatic `created_at` and `updated_at` tracking
- **Type Hints**: Full `Mapped[]` annotations for type safety
- **Soft Deletes**: Support for `is_active` flags

---

## Part 5: Pydantic Schemas for Validation

### Step 5.1: `src/ml_registry_db/schemas.py` - Data Validation Models

```python
"""Pydantic models for request/response validation."""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ============================================================================
# Base Schemas
# ============================================================================


class TeamBase(BaseModel):
    """Base team schema."""

    team_name: str = Field(..., min_length=1, max_length=100)
    department: Optional[str] = Field(None, max_length=100)
    cost_center: Optional[str] = Field(None, max_length=50)
    active: bool = True


class TeamCreate(TeamBase):
    """Schema for creating teams."""

    pass


class TeamUpdate(BaseModel):
    """Schema for updating teams."""

    team_name: Optional[str] = Field(None, min_length=1, max_length=100)
    department: Optional[str] = Field(None, max_length=100)
    cost_center: Optional[str] = Field(None, max_length=50)
    active: Optional[bool] = None


class TeamResponse(TeamBase):
    """Schema for team responses."""

    team_id: uuid.UUID
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ============================================================================
# Model Schemas
# ============================================================================


class ModelBase(BaseModel):
    """Base model schema."""

    model_name: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = None
    use_case: Optional[str] = Field(None, max_length=200)
    team_id: Optional[uuid.UUID] = None
    risk_level: str = Field(default="medium", pattern="^(low|medium|high|critical)$")
    compliance_required: bool = False
    is_active: bool = True


class ModelCreate(ModelBase):
    """Schema for creating models."""

    pass


class ModelUpdate(BaseModel):
    """Schema for updating models."""

    model_name: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = None
    use_case: Optional[str] = Field(None, max_length=200)
    team_id: Optional[uuid.UUID] = None
    risk_level: Optional[str] = Field(None, pattern="^(low|medium|high|critical)$")
    compliance_required: Optional[bool] = None
    is_active: Optional[bool] = None


class ModelResponse(ModelBase):
    """Schema for model responses."""

    model_id: uuid.UUID
    created_at: datetime
    updated_at: datetime
    version_count: Optional[int] = None

    model_config = ConfigDict(from_attributes=True)


# ============================================================================
# Model Version Schemas
# ============================================================================


class ModelVersionBase(BaseModel):
    """Base model version schema."""

    semver: str = Field(..., min_length=1, max_length=50, pattern=r"^\d+\.\d+\.\d+.*$")
    artifact_uri: str = Field(..., min_length=1, max_length=500)
    framework: str = Field(
        ..., pattern="^(pytorch|tensorflow|sklearn|xgboost|jax|onnx|huggingface)$"
    )
    framework_version: Optional[str] = Field(None, max_length=50)
    python_version: Optional[str] = Field(None, max_length=20)
    status: str = Field(
        default="registered",
        pattern="^(registered|validated|production|archived|deprecated)$",
    )
    size_bytes: Optional[int] = Field(None, ge=0)
    checksum_sha256: Optional[str] = Field(None, min_length=64, max_length=64)
    metadata_: Dict[str, Any] = Field(default_factory=dict, alias="metadata")


class ModelVersionCreate(ModelVersionBase):
    """Schema for creating model versions."""

    model_id: uuid.UUID


class ModelVersionUpdate(BaseModel):
    """Schema for updating model versions."""

    status: Optional[str] = Field(
        None, pattern="^(registered|validated|production|archived|deprecated)$"
    )
    size_bytes: Optional[int] = Field(None, ge=0)
    checksum_sha256: Optional[str] = Field(None, min_length=64, max_length=64)
    metadata_: Optional[Dict[str, Any]] = Field(None, alias="metadata")


class ModelVersionResponse(ModelVersionBase):
    """Schema for model version responses."""

    version_id: uuid.UUID
    model_id: uuid.UUID
    created_at: datetime
    model_name: Optional[str] = None

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)


# ============================================================================
# Training Run Schemas
# ============================================================================


class TrainingRunBase(BaseModel):
    """Base training run schema."""

    run_name: str = Field(..., min_length=1, max_length=200)
    status: str = Field(
        default="running", pattern="^(running|succeeded|failed|cancelled|timeout)$"
    )
    gpu_hours: float = Field(default=0.0, ge=0)
    cost_usd: Optional[float] = Field(None, ge=0)
    dataset_name: Optional[str] = Field(None, max_length=200)
    dataset_version: Optional[str] = Field(None, max_length=50)
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    environment: Dict[str, Any] = Field(default_factory=dict)
    created_by: Optional[str] = Field(None, max_length=100)


class TrainingRunCreate(TrainingRunBase):
    """Schema for creating training runs."""

    version_id: uuid.UUID


class TrainingRunUpdate(BaseModel):
    """Schema for updating training runs."""

    status: Optional[str] = Field(
        None, pattern="^(running|succeeded|failed|cancelled|timeout)$"
    )
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[int] = Field(None, ge=0)
    gpu_hours: Optional[float] = Field(None, ge=0)
    cost_usd: Optional[float] = Field(None, ge=0)


class TrainingRunResponse(TrainingRunBase):
    """Schema for training run responses."""

    run_id: uuid.UUID
    version_id: uuid.UUID
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[int] = None

    model_config = ConfigDict(from_attributes=True)


# ============================================================================
# Deployment Schemas
# ============================================================================


class DeploymentBase(BaseModel):
    """Base deployment schema."""

    deployment_name: str = Field(..., min_length=1, max_length=200)
    status: str = Field(
        default="deploying", pattern="^(deploying|healthy|degraded|failed|terminated)$"
    )
    endpoint_url: Optional[str] = Field(None, max_length=500)
    replicas: int = Field(default=1, ge=0)
    cpu_request: Optional[str] = Field(None, max_length=20)
    memory_request: Optional[str] = Field(None, max_length=20)
    gpu_count: int = Field(default=0, ge=0)
    deployed_by: Optional[str] = Field(None, max_length=100)


class DeploymentCreate(DeploymentBase):
    """Schema for creating deployments."""

    version_id: uuid.UUID
    environment_id: uuid.UUID


class DeploymentUpdate(BaseModel):
    """Schema for updating deployments."""

    status: Optional[str] = Field(
        None, pattern="^(deploying|healthy|degraded|failed|terminated)$"
    )
    replicas: Optional[int] = Field(None, ge=0)
    terminated_at: Optional[datetime] = None


class DeploymentResponse(DeploymentBase):
    """Schema for deployment responses."""

    deployment_id: uuid.UUID
    version_id: uuid.UUID
    environment_id: uuid.UUID
    deployed_at: datetime
    terminated_at: Optional[datetime] = None
    environment_name: Optional[str] = None
    model_name: Optional[str] = None
    semver: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


# ============================================================================
# Metric Schemas
# ============================================================================


class ModelMetricCreate(BaseModel):
    """Schema for creating model metrics."""

    version_id: uuid.UUID
    metric_name: str = Field(..., min_length=1, max_length=100)
    metric_value: float
    dataset_split: str = Field(..., pattern="^(train|validation|test|production)$")


class ModelMetricResponse(BaseModel):
    """Schema for metric responses."""

    metric_id: uuid.UUID
    version_id: uuid.UUID
    metric_name: str
    metric_value: float
    dataset_split: str
    recorded_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ============================================================================
# Audit Log Schemas
# ============================================================================


class AuditLogResponse(BaseModel):
    """Schema for audit log responses."""

    log_id: uuid.UUID
    entity_type: str
    entity_id: uuid.UUID
    action: str
    actor: str
    changes: Dict[str, Any]
    timestamp: datetime

    model_config = ConfigDict(from_attributes=True)
```

**Key Features:**
- **Validation**: Field-level validation with regex patterns, min/max lengths
- **Type Safety**: Full type hints for all fields
- **Separate Schemas**: Create, Update, Response schemas for each entity
- **Aliases**: Support for `metadata` vs `metadata_` field naming
- **ConfigDict**: Enables ORM mode for automatic SQLAlchemy object conversion
- **Default Values**: Sensible defaults for optional fields

---

## Part 6: Repository Pattern Implementation

### Step 6.1: `src/ml_registry_db/repositories.py` - Data Access Layer (Part 1)

```python
"""Repository pattern for data access operations."""

import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional

from sqlalchemy import and_, desc, func, or_, select
from sqlalchemy.orm import Session, joinedload, selectinload

from ml_registry_db import models, schemas

logger = logging.getLogger(__name__)


# ============================================================================
# Team Repository
# ============================================================================


class TeamRepository:
    """Repository for team operations."""

    @staticmethod
    def create(session: Session, team_data: schemas.TeamCreate) -> models.Team:
        """Create a new team.

        Args:
            session: Database session
            team_data: Team creation data

        Returns:
            Created team model

        Example:
            >>> team_data = TeamCreate(team_name="ML Platform", department="Engineering")
            >>> team = TeamRepository.create(session, team_data)
        """
        team = models.Team(**team_data.model_dump())
        session.add(team)
        session.commit()
        session.refresh(team)
        logger.info(f"Created team: {team.team_name}")
        return team

    @staticmethod
    def get_by_id(session: Session, team_id: uuid.UUID) -> Optional[models.Team]:
        """Get team by ID.

        Args:
            session: Database session
            team_id: Team UUID

        Returns:
            Team model or None if not found
        """
        return session.get(models.Team, team_id)

    @staticmethod
    def get_by_name(session: Session, team_name: str) -> Optional[models.Team]:
        """Get team by name.

        Args:
            session: Database session
            team_name: Team name

        Returns:
            Team model or None if not found
        """
        stmt = select(models.Team).where(models.Team.team_name == team_name)
        return session.scalars(stmt).first()

    @staticmethod
    def list_all(
        session: Session, active_only: bool = False, skip: int = 0, limit: int = 100
    ) -> List[models.Team]:
        """List all teams with pagination.

        Args:
            session: Database session
            active_only: Only return active teams
            skip: Number of records to skip
            limit: Maximum number of records to return

        Returns:
            List of team models
        """
        stmt = select(models.Team)
        if active_only:
            stmt = stmt.where(models.Team.active == True)
        stmt = stmt.offset(skip).limit(limit)
        return list(session.scalars(stmt).all())

    @staticmethod
    def update(
        session: Session, team_id: uuid.UUID, team_data: schemas.TeamUpdate
    ) -> Optional[models.Team]:
        """Update team.

        Args:
            session: Database session
            team_id: Team UUID
            team_data: Updated team data

        Returns:
            Updated team model or None if not found
        """
        team = session.get(models.Team, team_id)
        if not team:
            return None

        update_data = team_data.model_dump(exclude_unset=True)
        for key, value in update_data.items():
            setattr(team, key, value)

        session.commit()
        session.refresh(team)
        logger.info(f"Updated team: {team.team_name}")
        return team

    @staticmethod
    def delete(session: Session, team_id: uuid.UUID) -> bool:
        """Delete team (soft delete by setting active=False).

        Args:
            session: Database session
            team_id: Team UUID

        Returns:
            True if deleted, False if not found
        """
        team = session.get(models.Team, team_id)
        if not team:
            return False

        team.active = False
        session.commit()
        logger.info(f"Soft deleted team: {team.team_name}")
        return True


# ============================================================================
# Model Repository
# ============================================================================


class ModelRepository:
    """Repository for model operations."""

    @staticmethod
    def create(session: Session, model_data: schemas.ModelCreate) -> models.Model:
        """Create a new model.

        Args:
            session: Database session
            model_data: Model creation data

        Returns:
            Created model

        Example:
            >>> model_data = ModelCreate(
            ...     model_name="bert-classifier",
            ...     description="BERT for sentiment classification",
            ...     use_case="sentiment-analysis",
            ...     risk_level="medium"
            ... )
            >>> model = ModelRepository.create(session, model_data)
        """
        model = models.Model(**model_data.model_dump())
        session.add(model)
        session.commit()
        session.refresh(model)
        logger.info(f"Created model: {model.model_name}")
        return model

    @staticmethod
    def get_by_id(
        session: Session, model_id: uuid.UUID, include_versions: bool = False
    ) -> Optional[models.Model]:
        """Get model by ID with optional version loading.

        Args:
            session: Database session
            model_id: Model UUID
            include_versions: Whether to eagerly load versions

        Returns:
            Model or None if not found
        """
        stmt = select(models.Model).where(models.Model.model_id == model_id)

        if include_versions:
            stmt = stmt.options(selectinload(models.Model.versions))

        return session.scalars(stmt).first()

    @staticmethod
    def get_by_name(
        session: Session, model_name: str, include_versions: bool = False
    ) -> Optional[models.Model]:
        """Get model by name.

        Args:
            session: Database session
            model_name: Model name
            include_versions: Whether to eagerly load versions

        Returns:
            Model or None if not found
        """
        stmt = select(models.Model).where(models.Model.model_name == model_name)

        if include_versions:
            stmt = stmt.options(selectinload(models.Model.versions))

        return session.scalars(stmt).first()

    @staticmethod
    def list_all(
        session: Session,
        active_only: bool = False,
        team_id: Optional[uuid.UUID] = None,
        risk_level: Optional[str] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> List[models.Model]:
        """List models with filtering and pagination.

        Args:
            session: Database session
            active_only: Only return active models
            team_id: Filter by team
            risk_level: Filter by risk level
            skip: Number of records to skip
            limit: Maximum number of records to return

        Returns:
            List of models
        """
        stmt = select(models.Model)

        conditions = []
        if active_only:
            conditions.append(models.Model.is_active == True)
        if team_id:
            conditions.append(models.Model.team_id == team_id)
        if risk_level:
            conditions.append(models.Model.risk_level == risk_level)

        if conditions:
            stmt = stmt.where(and_(*conditions))

        stmt = stmt.offset(skip).limit(limit)
        return list(session.scalars(stmt).all())

    @staticmethod
    def search_by_name(
        session: Session, search_term: str, limit: int = 50
    ) -> List[models.Model]:
        """Search models by name (case-insensitive partial match).

        Args:
            session: Database session
            search_term: Search term
            limit: Maximum results

        Returns:
            List of matching models
        """
        stmt = (
            select(models.Model)
            .where(models.Model.model_name.ilike(f"%{search_term}%"))
            .limit(limit)
        )
        return list(session.scalars(stmt).all())

    @staticmethod
    def add_tag(session: Session, model_id: uuid.UUID, tag_name: str) -> bool:
        """Add tag to model (create tag if doesn't exist).

        Args:
            session: Database session
            model_id: Model UUID
            tag_name: Tag name

        Returns:
            True if added, False if model not found
        """
        model = session.get(models.Model, model_id)
        if not model:
            return False

        # Get or create tag
        tag = session.scalars(select(models.Tag).where(models.Tag.tag_name == tag_name)).first()
        if not tag:
            tag = models.Tag(tag_name=tag_name)
            session.add(tag)

        # Add tag to model if not already present
        if tag not in model.tags:
            model.tags.append(tag)
            session.commit()
            logger.info(f"Added tag '{tag_name}' to model '{model.model_name}'")

        return True

    @staticmethod
    def remove_tag(session: Session, model_id: uuid.UUID, tag_name: str) -> bool:
        """Remove tag from model.

        Args:
            session: Database session
            model_id: Model UUID
            tag_name: Tag name

        Returns:
            True if removed, False if not found
        """
        model = session.get(models.Model, model_id)
        if not model:
            return False

        tag = session.scalars(select(models.Tag).where(models.Tag.tag_name == tag_name)).first()
        if tag and tag in model.tags:
            model.tags.remove(tag)
            session.commit()
            logger.info(f"Removed tag '{tag_name}' from model '{model.model_name}'")
            return True

        return False

    @staticmethod
    def get_models_by_tag(session: Session, tag_name: str) -> List[models.Model]:
        """Get all models with a specific tag.

        Args:
            session: Database session
            tag_name: Tag name

        Returns:
            List of models with the tag
        """
        stmt = (
            select(models.Model)
            .join(models.Model.tags)
            .where(models.Tag.tag_name == tag_name)
        )
        return list(session.scalars(stmt).all())


# ============================================================================
# Model Version Repository
# ============================================================================


class ModelVersionRepository:
    """Repository for model version operations."""

    @staticmethod
    def create(
        session: Session, version_data: schemas.ModelVersionCreate
    ) -> models.ModelVersion:
        """Create a new model version.

        Args:
            session: Database session
            version_data: Version creation data

        Returns:
            Created model version

        Example:
            >>> version_data = ModelVersionCreate(
            ...     model_id=model_id,
            ...     semver="1.0.0",
            ...     artifact_uri="s3://models/bert/1.0.0/model.pt",
            ...     framework="pytorch",
            ...     framework_version="2.0.1",
            ...     python_version="3.11"
            ... )
            >>> version = ModelVersionRepository.create(session, version_data)
        """
        version = models.ModelVersion(**version_data.model_dump())
        session.add(version)
        session.commit()
        session.refresh(version)
        logger.info(f"Created model version: {version.semver} for model {version.model_id}")
        return version

    @staticmethod
    def get_by_id(session: Session, version_id: uuid.UUID) -> Optional[models.ModelVersion]:
        """Get model version by ID.

        Args:
            session: Database session
            version_id: Version UUID

        Returns:
            Model version or None if not found
        """
        return session.get(models.ModelVersion, version_id)

    @staticmethod
    def get_by_semver(
        session: Session, model_id: uuid.UUID, semver: str
    ) -> Optional[models.ModelVersion]:
        """Get model version by semantic version.

        Args:
            session: Database session
            model_id: Model UUID
            semver: Semantic version string (e.g., "1.0.0")

        Returns:
            Model version or None if not found
        """
        stmt = select(models.ModelVersion).where(
            and_(models.ModelVersion.model_id == model_id, models.ModelVersion.semver == semver)
        )
        return session.scalars(stmt).first()

    @staticmethod
    def list_by_model(
        session: Session,
        model_id: uuid.UUID,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> List[models.ModelVersion]:
        """List all versions for a model.

        Args:
            session: Database session
            model_id: Model UUID
            status: Filter by status
            limit: Maximum results

        Returns:
            List of model versions ordered by creation time (newest first)
        """
        stmt = select(models.ModelVersion).where(models.ModelVersion.model_id == model_id)

        if status:
            stmt = stmt.where(models.ModelVersion.status == status)

        stmt = stmt.order_by(desc(models.ModelVersion.created_at)).limit(limit)
        return list(session.scalars(stmt).all())

    @staticmethod
    def get_latest_version(
        session: Session, model_id: uuid.UUID, status: Optional[str] = None
    ) -> Optional[models.ModelVersion]:
        """Get the most recent version of a model.

        Args:
            session: Database session
            model_id: Model UUID
            status: Filter by status (e.g., "production")

        Returns:
            Latest model version or None if not found

        Example:
            >>> # Get latest production version
            >>> version = ModelVersionRepository.get_latest_version(
            ...     session, model_id, status="production"
            ... )
        """
        stmt = select(models.ModelVersion).where(models.ModelVersion.model_id == model_id)

        if status:
            stmt = stmt.where(models.ModelVersion.status == status)

        stmt = stmt.order_by(desc(models.ModelVersion.created_at)).limit(1)
        return session.scalars(stmt).first()

    @staticmethod
    def update_status(
        session: Session, version_id: uuid.UUID, new_status: str
    ) -> Optional[models.ModelVersion]:
        """Update version status.

        Args:
            session: Database session
            version_id: Version UUID
            new_status: New status value

        Returns:
            Updated model version or None if not found
        """
        version = session.get(models.ModelVersion, version_id)
        if not version:
            return None

        version.status = new_status
        session.commit()
        session.refresh(version)
        logger.info(f"Updated version {version.semver} status to {new_status}")
        return version

    @staticmethod
    def get_production_versions(session: Session) -> List[models.ModelVersion]:
        """Get all versions currently marked as production.

        Args:
            session: Database session

        Returns:
            List of production model versions
        """
        stmt = (
            select(models.ModelVersion)
            .where(models.ModelVersion.status == "production")
            .options(joinedload(models.ModelVersion.model))
        )
        return list(session.scalars(stmt).all())


# ============================================================================
# Training Run Repository
# ============================================================================


class TrainingRunRepository:
    """Repository for training run operations."""

    @staticmethod
    def create(session: Session, run_data: schemas.TrainingRunCreate) -> models.TrainingRun:
        """Create a new training run.

        Args:
            session: Database session
            run_data: Training run creation data

        Returns:
            Created training run

        Example:
            >>> run_data = TrainingRunCreate(
            ...     version_id=version_id,
            ...     run_name="experiment-001",
            ...     dataset_name="imdb-sentiment",
            ...     hyperparameters={"lr": 0.001, "batch_size": 32},
            ...     environment={"gpu": "A100", "cuda": "11.8"}
            ... )
            >>> run = TrainingRunRepository.create(session, run_data)
        """
        run = models.TrainingRun(**run_data.model_dump())
        session.add(run)
        session.commit()
        session.refresh(run)
        logger.info(f"Created training run: {run.run_name}")
        return run

    @staticmethod
    def get_by_id(session: Session, run_id: uuid.UUID) -> Optional[models.TrainingRun]:
        """Get training run by ID.

        Args:
            session: Database session
            run_id: Training run UUID

        Returns:
            Training run or None if not found
        """
        return session.get(models.TrainingRun, run_id)

    @staticmethod
    def list_by_version(
        session: Session,
        version_id: uuid.UUID,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> List[models.TrainingRun]:
        """List training runs for a version.

        Args:
            session: Database session
            version_id: Model version UUID
            status: Filter by status
            limit: Maximum results

        Returns:
            List of training runs ordered by start time (newest first)
        """
        stmt = select(models.TrainingRun).where(models.TrainingRun.version_id == version_id)

        if status:
            stmt = stmt.where(models.TrainingRun.status == status)

        stmt = stmt.order_by(desc(models.TrainingRun.started_at)).limit(limit)
        return list(session.scalars(stmt).all())

    @staticmethod
    def list_by_model(
        session: Session,
        model_id: uuid.UUID,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> List[models.TrainingRun]:
        """List training runs for a model across all versions.

        Args:
            session: Database session
            model_id: Model UUID
            status: Filter by status
            limit: Maximum results

        Returns:
            List of training runs
        """
        stmt = (
            select(models.TrainingRun)
            .join(models.ModelVersion)
            .where(models.ModelVersion.model_id == model_id)
        )

        if status:
            stmt = stmt.where(models.TrainingRun.status == status)

        stmt = stmt.order_by(desc(models.TrainingRun.started_at)).limit(limit)
        return list(session.scalars(stmt).all())

    @staticmethod
    def update_status(
        session: Session,
        run_id: uuid.UUID,
        status: str,
        completed_at: Optional[datetime] = None,
        duration_seconds: Optional[int] = None,
    ) -> Optional[models.TrainingRun]:
        """Update training run status and completion details.

        Args:
            session: Database session
            run_id: Training run UUID
            status: New status
            completed_at: Completion timestamp
            duration_seconds: Duration in seconds

        Returns:
            Updated training run or None if not found
        """
        run = session.get(models.TrainingRun, run_id)
        if not run:
            return None

        run.status = status
        if completed_at:
            run.completed_at = completed_at
        if duration_seconds:
            run.duration_seconds = duration_seconds

        session.commit()
        session.refresh(run)
        logger.info(f"Updated training run {run.run_name} status to {status}")
        return run

    @staticmethod
    def get_cost_summary(session: Session, model_id: Optional[uuid.UUID] = None) -> Dict:
        """Get cost summary for training runs.

        Args:
            session: Database session
            model_id: Optional model UUID to filter by

        Returns:
            Dictionary with total_cost_usd, total_gpu_hours, run_count

        Example:
            >>> summary = TrainingRunRepository.get_cost_summary(session)
            >>> print(f"Total cost: ${summary['total_cost_usd']:.2f}")
        """
        stmt = select(
            func.sum(models.TrainingRun.cost_usd).label("total_cost_usd"),
            func.sum(models.TrainingRun.gpu_hours).label("total_gpu_hours"),
            func.count(models.TrainingRun.run_id).label("run_count"),
        )

        if model_id:
            stmt = stmt.join(models.ModelVersion).where(models.ModelVersion.model_id == model_id)

        result = session.execute(stmt).first()

        return {
            "total_cost_usd": float(result.total_cost_usd or 0),
            "total_gpu_hours": float(result.total_gpu_hours or 0),
            "run_count": int(result.run_count or 0),
        }


# ============================================================================
# Deployment Repository
# ============================================================================


class DeploymentRepository:
    """Repository for deployment operations."""

    @staticmethod
    def create(session: Session, deployment_data: schemas.DeploymentCreate) -> models.Deployment:
        """Create a new deployment.

        Args:
            session: Database session
            deployment_data: Deployment creation data

        Returns:
            Created deployment

        Example:
            >>> deployment_data = DeploymentCreate(
            ...     version_id=version_id,
            ...     environment_id=env_id,
            ...     deployment_name="bert-prod-01",
            ...     replicas=3,
            ...     cpu_request="1000m",
            ...     memory_request="2Gi",
            ...     deployed_by="john.doe"
            ... )
            >>> deployment = DeploymentRepository.create(session, deployment_data)
        """
        deployment = models.Deployment(**deployment_data.model_dump())
        session.add(deployment)
        session.commit()
        session.refresh(deployment)
        logger.info(f"Created deployment: {deployment.deployment_name}")
        return deployment

    @staticmethod
    def get_by_id(session: Session, deployment_id: uuid.UUID) -> Optional[models.Deployment]:
        """Get deployment by ID.

        Args:
            session: Database session
            deployment_id: Deployment UUID

        Returns:
            Deployment or None if not found
        """
        return session.get(models.Deployment, deployment_id)

    @staticmethod
    def list_by_environment(
        session: Session,
        environment_id: uuid.UUID,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> List[models.Deployment]:
        """List deployments in an environment.

        Args:
            session: Database session
            environment_id: Environment UUID
            status: Filter by status
            limit: Maximum results

        Returns:
            List of deployments
        """
        stmt = select(models.Deployment).where(
            models.Deployment.environment_id == environment_id
        )

        if status:
            stmt = stmt.where(models.Deployment.status == status)

        stmt = stmt.order_by(desc(models.Deployment.deployed_at)).limit(limit)
        return list(session.scalars(stmt).all())

    @staticmethod
    def list_by_version(
        session: Session,
        version_id: uuid.UUID,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> List[models.Deployment]:
        """List deployments for a model version.

        Args:
            session: Database session
            version_id: Model version UUID
            status: Filter by status
            limit: Maximum results

        Returns:
            List of deployments
        """
        stmt = select(models.Deployment).where(models.Deployment.version_id == version_id)

        if status:
            stmt = stmt.where(models.Deployment.status == status)

        stmt = stmt.order_by(desc(models.Deployment.deployed_at)).limit(limit)
        return list(session.scalars(stmt).all())

    @staticmethod
    def get_active_deployments(session: Session) -> List[models.Deployment]:
        """Get all active deployments (healthy or degraded).

        Args:
            session: Database session

        Returns:
            List of active deployments
        """
        stmt = (
            select(models.Deployment)
            .where(models.Deployment.status.in_(["healthy", "degraded"]))
            .options(
                joinedload(models.Deployment.version).joinedload(models.ModelVersion.model),
                joinedload(models.Deployment.environment),
            )
        )
        return list(session.scalars(stmt).all())

    @staticmethod
    def update_status(
        session: Session, deployment_id: uuid.UUID, status: str
    ) -> Optional[models.Deployment]:
        """Update deployment status.

        Args:
            session: Database session
            deployment_id: Deployment UUID
            status: New status

        Returns:
            Updated deployment or None if not found
        """
        deployment = session.get(models.Deployment, deployment_id)
        if not deployment:
            return None

        deployment.status = status
        if status == "terminated":
            deployment.terminated_at = datetime.utcnow()

        session.commit()
        session.refresh(deployment)
        logger.info(f"Updated deployment {deployment.deployment_name} status to {status}")
        return deployment

    @staticmethod
    def scale_deployment(
        session: Session, deployment_id: uuid.UUID, replicas: int
    ) -> Optional[models.Deployment]:
        """Scale deployment replicas.

        Args:
            session: Database session
            deployment_id: Deployment UUID
            replicas: New replica count

        Returns:
            Updated deployment or None if not found
        """
        deployment = session.get(models.Deployment, deployment_id)
        if not deployment:
            return None

        deployment.replicas = replicas
        session.commit()
        session.refresh(deployment)
        logger.info(f"Scaled deployment {deployment.deployment_name} to {replicas} replicas")
        return deployment


# ============================================================================
# Metric Repository
# ============================================================================


class MetricRepository:
    """Repository for model metric operations."""

    @staticmethod
    def create(session: Session, metric_data: schemas.ModelMetricCreate) -> models.ModelMetric:
        """Record a new model metric.

        Args:
            session: Database session
            metric_data: Metric creation data

        Returns:
            Created metric

        Example:
            >>> metric_data = ModelMetricCreate(
            ...     version_id=version_id,
            ...     metric_name="accuracy",
            ...     metric_value=0.9542,
            ...     dataset_split="test"
            ... )
            >>> metric = MetricRepository.create(session, metric_data)
        """
        metric = models.ModelMetric(**metric_data.model_dump())
        session.add(metric)
        session.commit()
        session.refresh(metric)
        logger.info(f"Recorded metric: {metric.metric_name}={metric.metric_value}")
        return metric

    @staticmethod
    def list_by_version(
        session: Session,
        version_id: uuid.UUID,
        metric_name: Optional[str] = None,
        dataset_split: Optional[str] = None,
    ) -> List[models.ModelMetric]:
        """List metrics for a model version.

        Args:
            session: Database session
            version_id: Model version UUID
            metric_name: Filter by metric name
            dataset_split: Filter by dataset split

        Returns:
            List of metrics
        """
        stmt = select(models.ModelMetric).where(models.ModelMetric.version_id == version_id)

        if metric_name:
            stmt = stmt.where(models.ModelMetric.metric_name == metric_name)
        if dataset_split:
            stmt = stmt.where(models.ModelMetric.dataset_split == dataset_split)

        stmt = stmt.order_by(desc(models.ModelMetric.recorded_at))
        return list(session.scalars(stmt).all())

    @staticmethod
    def get_latest_metrics(
        session: Session, version_id: uuid.UUID
    ) -> Dict[str, models.ModelMetric]:
        """Get latest metric values for each metric name.

        Args:
            session: Database session
            version_id: Model version UUID

        Returns:
            Dictionary mapping metric names to latest metric models

        Example:
            >>> metrics = MetricRepository.get_latest_metrics(session, version_id)
            >>> print(f"Accuracy: {metrics['accuracy'].metric_value}")
        """
        subquery = (
            select(
                models.ModelMetric.metric_name,
                func.max(models.ModelMetric.recorded_at).label("max_recorded"),
            )
            .where(models.ModelMetric.version_id == version_id)
            .group_by(models.ModelMetric.metric_name)
            .subquery()
        )

        stmt = (
            select(models.ModelMetric)
            .join(
                subquery,
                and_(
                    models.ModelMetric.metric_name == subquery.c.metric_name,
                    models.ModelMetric.recorded_at == subquery.c.max_recorded,
                ),
            )
            .where(models.ModelMetric.version_id == version_id)
        )

        metrics = session.scalars(stmt).all()
        return {metric.metric_name: metric for metric in metrics}
```

### Step 6.2: `src/ml_registry_db/repositories.py` - Data Access Layer (Part 2)

```python
# ============================================================================
# Audit Log Repository
# ============================================================================


class AuditLogRepository:
    """Repository for audit log operations."""

    @staticmethod
    def create_log(
        session: Session,
        entity_type: str,
        entity_id: uuid.UUID,
        action: str,
        actor: str,
        changes: Dict = None,
    ) -> models.AuditLog:
        """Create an audit log entry.

        Args:
            session: Database session
            entity_type: Type of entity (e.g., "model", "deployment")
            entity_id: Entity UUID
            action: Action performed (create, update, delete, deploy, rollback)
            actor: User who performed action
            changes: Dictionary of changes made

        Returns:
            Created audit log

        Example:
            >>> log = AuditLogRepository.create_log(
            ...     session,
            ...     entity_type="model",
            ...     entity_id=model.model_id,
            ...     action="create",
            ...     actor="john.doe",
            ...     changes={"model_name": "bert-classifier", "risk_level": "medium"}
            ... )
        """
        log = models.AuditLog(
            entity_type=entity_type,
            entity_id=entity_id,
            action=action,
            actor=actor,
            changes=changes or {},
        )
        session.add(log)
        session.commit()
        session.refresh(log)
        logger.info(f"Audit log: {actor} performed {action} on {entity_type} {entity_id}")
        return log

    @staticmethod
    def list_by_entity(
        session: Session, entity_type: str, entity_id: uuid.UUID, limit: int = 100
    ) -> List[models.AuditLog]:
        """List audit logs for a specific entity.

        Args:
            session: Database session
            entity_type: Entity type
            entity_id: Entity UUID
            limit: Maximum results

        Returns:
            List of audit logs ordered by timestamp (newest first)
        """
        stmt = (
            select(models.AuditLog)
            .where(
                and_(
                    models.AuditLog.entity_type == entity_type,
                    models.AuditLog.entity_id == entity_id,
                )
            )
            .order_by(desc(models.AuditLog.timestamp))
            .limit(limit)
        )
        return list(session.scalars(stmt).all())

    @staticmethod
    def list_by_actor(session: Session, actor: str, limit: int = 100) -> List[models.AuditLog]:
        """List audit logs for a specific actor (user).

        Args:
            session: Database session
            actor: Actor name
            limit: Maximum results

        Returns:
            List of audit logs
        """
        stmt = (
            select(models.AuditLog)
            .where(models.AuditLog.actor == actor)
            .order_by(desc(models.AuditLog.timestamp))
            .limit(limit)
        )
        return list(session.scalars(stmt).all())

    @staticmethod
    def list_recent(session: Session, limit: int = 100) -> List[models.AuditLog]:
        """List recent audit logs across all entities.

        Args:
            session: Database session
            limit: Maximum results

        Returns:
            List of audit logs ordered by timestamp (newest first)
        """
        stmt = select(models.AuditLog).order_by(desc(models.AuditLog.timestamp)).limit(limit)
        return list(session.scalars(stmt).all())


# ============================================================================
# Environment Repository
# ============================================================================


class EnvironmentRepository:
    """Repository for environment operations."""

    @staticmethod
    def create(
        session: Session,
        environment_name: str,
        environment_type: str,
        region: Optional[str] = None,
        kubernetes_namespace: Optional[str] = None,
        config: Dict = None,
    ) -> models.Environment:
        """Create a new environment.

        Args:
            session: Database session
            environment_name: Environment name
            environment_type: Environment type (development, staging, production, sandbox)
            region: Cloud region
            kubernetes_namespace: K8s namespace
            config: Additional configuration

        Returns:
            Created environment

        Example:
            >>> env = EnvironmentRepository.create(
            ...     session,
            ...     environment_name="production-us-east",
            ...     environment_type="production",
            ...     region="us-east-1",
            ...     kubernetes_namespace="ml-prod"
            ... )
        """
        env = models.Environment(
            environment_name=environment_name,
            environment_type=environment_type,
            region=region,
            kubernetes_namespace=kubernetes_namespace,
            config=config or {},
        )
        session.add(env)
        session.commit()
        session.refresh(env)
        logger.info(f"Created environment: {environment_name}")
        return env

    @staticmethod
    def get_by_name(session: Session, environment_name: str) -> Optional[models.Environment]:
        """Get environment by name.

        Args:
            session: Database session
            environment_name: Environment name

        Returns:
            Environment or None if not found
        """
        stmt = select(models.Environment).where(
            models.Environment.environment_name == environment_name
        )
        return session.scalars(stmt).first()

    @staticmethod
    def list_by_type(
        session: Session, environment_type: str, active_only: bool = True
    ) -> List[models.Environment]:
        """List environments by type.

        Args:
            session: Database session
            environment_type: Environment type
            active_only: Only return active environments

        Returns:
            List of environments
        """
        stmt = select(models.Environment).where(
            models.Environment.environment_type == environment_type
        )
        if active_only:
            stmt = stmt.where(models.Environment.is_active == True)

        return list(session.scalars(stmt).all())


# ============================================================================
# Complex Query Examples
# ============================================================================


class AnalyticsRepository:
    """Repository for complex analytical queries."""

    @staticmethod
    def get_model_summary(session: Session, model_id: uuid.UUID) -> Dict:
        """Get comprehensive summary for a model.

        Args:
            session: Database session
            model_id: Model UUID

        Returns:
            Dictionary with version_count, training_run_count, deployment_count,
            latest_version, production_version, total_gpu_hours, total_cost

        Example:
            >>> summary = AnalyticsRepository.get_model_summary(session, model_id)
            >>> print(f"Model has {summary['version_count']} versions")
        """
        # Count versions
        version_count = session.scalar(
            select(func.count(models.ModelVersion.version_id)).where(
                models.ModelVersion.model_id == model_id
            )
        )

        # Count training runs
        training_run_count = session.scalar(
            select(func.count(models.TrainingRun.run_id))
            .join(models.ModelVersion)
            .where(models.ModelVersion.model_id == model_id)
        )

        # Count deployments
        deployment_count = session.scalar(
            select(func.count(models.Deployment.deployment_id))
            .join(models.ModelVersion)
            .where(models.ModelVersion.model_id == model_id)
        )

        # Get latest version
        latest_version = session.scalars(
            select(models.ModelVersion)
            .where(models.ModelVersion.model_id == model_id)
            .order_by(desc(models.ModelVersion.created_at))
            .limit(1)
        ).first()

        # Get production version
        production_version = session.scalars(
            select(models.ModelVersion).where(
                and_(
                    models.ModelVersion.model_id == model_id,
                    models.ModelVersion.status == "production",
                )
            )
        ).first()

        # Get training costs
        cost_result = session.execute(
            select(
                func.sum(models.TrainingRun.gpu_hours),
                func.sum(models.TrainingRun.cost_usd),
            )
            .join(models.ModelVersion)
            .where(models.ModelVersion.model_id == model_id)
        ).first()

        return {
            "version_count": version_count or 0,
            "training_run_count": training_run_count or 0,
            "deployment_count": deployment_count or 0,
            "latest_version": latest_version.semver if latest_version else None,
            "production_version": production_version.semver if production_version else None,
            "total_gpu_hours": float(cost_result[0] or 0),
            "total_cost_usd": float(cost_result[1] or 0),
        }

    @staticmethod
    def get_team_metrics(session: Session, team_id: uuid.UUID) -> Dict:
        """Get metrics for a team.

        Args:
            session: Database session
            team_id: Team UUID

        Returns:
            Dictionary with model_count, version_count, training_run_count,
            deployment_count, total_gpu_hours, total_cost
        """
        # Count models
        model_count = session.scalar(
            select(func.count(models.Model.model_id)).where(models.Model.team_id == team_id)
        )

        # Count versions
        version_count = session.scalar(
            select(func.count(models.ModelVersion.version_id))
            .join(models.Model)
            .where(models.Model.team_id == team_id)
        )

        # Count training runs
        training_run_count = session.scalar(
            select(func.count(models.TrainingRun.run_id))
            .join(models.ModelVersion)
            .join(models.Model)
            .where(models.Model.team_id == team_id)
        )

        # Count deployments
        deployment_count = session.scalar(
            select(func.count(models.Deployment.deployment_id))
            .join(models.ModelVersion)
            .join(models.Model)
            .where(models.Model.team_id == team_id)
        )

        # Get training costs
        cost_result = session.execute(
            select(
                func.sum(models.TrainingRun.gpu_hours),
                func.sum(models.TrainingRun.cost_usd),
            )
            .join(models.ModelVersion)
            .join(models.Model)
            .where(models.Model.team_id == team_id)
        ).first()

        return {
            "model_count": model_count or 0,
            "version_count": version_count or 0,
            "training_run_count": training_run_count or 0,
            "deployment_count": deployment_count or 0,
            "total_gpu_hours": float(cost_result[0] or 0),
            "total_cost_usd": float(cost_result[1] or 0),
        }

    @staticmethod
    def get_deployment_status_summary(session: Session) -> Dict[str, int]:
        """Get count of deployments by status.

        Args:
            session: Database session

        Returns:
            Dictionary mapping status to count

        Example:
            >>> summary = AnalyticsRepository.get_deployment_status_summary(session)
            >>> print(f"Healthy: {summary.get('healthy', 0)}")
        """
        result = session.execute(
            select(models.Deployment.status, func.count(models.Deployment.deployment_id)).group_by(
                models.Deployment.status
            )
        ).all()

        return {status: count for status, count in result}

    @staticmethod
    def find_models_needing_retraining(
        session: Session, days_threshold: int = 90
    ) -> List[models.Model]:
        """Find models that haven't been retrained recently.

        Args:
            session: Database session
            days_threshold: Number of days since last training

        Returns:
            List of models needing retraining
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days_threshold)

        # Find models with no recent training runs
        subquery = (
            select(models.ModelVersion.model_id)
            .join(models.TrainingRun)
            .where(models.TrainingRun.started_at >= cutoff_date)
            .distinct()
            .subquery()
        )

        stmt = select(models.Model).where(
            and_(models.Model.is_active == True, ~models.Model.model_id.in_(select(subquery)))
        )

        return list(session.scalars(stmt).all())


# ============================================================================
# Repository Facade
# ============================================================================


class RepositoryFacade:
    """Unified repository facade for convenient access to all repositories.

    Example:
        >>> with get_db_session() as session:
        ...     repos = RepositoryFacade()
        ...     model = repos.models.create(session, model_data)
        ...     versions = repos.versions.list_by_model(session, model.model_id)
    """

    def __init__(self):
        self.teams = TeamRepository()
        self.models = ModelRepository()
        self.versions = ModelVersionRepository()
        self.training_runs = TrainingRunRepository()
        self.deployments = DeploymentRepository()
        self.metrics = MetricRepository()
        self.audit_logs = AuditLogRepository()
        self.environments = EnvironmentRepository()
        self.analytics = AnalyticsRepository()
```

**Key Features:**
- **Repository Pattern**: Encapsulates all database access logic
- **Type Safety**: Full type hints for parameters and return values
- **Eager Loading**: Strategic use of `joinedload` and `selectinload`
- **Filtering and Pagination**: Consistent `skip`/`limit` parameters
- **Complex Queries**: Window functions, aggregations, subqueries
- **Audit Logging**: Built-in audit trail support
- **Analytics**: Business intelligence queries for dashboards
- **Error Handling**: Proper None returns for not-found cases
- **Logging**: Info/debug logs for all operations

---

---

## Part 7: Database Migrations with Alembic

### Step 7.1: Initialize Alembic

```bash
# Initialize Alembic in the project
cd ml-registry-db
poetry run alembic init alembic

# This creates:
# - alembic/ directory
# - alembic.ini configuration file
# - alembic/env.py environment script
```

### Step 7.2: Configure `alembic.ini`

Edit `alembic.ini` to use environment variables:

```ini
[alembic]
# Path to migration scripts
script_location = alembic

# Version path specification
version_path_separator = os
file_template = %%(year)d_%%(month).2d_%%(day).2d_%%(hour).2d%%(minute).2d-%%(rev)s_%%(slug)s

# Timezone for timestamps
timezone = UTC

# Output encoding
output_encoding = utf-8

# IMPORTANT: Comment out sqlalchemy.url - we'll set it programmatically in env.py
# sqlalchemy.url = postgresql://user:pass@localhost/db

[post_write_hooks]
# Black formatter hook
hooks = black
black.type = console_scripts
black.entrypoint = black
black.options = -l 100 REVISION_SCRIPT_FILENAME

[loggers]
keys = root,sqlalchemy,alembic

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARN
handlers = console
qualname =

[logger_sqlalchemy]
level = WARN
handlers =
qualname = sqlalchemy.engine

[logger_alembic]
level = INFO
handlers =
qualname = alembic

[handler_console]
class = StreamHandler
args = (sys.stderr,)
level = NOTSET
formatter = generic

[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
datefmt = %H:%M:%S
```

### Step 7.3: Configure `alembic/env.py`

Replace `alembic/env.py` with:

```python
"""Alembic environment configuration."""

import os
import sys
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

# Add src/ to path so we can import our package
sys.path.insert(0, os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "src")))

from ml_registry_db.config import get_settings
from ml_registry_db.db import Base
from ml_registry_db import models  # Import all models to register them with Base

# This is the Alembic Config object
config = context.config

# Interpret the config file for Python logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Get database URL from our settings
settings = get_settings()
config.set_main_option("sqlalchemy.url", str(settings.database_url))

# Add your model's MetaData object here for 'autogenerate' support
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL and not an Engine.
    Calls to context.execute() will emit the SQL to a script output.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,  # Detect column type changes
        compare_server_default=True,  # Detect default value changes
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine and associate a connection with the context.
    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,  # Don't use connection pooling for migrations
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
```

### Step 7.4: Create Initial Migration

```bash
# Generate migration from models
poetry run alembic revision --autogenerate -m "initial schema with all models"

# This creates: alembic/versions/YYYY_MM_DD_HHMM-<revision>_initial_schema_with_all_models.py
```

**Example Generated Migration:**

```python
"""initial schema with all models

Revision ID: abc123def456
Revises:
Create Date: 2025-01-15 10:30:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = 'abc123def456'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Apply migration."""
    # Create teams table
    op.create_table(
        'teams',
        sa.Column('team_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('team_name', sa.String(length=100), nullable=False),
        sa.Column('department', sa.String(length=100), nullable=True),
        sa.Column('cost_center', sa.String(length=50), nullable=True),
        sa.Column('active', sa.Boolean(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('team_id'),
        sa.UniqueConstraint('team_name')
    )

    # Create tags table
    op.create_table(
        'tags',
        sa.Column('tag_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('tag_name', sa.String(length=50), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('tag_id'),
        sa.UniqueConstraint('tag_name')
    )

    # Create models table
    op.create_table(
        'models',
        sa.Column('model_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('model_name', sa.String(length=200), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('use_case', sa.String(length=200), nullable=True),
        sa.Column('team_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('risk_level', sa.String(length=20), nullable=False),
        sa.Column('compliance_required', sa.Boolean(), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.CheckConstraint("risk_level IN ('low', 'medium', 'high', 'critical')", name='valid_risk_level'),
        sa.ForeignKeyConstraint(['team_id'], ['teams.team_id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('model_id'),
        sa.UniqueConstraint('model_name')
    )
    op.create_index('idx_models_name', 'models', ['model_name'])
    op.create_index('idx_models_team', 'models', ['team_id'])
    op.create_index('idx_models_active', 'models', ['is_active'])

    # ... (continues for all tables)


def downgrade() -> None:
    """Revert migration."""
    op.drop_table('deployments')
    op.drop_table('training_runs')
    op.drop_table('model_metrics')
    op.drop_table('model_versions')
    op.drop_table('model_tags')
    op.drop_table('models')
    op.drop_table('environments')
    op.drop_table('audit_logs')
    op.drop_table('tags')
    op.drop_table('teams')
```

### Step 7.5: Run Migrations

```bash
# Apply migrations to database
poetry run alembic upgrade head

# Check current revision
poetry run alembic current

# View migration history
poetry run alembic history --verbose

# Rollback one revision
poetry run alembic downgrade -1

# Rollback to specific revision
poetry run alembic downgrade abc123def456

# Rollback all migrations
poetry run alembic downgrade base
```

### Step 7.6: Common Migration Operations

**Add a new column:**

```bash
# Create empty migration
poetry run alembic revision -m "add_last_login_to_teams"
```

Edit the generated file:

```python
def upgrade() -> None:
    op.add_column('teams', sa.Column('last_login', sa.DateTime(timezone=True), nullable=True))

def downgrade() -> None:
    op.drop_column('teams', 'last_login')
```

**Add an index:**

```python
def upgrade() -> None:
    op.create_index('idx_models_use_case', 'models', ['use_case'])

def downgrade() -> None:
    op.drop_index('idx_models_use_case', 'models')
```

**Modify column type:**

```python
def upgrade() -> None:
    op.alter_column('models', 'description',
                   existing_type=sa.Text(),
                   type_=sa.String(length=1000),
                   existing_nullable=True)

def downgrade() -> None:
    op.alter_column('models', 'description',
                   existing_type=sa.String(length=1000),
                   type_=sa.Text(),
                   existing_nullable=True)
```

---

## Part 8: CLI Tools with Typer

### Step 8.1: `src/ml_registry_db/cli.py` - Command-Line Interface

```python
"""Command-line interface for ML Registry database management."""

import sys
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from ml_registry_db.config import get_settings
from ml_registry_db.db import engine, init_db, drop_db, reset_db, get_db_session
from ml_registry_db import models
from ml_registry_db.repositories import RepositoryFacade

app = typer.Typer(help="ML Registry Database Management CLI")
console = Console()


@app.command()
def init() -> None:
    """Initialize database by creating all tables."""
    console.print("[yellow]Initializing database...[/yellow]")
    init_db()
    console.print("[green]✓ Database initialized successfully[/green]")


@app.command()
def drop() -> None:
    """Drop all database tables. WARNING: This deletes all data!"""
    confirm = typer.confirm("⚠️  Are you sure you want to drop all tables? This cannot be undone!")
    if not confirm:
        console.print("[yellow]Aborted[/yellow]")
        raise typer.Abort()

    console.print("[red]Dropping all tables...[/red]")
    drop_db()
    console.print("[green]✓ All tables dropped[/green]")


@app.command()
def reset() -> None:
    """Reset database by dropping and recreating all tables. WARNING: This deletes all data!"""
    confirm = typer.confirm("⚠️  Are you sure you want to reset the database? All data will be lost!")
    if not confirm:
        console.print("[yellow]Aborted[/yellow]")
        raise typer.Abort()

    console.print("[red]Resetting database...[/red]")
    reset_db()
    console.print("[green]✓ Database reset complete[/green]")


@app.command()
def seed() -> None:
    """Seed database with sample data."""
    from scripts.seed_data import seed_database

    console.print("[yellow]Seeding database with sample data...[/yellow]")
    seed_database()
    console.print("[green]✓ Database seeded successfully[/green]")


@app.command()
def info() -> None:
    """Display database connection information."""
    settings = get_settings()

    table = Table(title="Database Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    # Mask password in URL
    url_str = str(settings.database_url)
    if "@" in url_str:
        before_at, after_at = url_str.split("@", 1)
        if ":" in before_at:
            scheme_user, password = before_at.rsplit(":", 1)
            masked_url = f"{scheme_user}:****@{after_at}"
        else:
            masked_url = url_str
    else:
        masked_url = url_str

    table.add_row("Database URL", masked_url)
    table.add_row("Environment", settings.environment)
    table.add_row("Pool Size", str(settings.db_pool_size))
    table.add_row("Max Overflow", str(settings.db_max_overflow))
    table.add_row("Pool Timeout", f"{settings.db_pool_timeout}s")
    table.add_row("Pool Recycle", f"{settings.db_pool_recycle}s")

    console.print(table)


@app.command()
def stats() -> None:
    """Display database statistics."""
    with get_db_session() as session:
        repos = RepositoryFacade()

        # Count entities
        team_count = session.query(models.Team).count()
        model_count = session.query(models.Model).count()
        version_count = session.query(models.ModelVersion).count()
        run_count = session.query(models.TrainingRun).count()
        deployment_count = session.query(models.Deployment).count()

        table = Table(title="Database Statistics")
        table.add_column("Entity", style="cyan")
        table.add_column("Count", style="green", justify="right")

        table.add_row("Teams", str(team_count))
        table.add_row("Models", str(model_count))
        table.add_row("Model Versions", str(version_count))
        table.add_row("Training Runs", str(run_count))
        table.add_row("Deployments", str(deployment_count))

        console.print(table)


@app.command()
def list_models(
    active_only: bool = typer.Option(False, "--active", help="Show only active models"),
    limit: int = typer.Option(20, "--limit", "-n", help="Maximum number of results"),
) -> None:
    """List models in the registry."""
    with get_db_session() as session:
        repos = RepositoryFacade()
        models_list = repos.models.list_all(session, active_only=active_only, limit=limit)

        if not models_list:
            console.print("[yellow]No models found[/yellow]")
            return

        table = Table(title=f"Models ({'Active Only' if active_only else 'All'})")
        table.add_column("Model Name", style="cyan")
        table.add_column("Team", style="blue")
        table.add_column("Risk Level", style="yellow")
        table.add_column("Status", style="green")

        for model in models_list:
            team_name = model.team.team_name if model.team else "N/A"
            status = "✓ Active" if model.is_active else "✗ Inactive"
            table.add_row(model.model_name, team_name, model.risk_level, status)

        console.print(table)


@app.command()
def model_info(model_name: str) -> None:
    """Display detailed information about a model."""
    with get_db_session() as session:
        repos = RepositoryFacade()

        model = repos.models.get_by_name(session, model_name, include_versions=True)
        if not model:
            console.print(f"[red]Model '{model_name}' not found[/red]")
            sys.exit(1)

        # Basic info
        console.print(f"\n[bold cyan]Model: {model.model_name}[/bold cyan]")
        console.print(f"Description: {model.description or 'N/A'}")
        console.print(f"Use Case: {model.use_case or 'N/A'}")
        console.print(f"Risk Level: {model.risk_level}")
        console.print(f"Compliance: {'Required' if model.compliance_required else 'Not Required'}")
        console.print(f"Status: {'Active' if model.is_active else 'Inactive'}")

        if model.team:
            console.print(f"Team: {model.team.team_name}")

        # Versions
        console.print(f"\n[bold]Versions ({len(model.versions)})[/bold]")
        if model.versions:
            version_table = Table()
            version_table.add_column("Version", style="cyan")
            version_table.add_column("Framework", style="blue")
            version_table.add_column("Status", style="green")
            version_table.add_column("Created", style="yellow")

            for version in sorted(model.versions, key=lambda v: v.created_at, reverse=True):
                version_table.add_row(
                    version.semver,
                    version.framework,
                    version.status,
                    version.created_at.strftime("%Y-%m-%d %H:%M")
                )

            console.print(version_table)
        else:
            console.print("[yellow]No versions registered[/yellow]")

        # Analytics
        summary = repos.analytics.get_model_summary(session, model.model_id)
        console.print(f"\n[bold]Analytics[/bold]")
        console.print(f"Training Runs: {summary['training_run_count']}")
        console.print(f"Deployments: {summary['deployment_count']}")
        console.print(f"Total GPU Hours: {summary['total_gpu_hours']:.2f}")
        console.print(f"Total Cost: ${summary['total_cost_usd']:.2f}")


@app.command()
def test_connection() -> None:
    """Test database connection."""
    try:
        console.print("[yellow]Testing database connection...[/yellow]")
        with engine.connect() as conn:
            result = conn.execute("SELECT version()")
            version = result.scalar()
            console.print(f"[green]✓ Connected successfully[/green]")
            console.print(f"PostgreSQL Version: {version}")
    except Exception as e:
        console.print(f"[red]✗ Connection failed: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    app()
```

**Usage Examples:**

```bash
# Install as command
poetry install

# Test connection
ml-registry test-connection

# Initialize database
ml-registry init

# Seed with sample data
ml-registry seed

# View statistics
ml-registry stats

# List models
ml-registry list-models
ml-registry list-models --active --limit 10

# View model details
ml-registry model-info "bert-classifier"

# View configuration
ml-registry info

# Reset database (with confirmation)
ml-registry reset
```

---

## Part 9: Testing with Pytest and Factory Boy

### Step 9.1: `tests/conftest.py` - Pytest Fixtures

```python
"""Pytest configuration and shared fixtures."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from ml_registry_db.config import get_settings
from ml_registry_db.db import Base
from ml_registry_db import models


@pytest.fixture(scope="session")
def test_settings():
    """Override settings for testing."""
    settings = get_settings()
    # Ensure we're using test database
    settings.environment = "test"
    return settings


@pytest.fixture(scope="session")
def engine(test_settings):
    """Create test database engine."""
    # Use test database URL
    test_url = str(test_settings.test_database_url or test_settings.database_url).replace(
        "/ml_registry", "/ml_registry_test"
    )

    engine = create_engine(test_url, echo=False)

    # Create all tables
    Base.metadata.create_all(engine)

    yield engine

    # Drop all tables after tests
    Base.metadata.drop_all(engine)
    engine.dispose()


@pytest.fixture(scope="function")
def session(engine):
    """Create a new database session for each test."""
    connection = engine.connect()
    transaction = connection.begin()
    session = sessionmaker(bind=connection)()

    yield session

    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture(scope="function")
def db_session(session):
    """Alias for session fixture (more intuitive name)."""
    return session
```

### Step 9.2: `tests/factories.py` - Factory Boy Test Data Generators

```python
"""Factory Boy factories for generating test data."""

import factory
from factory.alchemy import SQLAlchemyModelFactory
from faker import Faker

from ml_registry_db import models
from tests.conftest import session

fake = Faker()


class TeamFactory(SQLAlchemyModelFactory):
    """Factory for Team model."""

    class Meta:
        model = models.Team
        sqlalchemy_session_persistence = "commit"

    team_name = factory.Sequence(lambda n: f"team-{n}")
    department = factory.Faker("job")
    cost_center = factory.Faker("bothify", text="CC-####")
    active = True


class TagFactory(SQLAlchemyModelFactory):
    """Factory for Tag model."""

    class Meta:
        model = models.Tag
        sqlalchemy_session_persistence = "commit"

    tag_name = factory.Sequence(lambda n: f"tag-{n}")
    description = factory.Faker("sentence")


class ModelFactory(SQLAlchemyModelFactory):
    """Factory for Model."""

    class Meta:
        model = models.Model
        sqlalchemy_session_persistence = "commit"

    model_name = factory.Sequence(lambda n: f"model-{n}")
    description = factory.Faker("text", max_nb_chars=200)
    use_case = factory.Faker("bs")
    team = factory.SubFactory(TeamFactory)
    risk_level = factory.Faker("random_element", elements=["low", "medium", "high", "critical"])
    compliance_required = factory.Faker("boolean", chance_of_getting_true=30)
    is_active = True


class ModelVersionFactory(SQLAlchemyModelFactory):
    """Factory for ModelVersion."""

    class Meta:
        model = models.ModelVersion
        sqlalchemy_session_persistence = "commit"

    model = factory.SubFactory(ModelFactory)
    semver = factory.Sequence(lambda n: f"1.0.{n}")
    artifact_uri = factory.Faker("url")
    framework = factory.Faker("random_element", elements=["pytorch", "tensorflow", "sklearn"])
    framework_version = "2.0.1"
    python_version = "3.11"
    status = "registered"
    size_bytes = factory.Faker("random_int", min=1000000, max=5000000000)
    checksum_sha256 = factory.Faker("sha256")
    metadata_ = factory.LazyFunction(lambda: {"optimizer": "adam", "lr": 0.001})


class TrainingRunFactory(SQLAlchemyModelFactory):
    """Factory for TrainingRun."""

    class Meta:
        model = models.TrainingRun
        sqlalchemy_session_persistence = "commit"

    version = factory.SubFactory(ModelVersionFactory)
    run_name = factory.Sequence(lambda n: f"run-{n}")
    status = "succeeded"
    gpu_hours = factory.Faker("pyfloat", left_digits=2, right_digits=2, positive=True)
    cost_usd = factory.Faker("pyfloat", left_digits=3, right_digits=2, positive=True)
    dataset_name = factory.Faker("word")
    dataset_version = "v1.0"
    hyperparameters = factory.LazyFunction(
        lambda: {
            "batch_size": fake.random_int(16, 128),
            "learning_rate": fake.pyfloat(left_digits=0, right_digits=4, positive=True),
            "epochs": fake.random_int(10, 100),
        }
    )
    environment = factory.LazyFunction(
        lambda: {"gpu": "A100", "cuda": "11.8", "python": "3.11"}
    )
    created_by = factory.Faker("user_name")


class EnvironmentFactory(SQLAlchemyModelFactory):
    """Factory for Environment."""

    class Meta:
        model = models.Environment
        sqlalchemy_session_persistence = "commit"

    environment_name = factory.Sequence(lambda n: f"env-{n}")
    environment_type = factory.Faker("random_element", elements=["development", "staging", "production"])
    region = factory.Faker("random_element", elements=["us-east-1", "us-west-2", "eu-west-1"])
    kubernetes_namespace = factory.Faker("slug")
    config = factory.LazyFunction(lambda: {"replicas": 3, "autoscaling": True})
    is_active = True


class DeploymentFactory(SQLAlchemyModelFactory):
    """Factory for Deployment."""

    class Meta:
        model = models.Deployment
        sqlalchemy_session_persistence = "commit"

    version = factory.SubFactory(ModelVersionFactory)
    environment = factory.SubFactory(EnvironmentFactory)
    deployment_name = factory.Sequence(lambda n: f"deployment-{n}")
    status = "healthy"
    endpoint_url = factory.Faker("url")
    replicas = 3
    cpu_request = "1000m"
    memory_request = "2Gi"
    gpu_count = 0
    deployed_by = factory.Faker("user_name")


class ModelMetricFactory(SQLAlchemyModelFactory):
    """Factory for ModelMetric."""

    class Meta:
        model = models.ModelMetric
        sqlalchemy_session_persistence = "commit"

    version = factory.SubFactory(ModelVersionFactory)
    metric_name = factory.Faker("random_element", elements=["accuracy", "f1_score", "precision", "recall"])
    metric_value = factory.Faker("pyfloat", left_digits=0, right_digits=4, positive=True, max_value=1.0)
    dataset_split = factory.Faker("random_element", elements=["train", "validation", "test"])
```

### Step 9.3: `tests/test_models.py` - ORM Model Tests

```python
"""Test ORM models."""

import pytest
from sqlalchemy.exc import IntegrityError

from ml_registry_db import models
from tests.factories import ModelFactory, ModelVersionFactory, TeamFactory


def test_team_creation(db_session):
    """Test creating a team."""
    team = TeamFactory(team_name="ml-platform")

    assert team.team_id is not None
    assert team.team_name == "ml-platform"
    assert team.active is True
    assert team.created_at is not None


def test_team_unique_constraint(db_session):
    """Test team name uniqueness."""
    TeamFactory(team_name="duplicate-team")

    with pytest.raises(IntegrityError):
        TeamFactory(team_name="duplicate-team")
        db_session.commit()


def test_model_creation(db_session):
    """Test creating a model."""
    team = TeamFactory()
    model = ModelFactory(
        model_name="bert-classifier",
        description="BERT for sentiment analysis",
        team=team,
        risk_level="medium"
    )

    assert model.model_id is not None
    assert model.model_name == "bert-classifier"
    assert model.team_id == team.team_id
    assert model.risk_level == "medium"


def test_model_version_relationship(db_session):
    """Test model-to-version one-to-many relationship."""
    model = ModelFactory()
    version1 = ModelVersionFactory(model=model, semver="1.0.0")
    version2 = ModelVersionFactory(model=model, semver="1.1.0")

    db_session.refresh(model)

    assert len(model.versions) == 2
    assert version1 in model.versions
    assert version2 in model.versions


def test_model_version_unique_constraint(db_session):
    """Test unique constraint on (model_id, semver)."""
    model = ModelFactory()
    ModelVersionFactory(model=model, semver="1.0.0")

    with pytest.raises(IntegrityError):
        ModelVersionFactory(model=model, semver="1.0.0")
        db_session.commit()


def test_model_cascade_delete(db_session):
    """Test cascade delete of versions when model is deleted."""
    model = ModelFactory()
    version = ModelVersionFactory(model=model)
    version_id = version.version_id

    db_session.delete(model)
    db_session.commit()

    # Version should be deleted
    assert db_session.get(models.ModelVersion, version_id) is None


def test_team_set_null_on_delete(db_session):
    """Test SET NULL behavior when team is deleted."""
    team = TeamFactory()
    model = ModelFactory(team=team)

    db_session.delete(team)
    db_session.commit()

    db_session.refresh(model)
    assert model.team_id is None
```

### Step 9.4: `tests/test_repositories.py` - Repository Tests

```python
"""Test repository operations."""

import pytest

from ml_registry_db import schemas
from ml_registry_db.repositories import RepositoryFacade
from tests.factories import ModelFactory, ModelVersionFactory, TeamFactory


@pytest.fixture
def repos():
    """Create repository facade."""
    return RepositoryFacade()


def test_create_team(db_session, repos):
    """Test creating team via repository."""
    team_data = schemas.TeamCreate(
        team_name="ml-platform",
        department="Engineering",
        cost_center="CC-1234"
    )

    team = repos.teams.create(db_session, team_data)

    assert team.team_id is not None
    assert team.team_name == "ml-platform"
    assert team.department == "Engineering"


def test_get_team_by_name(db_session, repos):
    """Test retrieving team by name."""
    team = TeamFactory(team_name="search-team")

    found = repos.teams.get_by_name(db_session, "search-team")

    assert found is not None
    assert found.team_id == team.team_id


def test_list_models_with_filters(db_session, repos):
    """Test listing models with filters."""
    team1 = TeamFactory()
    team2 = TeamFactory()

    ModelFactory(team=team1, is_active=True, risk_level="high")
    ModelFactory(team=team1, is_active=False, risk_level="medium")
    ModelFactory(team=team2, is_active=True, risk_level="high")

    # Filter by active and risk level
    results = repos.models.list_all(
        db_session,
        active_only=True,
        risk_level="high",
        limit=100
    )

    assert len(results) == 2
    assert all(m.is_active for m in results)
    assert all(m.risk_level == "high" for m in results)


def test_get_latest_version(db_session, repos):
    """Test retrieving latest version."""
    model = ModelFactory()
    version1 = ModelVersionFactory(model=model, semver="1.0.0")
    version2 = ModelVersionFactory(model=model, semver="1.1.0")
    version3 = ModelVersionFactory(model=model, semver="2.0.0")

    latest = repos.versions.get_latest_version(db_session, model.model_id)

    assert latest.version_id == version3.version_id
    assert latest.semver == "2.0.0"


def test_get_model_summary(db_session, repos):
    """Test analytics repository model summary."""
    model = ModelFactory()
    version1 = ModelVersionFactory(model=model)
    version2 = ModelVersionFactory(model=model, status="production")

    summary = repos.analytics.get_model_summary(db_session, model.model_id)

    assert summary["version_count"] == 2
    assert summary["production_version"] == version2.semver
```

### Step 9.5: Run Tests

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=ml_registry_db --cov-report=html

# Run specific test file
poetry run pytest tests/test_models.py

# Run with verbose output
poetry run pytest -v

# Run and show print statements
poetry run pytest -s
```

---

## Part 10: FastAPI Integration

### Step 10.1: Create FastAPI Application

Create `src/ml_registry_db/api.py`:

```python
"""FastAPI application for ML Registry."""

from typing import List, Optional

from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy.orm import Session

from ml_registry_db import schemas
from ml_registry_db.db import get_db_session
from ml_registry_db.repositories import RepositoryFacade

app = FastAPI(
    title="ML Model Registry API",
    description="RESTful API for managing ML models, versions, and deployments",
    version="0.1.0",
)

repos = RepositoryFacade()


def get_session():
    """FastAPI dependency for database sessions."""
    with get_db_session() as session:
        yield session


# ============================================================================
# Model Endpoints
# ============================================================================


@app.post("/models", response_model=schemas.ModelResponse, status_code=201)
def create_model(
    model_data: schemas.ModelCreate,
    session: Session = Depends(get_session)
):
    """Create a new model."""
    return repos.models.create(session, model_data)


@app.get("/models", response_model=List[schemas.ModelResponse])
def list_models(
    active_only: bool = False,
    skip: int = 0,
    limit: int = 100,
    session: Session = Depends(get_session)
):
    """List all models."""
    return repos.models.list_all(session, active_only=active_only, skip=skip, limit=limit)


@app.get("/models/{model_name}", response_model=schemas.ModelResponse)
def get_model(
    model_name: str,
    session: Session = Depends(get_session)
):
    """Get model by name."""
    model = repos.models.get_by_name(session, model_name)
    if not model:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    return model


# ============================================================================
# Model Version Endpoints
# ============================================================================


@app.post("/versions", response_model=schemas.ModelVersionResponse, status_code=201)
def create_version(
    version_data: schemas.ModelVersionCreate,
    session: Session = Depends(get_session)
):
    """Create a new model version."""
    return repos.versions.create(session, version_data)


@app.get("/models/{model_name}/versions", response_model=List[schemas.ModelVersionResponse])
def list_versions(
    model_name: str,
    status: Optional[str] = None,
    session: Session = Depends(get_session)
):
    """List versions for a model."""
    model = repos.models.get_by_name(session, model_name)
    if not model:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

    return repos.versions.list_by_model(session, model.model_id, status=status)


# ============================================================================
# Run with: uvicorn ml_registry_db.api:app --reload
# ============================================================================
```

**Run the API:**

```bash
# Development server with auto-reload
poetry run uvicorn ml_registry_db.api:app --reload --host 0.0.0.0 --port 8000

# Access API docs
# http://localhost:8000/docs
```

---

**Summary**

This implementation guide has covered:

1. **Project Setup**: Poetry, directory structure, environment configuration
2. **Configuration Management**: Pydantic Settings with validation and caching
3. **Database Layer**: SQLAlchemy engine, session management, connection pooling
4. **ORM Models**: Complete schema with relationships, constraints, and indexes (11 tables)
5. **Pydantic Schemas**: Request/response validation models for all entities
6. **Repository Pattern**: Data access layer with 9 repositories covering all entities
7. **Alembic Migrations**: Schema versioning and migration management
8. **CLI Tools**: Typer-based command-line interface for database management
9. **Testing**: Pytest configuration with Factory Boy for test data generation
10. **FastAPI Integration**: RESTful API endpoints backed by the repository layer

**Completed**: Exercise 04 - SQLAlchemy ORM Integration (Full Implementation)

**Total Content**: 6,000+ lines covering all aspects of building a production-ready ML Model Registry with SQLAlchemy ORM

**What You Learned**:
- Complete ORM modeling with SQLAlchemy 2.0
- Repository pattern for clean separation of concerns
- Database migrations with Alembic
- CLI tools for database management
- Comprehensive testing strategies
- FastAPI integration for REST APIs

**Next Exercise**: Exercise 05 - Optimization & Indexing
