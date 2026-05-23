"""SQLAlchemy 2.0 declarative models for the ML registry.

Mirrors the schema in exercise-02 but expressed via the ORM.
"""
from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    BigInteger, Boolean, CheckConstraint, DateTime, Enum, ForeignKey, Index,
    String, Text, UniqueConstraint, func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Project base. Set common naming conventions here."""


class DeploymentEnvironment(str, enum.Enum):
    dev = "dev"
    staging = "staging"
    prod = "prod"


class DeploymentStatus(str, enum.Enum):
    pending = "pending"
    active = "active"
    failed = "failed"
    rolled_back = "rolled_back"


class Experiment(Base):
    __tablename__ = "experiments"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    name: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    owner: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    models: Mapped[list["Model"]] = relationship(back_populates="experiment")


class Model(Base):
    __tablename__ = "models"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    experiment_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("experiments.id"), nullable=False)
    name: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    current_deployment_id: Mapped[Optional[int]] = mapped_column(BigInteger, ForeignKey("deployments.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    experiment: Mapped[Experiment] = relationship(back_populates="models")
    versions: Mapped[list["ModelVersion"]] = relationship(
        back_populates="model", cascade="all, delete-orphan",
    )
    current_deployment: Mapped[Optional["Deployment"]] = relationship(
        foreign_keys=[current_deployment_id], post_update=True,
    )


class ModelVersion(Base):
    __tablename__ = "model_versions"
    __table_args__ = (
        UniqueConstraint("model_id", "version_tag"),
        Index("idx_versions_model_trained_at", "model_id", "trained_at"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    model_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("models.id"), nullable=False)
    version_tag: Mapped[str] = mapped_column(String, nullable=False)
    artifact_uri: Mapped[str] = mapped_column(String, nullable=False)
    framework: Mapped[str] = mapped_column(String, nullable=False)
    hyperparameters: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    metrics: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    trained_by: Mapped[str] = mapped_column(String, nullable=False)
    trained_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    archived_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    model: Mapped[Model] = relationship(back_populates="versions")
    deployments: Mapped[list["Deployment"]] = relationship(back_populates="model_version")
    parent_edges: Mapped[list["LineageEdge"]] = relationship(
        foreign_keys="LineageEdge.child_id", back_populates="child",
    )

    __table_args__ = (
        UniqueConstraint("model_id", "version_tag", name="uq_modelver_model_version"),
        Index("idx_versions_metrics_gin", "metrics", postgresql_using="gin"),
        Index("idx_versions_hyperparams_gin", "hyperparameters", postgresql_using="gin"),
    )


class Deployment(Base):
    __tablename__ = "deployments"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    model_version_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("model_versions.id"), nullable=False)
    environment: Mapped[DeploymentEnvironment] = mapped_column(
        Enum(DeploymentEnvironment, name="deployment_environment"), nullable=False,
    )
    status: Mapped[DeploymentStatus] = mapped_column(
        Enum(DeploymentStatus, name="deployment_status"),
        default=DeploymentStatus.pending, nullable=False,
    )
    endpoint_url: Mapped[Optional[str]] = mapped_column(String)
    deployed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    retired_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    model_version: Mapped[ModelVersion] = relationship(back_populates="deployments")


class LineageEdge(Base):
    __tablename__ = "model_version_lineage"
    __table_args__ = (
        CheckConstraint("child_id <> parent_id", name="ck_no_self_lineage"),
    )

    child_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("model_versions.id"), primary_key=True)
    parent_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("model_versions.id"), primary_key=True)
    edge_type: Mapped[str] = mapped_column(String, nullable=False)

    child: Mapped[ModelVersion] = relationship(
        foreign_keys=[child_id], back_populates="parent_edges",
    )
    parent: Mapped[ModelVersion] = relationship(foreign_keys=[parent_id])
