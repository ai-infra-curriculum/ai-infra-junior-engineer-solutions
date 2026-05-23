"""Repository layer — keeps ORM-specific code out of business logic.

Business code talks to repositories; repositories talk to the session.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from .models import (
    Deployment, DeploymentEnvironment, DeploymentStatus, Experiment,
    LineageEdge, Model, ModelVersion,
)


class ExperimentRepo:
    def __init__(self, session: Session) -> None:
        self.session = session

    def get_or_create(self, name: str, owner: str, description: str | None = None) -> Experiment:
        existing = self.session.scalar(select(Experiment).where(Experiment.name == name))
        if existing:
            return existing
        exp = Experiment(name=name, owner=owner, description=description)
        self.session.add(exp)
        self.session.flush()
        return exp


class ModelRepo:
    def __init__(self, session: Session) -> None:
        self.session = session

    def get_by_name(self, name: str) -> Optional[Model]:
        return self.session.scalar(
            select(Model)
            .where(Model.name == name)
            .options(selectinload(Model.versions)),
        )

    def create(self, experiment: Experiment, name: str) -> Model:
        m = Model(experiment_id=experiment.id, name=name)
        self.session.add(m)
        self.session.flush()
        return m


class ModelVersionRepo:
    def __init__(self, session: Session) -> None:
        self.session = session

    def register(
        self,
        model: Model,
        version_tag: str,
        artifact_uri: str,
        framework: str,
        hyperparameters: dict,
        metrics: dict,
        trained_by: str,
    ) -> ModelVersion:
        mv = ModelVersion(
            model_id=model.id,
            version_tag=version_tag,
            artifact_uri=artifact_uri,
            framework=framework,
            hyperparameters=hyperparameters,
            metrics=metrics,
            trained_by=trained_by,
        )
        self.session.add(mv)
        self.session.flush()
        return mv

    def latest(self, model: Model) -> Optional[ModelVersion]:
        return self.session.scalar(
            select(ModelVersion)
            .where(ModelVersion.model_id == model.id, ModelVersion.archived_at.is_(None))
            .order_by(ModelVersion.trained_at.desc())
            .limit(1),
        )

    def top_by_metric(self, model: Model, metric: str, n: int = 3) -> list[ModelVersion]:
        # Filter to versions with the metric present and order by the JSONB key cast to float.
        return list(
            self.session.scalars(
                select(ModelVersion)
                .where(
                    ModelVersion.model_id == model.id,
                    ModelVersion.metrics.has_key(metric),  # noqa: W601
                )
                .order_by((ModelVersion.metrics[metric].astext.cast(float)).desc())
                .limit(n),
            ),
        )


class DeploymentRepo:
    def __init__(self, session: Session) -> None:
        self.session = session

    def deploy(
        self,
        model_version: ModelVersion,
        environment: DeploymentEnvironment,
        endpoint_url: str,
    ) -> Deployment:
        """Retire any active deployment in the same env, then activate the new version."""
        active = self.session.scalar(
            select(Deployment)
            .where(
                Deployment.environment == environment,
                Deployment.status == DeploymentStatus.active,
                Deployment.retired_at.is_(None),
                Deployment.model_version_id.in_(
                    select(ModelVersion.id).where(ModelVersion.model_id == model_version.model_id),
                ),
            ),
        )
        if active is not None:
            active.status = DeploymentStatus.rolled_back
            active.retired_at = datetime.now(tz=timezone.utc)

        d = Deployment(
            model_version_id=model_version.id,
            environment=environment,
            status=DeploymentStatus.active,
            endpoint_url=endpoint_url,
        )
        self.session.add(d)
        self.session.flush()
        return d


class LineageRepo:
    def __init__(self, session: Session) -> None:
        self.session = session

    def link(self, child: ModelVersion, parent: ModelVersion, edge_type: str) -> LineageEdge:
        edge = LineageEdge(child_id=child.id, parent_id=parent.id, edge_type=edge_type)
        self.session.add(edge)
        self.session.flush()
        return edge

    def ancestors(self, version: ModelVersion) -> list[tuple[int, ModelVersion]]:
        """Recursive walk using a raw CTE — ORM doesn't render recursive CTEs naturally."""
        result = self.session.execute(
            select(LineageEdge).where(LineageEdge.child_id == version.id),
        )
        # Simplified BFS to keep the example self-contained.
        out: list[tuple[int, ModelVersion]] = []
        seen: set[int] = set()
        frontier = [(1, edge.parent_id) for edge in result.scalars()]
        while frontier:
            depth, vid = frontier.pop(0)
            if vid in seen:
                continue
            seen.add(vid)
            mv = self.session.get(ModelVersion, vid)
            if mv is None:
                continue
            out.append((depth, mv))
            edges = self.session.scalars(
                select(LineageEdge).where(LineageEdge.child_id == vid),
            )
            frontier.extend((depth + 1, e.parent_id) for e in edges)
        return out
