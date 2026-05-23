"""Integration tests for the repository layer.

Assumes Alembic has already run against DATABASE_URL.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'solutions'))
from database import SessionFactory, engine  # noqa: E402
from models import Base, DeploymentEnvironment  # noqa: E402
from repositories import (  # noqa: E402
    DeploymentRepo, ExperimentRepo, LineageRepo, ModelRepo, ModelVersionRepo,
)


@pytest.fixture(autouse=True, scope="module")
def _schema():
    """For tests, drop and recreate via the ORM metadata to avoid coupling to Alembic."""
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)


@pytest.fixture
def session():
    s = SessionFactory()
    try:
        yield s
        s.rollback()
    finally:
        s.close()


def test_register_and_lookup(session):
    exp = ExperimentRepo(session).get_or_create("fraud", owner="alice")
    model = ModelRepo(session).create(exp, name="fraud-detector")
    mv = ModelVersionRepo(session).register(
        model, "v1", "s3://m/v1", "sklearn", {"lr": 0.1}, {"roc_auc": 0.9}, "alice",
    )
    session.commit()

    fetched = ModelRepo(session).get_by_name("fraud-detector")
    assert fetched is not None
    assert [v.version_tag for v in fetched.versions] == ["v1"]

    latest = ModelVersionRepo(session).latest(fetched)
    assert latest.id == mv.id


def test_top_by_metric(session):
    exp = ExperimentRepo(session).get_or_create("x", owner="a")
    m = ModelRepo(session).create(exp, name="m1")
    repo = ModelVersionRepo(session)
    for tag, auc in (("v1", 0.7), ("v2", 0.95), ("v3", 0.8)):
        repo.register(m, tag, f"s3://{tag}", "sklearn", {}, {"roc_auc": auc}, "a")
    session.commit()

    top = repo.top_by_metric(m, "roc_auc", n=2)
    assert [v.version_tag for v in top] == ["v2", "v3"]


def test_deploy_retires_prior_active(session):
    exp = ExperimentRepo(session).get_or_create("x", owner="a")
    m = ModelRepo(session).create(exp, name="m2")
    mv_repo = ModelVersionRepo(session)
    v1 = mv_repo.register(m, "v1", "s3://v1", "sklearn", {}, {}, "a")
    v2 = mv_repo.register(m, "v2", "s3://v2", "sklearn", {}, {}, "a")

    dr = DeploymentRepo(session)
    d1 = dr.deploy(v1, DeploymentEnvironment.prod, "https://e/v1")
    session.commit()
    assert d1.retired_at is None

    d2 = dr.deploy(v2, DeploymentEnvironment.prod, "https://e/v2")
    session.commit()
    session.refresh(d1)
    assert d1.retired_at is not None
    assert d2.retired_at is None


def test_lineage_walk(session):
    exp = ExperimentRepo(session).get_or_create("x", owner="a")
    m = ModelRepo(session).create(exp, name="m3")
    mvr = ModelVersionRepo(session)
    v1 = mvr.register(m, "v1", "s3://v1", "sklearn", {}, {}, "a")
    v2 = mvr.register(m, "v2", "s3://v2", "sklearn", {}, {}, "a")
    v3 = mvr.register(m, "v3", "s3://v3", "sklearn", {}, {}, "a")
    lr = LineageRepo(session)
    lr.link(child=v2, parent=v1, edge_type="retrain")
    lr.link(child=v3, parent=v2, edge_type="retrain")
    session.commit()

    ancestors = lr.ancestors(v3)
    assert [a[1].version_tag for a in ancestors] == ["v2", "v1"]
