"""Behavioral tests for the CRUD layer.

These tests run against a real Postgres instance reachable via DATABASE_URL.
They create and drop a unique schema per test session to avoid polluting
shared environments.
"""
from __future__ import annotations

import os
import uuid
from pathlib import Path

import psycopg
import pytest

from modules.mod_008_databases_sql.exercise_01_sql_basics_crud.solutions import crud  # noqa: E402


DB_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:devpass@localhost:5432/ml")
SCHEMA_FILE = Path(__file__).resolve().parent.parent / "solutions" / "schema.sql"


@pytest.fixture(scope="session", autouse=True)
def _apply_schema():
    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS predictions CASCADE")
            cur.execute("DROP TABLE IF EXISTS models CASCADE")
            cur.execute(SCHEMA_FILE.read_text())
        conn.commit()


@pytest.fixture(autouse=True)
def _isolate():
    """Truncate between tests so each starts fresh."""
    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE predictions, models RESTART IDENTITY CASCADE")
        conn.commit()


def test_register_is_idempotent():
    id1 = crud.register_model("fraud", "v1", "pytorch", "s3://m/v1")
    id2 = crud.register_model("fraud", "v1", "pytorch", "s3://m/v1-new")
    assert id1 == id2, "second register should update, not duplicate"


def test_activate_exclusive():
    crud.register_model("fraud", "v1", "pytorch", "s3://m/v1")
    crud.register_model("fraud", "v2", "pytorch", "s3://m/v2")

    crud.activate_version("fraud", "v1")
    assert crud.get_active_model("fraud").version == "v1"

    crud.activate_version("fraud", "v2")
    assert crud.get_active_model("fraud").version == "v2"


def test_activate_unknown_raises():
    crud.register_model("fraud", "v1", "pytorch", "s3://m/v1")
    with pytest.raises(ValueError):
        crud.activate_version("fraud", "doesnotexist")


def test_record_and_list():
    mid = crud.register_model("fraud", "v1", "pytorch", "s3://m/v1")
    rid = uuid.uuid4()
    pred_id = crud.record_prediction(mid, {"a": 1.0}, prediction=0.7, latency_ms=12.3, request_id=rid)
    assert isinstance(pred_id, int) and pred_id > 0

    rows = crud.list_recent("fraud", "v1")
    assert len(rows) == 1
    assert rows[0]["prediction"] == pytest.approx(0.7)


def test_record_batch_counts_inserted():
    mid = crud.register_model("fraud", "v1", "pytorch", "s3://m/v1")
    inserted = crud.record_predictions_batch(
        {"model_id": mid, "features": {"x": i}, "prediction": float(i), "latency_ms": 1.0}
        for i in range(25)
    )
    assert inserted == 25
    assert len(crud.list_recent("fraud", "v1", limit=100)) == 25


def test_summary_returns_per_version():
    m1 = crud.register_model("fraud", "v1", "pytorch", "s3://m/v1")
    m2 = crud.register_model("fraud", "v2", "pytorch", "s3://m/v2")
    for mid, latency in ((m1, 5.0), (m1, 10.0), (m2, 100.0)):
        crud.record_prediction(mid, {}, 0.5, latency)

    summary = {(r["name"], r["version"]): r for r in crud.per_model_summary()}
    assert summary[("fraud", "v1")]["predictions_count"] == 2
    assert summary[("fraud", "v2")]["predictions_count"] == 1


def test_prune_removes_old_rows():
    mid = crud.register_model("fraud", "v1", "pytorch", "s3://m/v1")
    crud.record_prediction(mid, {}, 0.5, 5.0)
    # Backdate the row by 100 days
    with psycopg.connect(DB_URL) as conn, conn.cursor() as cur:
        cur.execute("UPDATE predictions SET created_at = NOW() - INTERVAL '100 days'")
        conn.commit()

    deleted = crud.prune_old(days=90)
    assert deleted == 1
    assert crud.list_recent("fraud", "v1") == []
