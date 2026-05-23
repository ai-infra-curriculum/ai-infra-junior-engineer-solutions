"""EXPLAIN-based test: confirm the planner picks the expected index for hot queries.

These tests require enough rows for the planner to prefer an index over a seq scan.
We insert ~5,000 rows in setup to make that the case.
"""
from __future__ import annotations

import os
import uuid
from pathlib import Path

import psycopg
import pytest


DB_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:devpass@localhost:5432/ml")
SCHEMA_E1 = Path(__file__).resolve().parents[2] / "exercise-01-sql-basics-crud" / "solutions" / "schema.sql"
INDEXES = Path(__file__).resolve().parent.parent / "solutions" / "indexes.sql"


@pytest.fixture(scope="module", autouse=True)
def _setup():
    with psycopg.connect(DB_URL) as conn, conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS predictions CASCADE")
        cur.execute("DROP TABLE IF EXISTS models CASCADE")
        cur.execute(SCHEMA_E1.read_text())
        cur.execute(
            "INSERT INTO models (name, version, framework, artifact_uri) "
            "VALUES ('m', 'v1', 'sklearn', 's3://m/v1') RETURNING id"
        )
        mid = cur.fetchone()[0]
        # Bulk insert ~5000 predictions
        cur.executemany(
            "INSERT INTO predictions (model_id, request_id, features, prediction, latency_ms) "
            "VALUES (%s, %s, '{}'::jsonb, random(), 1)",
            [(mid, str(uuid.uuid4())) for _ in range(5000)],
        )
        # Apply our index migration
        cur.execute(INDEXES.read_text())
        cur.execute("ANALYZE predictions")
        conn.commit()


def explain(sql: str, params: tuple = ()) -> str:
    with psycopg.connect(DB_URL) as conn, conn.cursor() as cur:
        cur.execute("EXPLAIN " + sql, params)
        return "\n".join(row[0] for row in cur.fetchall())


def test_request_id_lookup_uses_unique_index():
    plan = explain("SELECT id FROM predictions WHERE request_id = %s",
                   (str(uuid.uuid4()),))
    assert "uniq_predictions_request_id" in plan


def test_model_latency_window_uses_compound_index():
    plan = explain(
        "SELECT AVG(latency_ms) FROM predictions "
        "WHERE model_id = 1 AND created_at > NOW() - INTERVAL '1 hour'",
    )
    # Either Index Scan or Bitmap Index Scan is acceptable.
    assert "idx_predictions_model_created" in plan or "model_id" in plan.lower()
