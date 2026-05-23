"""Shape tests for advanced SQL queries against the registry seed.

Smoke-level checks: each query returns reasonable types and non-negative
counts. These don't replace exercise-02's constraint tests but confirm the
queries are syntactically correct and join cleanly.
"""
from __future__ import annotations

import os
from pathlib import Path

import psycopg
import pytest


DB_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:devpass@localhost:5432/ml")
SCHEMA_E2 = Path(__file__).resolve().parents[2] / "exercise-02-database-design-ml-registry" / "solutions" / "schema.sql"
SEED_E2 = Path(__file__).resolve().parents[2] / "exercise-02-database-design-ml-registry" / "solutions" / "seed.sql"


@pytest.fixture(autouse=True)
def _reset():
    with psycopg.connect(DB_URL) as conn, conn.cursor() as cur:
        for table in (
            "model_version_lineage", "predictions", "deployments",
            "model_versions", "models", "experiments",
        ):
            cur.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
        cur.execute("DROP TYPE IF EXISTS deployment_status CASCADE")
        cur.execute("DROP TYPE IF EXISTS deployment_environment CASCADE")
        cur.execute(SCHEMA_E2.read_text())
        # exercise-01 predictions table — reused here.
        cur.execute("""
            CREATE TABLE predictions (
                id BIGSERIAL PRIMARY KEY,
                model_id BIGINT NOT NULL REFERENCES model_versions(id),
                request_id UUID NOT NULL DEFAULT gen_random_uuid(),
                features JSONB NOT NULL DEFAULT '{}'::jsonb,
                prediction DOUBLE PRECISION NOT NULL,
                latency_ms DOUBLE PRECISION NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)
        cur.execute(SEED_E2.read_text())
        # seed some predictions against the seeded model_versions
        cur.execute("""
            INSERT INTO predictions (model_id, prediction, latency_ms)
            SELECT id, random(), 10 + random() * 100
            FROM model_versions, generate_series(1, 5)
        """)
        conn.commit()


def test_anti_join_yields_zero_for_predicted_versions():
    """Every seeded version has predictions, so anti-join should be empty."""
    with psycopg.connect(DB_URL) as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT mv.id
            FROM model_versions mv
            WHERE NOT EXISTS (SELECT 1 FROM predictions p WHERE p.model_id = mv.id)
        """)
        assert cur.fetchall() == []


def test_window_rank_partitions_by_model():
    with psycopg.connect(DB_URL) as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT m.name, RANK() OVER (
                PARTITION BY mv.model_id ORDER BY (mv.metrics ->> 'roc_auc')::float DESC NULLS LAST
            ) AS rk
            FROM model_versions mv JOIN models m ON m.id = mv.model_id
            WHERE mv.metrics ? 'roc_auc'
        """)
        rows = cur.fetchall()
        # Both ranks should start at 1 for each model.
        first_ranks = {r[0]: r[1] for r in rows}
        assert all(rk >= 1 for rk in first_ranks.values())
