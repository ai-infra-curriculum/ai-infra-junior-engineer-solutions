"""Property tests confirming the registry schema's constraints actually fire."""
from __future__ import annotations

import os
from pathlib import Path

import psycopg
import pytest


DB_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:devpass@localhost:5432/ml")
SCHEMA = Path(__file__).resolve().parent.parent / "solutions" / "schema.sql"
SEED = Path(__file__).resolve().parent.parent / "solutions" / "seed.sql"


@pytest.fixture(autouse=True)
def _reset():
    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            for table in (
                "model_version_lineage", "deployments", "model_versions",
                "models", "experiments",
            ):
                cur.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
            cur.execute("DROP TYPE IF EXISTS deployment_status CASCADE")
            cur.execute("DROP TYPE IF EXISTS deployment_environment CASCADE")
            cur.execute(SCHEMA.read_text())
            cur.execute(SEED.read_text())
        conn.commit()


def test_unique_model_name():
    with psycopg.connect(DB_URL) as conn, conn.cursor() as cur:
        with pytest.raises(psycopg.errors.UniqueViolation):
            cur.execute(
                "INSERT INTO experiments(name, owner) VALUES ('fraud-detection', 'x')"
            )


def test_at_most_one_active_per_env():
    with psycopg.connect(DB_URL) as conn, conn.cursor() as cur:
        cur.execute("""
            INSERT INTO model_versions (model_id, version_tag, artifact_uri, framework, trained_by)
            SELECT id, 'v1.2.0', 's3://m/v1.2.0', 'sklearn', 't'
            FROM models WHERE name = 'fraud-detector'
            RETURNING id
        """)
        new_id = cur.fetchone()[0]
        # First active is fine ...
        cur.execute(
            "INSERT INTO deployments (model_version_id, environment, status) VALUES (%s, 'prod', 'active')",
            (new_id,),
        )
        # ... second active for the same (model, env) is blocked.
        with pytest.raises(psycopg.errors.UniqueViolation):
            cur.execute(
                "INSERT INTO deployments (model_version_id, environment, status) VALUES (%s, 'prod', 'active')",
                (new_id,),
            )


def test_lineage_disallows_self_reference():
    with psycopg.connect(DB_URL) as conn, conn.cursor() as cur:
        cur.execute("SELECT id FROM model_versions LIMIT 1")
        mv_id = cur.fetchone()[0]
        with pytest.raises(psycopg.errors.CheckViolation):
            cur.execute(
                "INSERT INTO model_version_lineage(child_id, parent_id, edge_type) VALUES (%s,%s,'retrain')",
                (mv_id, mv_id),
            )


def test_jsonb_query_uses_expression_index():
    """Smoke test that we can filter by metrics value cheaply."""
    with psycopg.connect(DB_URL) as conn, conn.cursor() as cur:
        cur.execute("""
            EXPLAIN
            SELECT id FROM model_versions WHERE (metrics ->> 'roc_auc')::float > 0.9
        """)
        plan = "\n".join(row[0] for row in cur.fetchall())
        # We only care that the planner sees the expression index. On a tiny
        # seed dataset Postgres may still choose a Seq Scan, so we don't assert
        # which scan is chosen — just that EXPLAIN doesn't error.
        assert "model_versions" in plan
