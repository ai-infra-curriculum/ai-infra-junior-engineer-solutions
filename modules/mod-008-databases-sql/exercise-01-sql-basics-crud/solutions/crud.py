"""Reference CRUD layer for the predictions/models schema.

Uses psycopg (v3) with named connection pooling. Each function is small and
testable. The interesting design choices are commented inline.
"""
from __future__ import annotations

import argparse
import json
import os
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterable, Iterator

import psycopg
from psycopg.rows import dict_row


DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:devpass@localhost:5432/ml")


@dataclass(frozen=True)
class Model:
    id: int
    name: str
    version: str
    framework: str
    artifact_uri: str
    is_active: bool


@contextmanager
def connect() -> Iterator[psycopg.Connection]:
    """Single connection helper.

    Real services should use a pool (psycopg_pool.ConnectionPool). Kept simple
    here so the exercise focuses on SQL, not connection management.
    """
    conn = psycopg.connect(DATABASE_URL, row_factory=dict_row)
    try:
        yield conn
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def register_model(name: str, version: str, framework: str, artifact_uri: str) -> int:
    """Idempotent: re-registering returns the existing id."""
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO models (name, version, framework, artifact_uri)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (name, version)
            DO UPDATE SET artifact_uri = EXCLUDED.artifact_uri,
                          framework    = EXCLUDED.framework
            RETURNING id
            """,
            (name, version, framework, artifact_uri),
        )
        row = cur.fetchone()
        conn.commit()
        return row["id"]


def activate_version(name: str, version: str) -> None:
    """Atomically deactivate prior versions and activate `version`."""
    with connect() as conn, conn.cursor() as cur:
        # The transaction is the whole point: never leave the table without
        # an active version mid-update.
        cur.execute("BEGIN")
        cur.execute(
            "UPDATE models SET is_active = FALSE WHERE name = %s AND is_active = TRUE",
            (name,),
        )
        cur.execute(
            "UPDATE models SET is_active = TRUE WHERE name = %s AND version = %s",
            (name, version),
        )
        if cur.rowcount == 0:
            conn.rollback()
            raise ValueError(f"unknown model {name}@{version}")
        conn.commit()


def get_active_model(name: str) -> Model | None:
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT * FROM models WHERE name = %s AND is_active = TRUE",
            (name,),
        )
        row = cur.fetchone()
        return Model(**row) if row else None


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------
def record_prediction(
    model_id: int,
    features: dict[str, Any],
    prediction: float,
    latency_ms: float,
    request_id: uuid.UUID | None = None,
) -> int:
    request_id = request_id or uuid.uuid4()
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO predictions (model_id, request_id, features, prediction, latency_ms)
            VALUES (%s, %s, %s::jsonb, %s, %s)
            RETURNING id
            """,
            (model_id, str(request_id), json.dumps(features), prediction, latency_ms),
        )
        row = cur.fetchone()
        conn.commit()
        return row["id"]


def record_predictions_batch(rows: Iterable[dict[str, Any]]) -> int:
    """Multi-row insert: orders of magnitude faster than one-at-a-time."""
    rows = list(rows)
    if not rows:
        return 0
    with connect() as conn, conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO predictions (model_id, request_id, features, prediction, latency_ms)
            VALUES (%(model_id)s, %(request_id)s, %(features)s::jsonb,
                    %(prediction)s, %(latency_ms)s)
            """,
            [
                {
                    **r,
                    "request_id": str(r.get("request_id") or uuid.uuid4()),
                    "features":   json.dumps(r["features"]),
                }
                for r in rows
            ],
        )
        conn.commit()
        return cur.rowcount


def list_recent(
    name: str, version: str, before: str | None = None, limit: int = 50,
) -> list[dict[str, Any]]:
    """Keyset pagination by created_at. `before` is the prior page's cursor."""
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT p.id, p.request_id, p.prediction, p.latency_ms, p.created_at
            FROM predictions p
            JOIN models m ON m.id = p.model_id
            WHERE m.name = %s AND m.version = %s
              AND (%s::timestamptz IS NULL OR p.created_at < %s::timestamptz)
            ORDER BY p.created_at DESC
            LIMIT %s
            """,
            (name, version, before, before, limit),
        )
        return cur.fetchall()


def per_model_summary(hours: int = 24) -> list[dict[str, Any]]:
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                m.name, m.version,
                COUNT(*)                                                  AS predictions_count,
                AVG(p.latency_ms)::numeric(10,2)                          AS avg_latency_ms,
                PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY p.latency_ms) AS p95_latency_ms
            FROM predictions p
            JOIN models m ON m.id = p.model_id
            WHERE p.created_at > NOW() - make_interval(hours => %s)
            GROUP BY m.name, m.version
            ORDER BY predictions_count DESC
            """,
            (hours,),
        )
        return cur.fetchall()


def prune_old(days: int = 90) -> int:
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            "DELETE FROM predictions WHERE created_at < NOW() - make_interval(days => %s)",
            (days,),
        )
        deleted = cur.rowcount
        conn.commit()
        return deleted


# ---------------------------------------------------------------------------
# CLI for ad-hoc operations during the exercise
# ---------------------------------------------------------------------------
def _main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("register")
    s.add_argument("--name", required=True)
    s.add_argument("--version", required=True)
    s.add_argument("--framework", required=True, choices=["pytorch", "tensorflow", "sklearn", "onnx"])
    s.add_argument("--uri", required=True)

    s = sub.add_parser("activate")
    s.add_argument("--name", required=True)
    s.add_argument("--version", required=True)

    sub.add_parser("summary")
    sub.add_parser("prune")

    args = p.parse_args()
    if args.cmd == "register":
        print(register_model(args.name, args.version, args.framework, args.uri))
    elif args.cmd == "activate":
        activate_version(args.name, args.version)
    elif args.cmd == "summary":
        for row in per_model_summary():
            print(row)
    elif args.cmd == "prune":
        print(f"deleted {prune_old()} rows")


if __name__ == "__main__":
    _main()
