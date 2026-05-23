"""Tiny benchmark harness — run each query 100x and report p50/p95.

For real production profiling, use pg_stat_statements. This script is just to
demonstrate "the index actually helps" during the exercise.
"""
from __future__ import annotations

import os
import statistics
import time
from collections.abc import Callable
from typing import Any

import psycopg


DB_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:devpass@localhost:5432/ml")
ITERATIONS = int(os.environ.get("ITER", "100"))


def time_query(sql: str, params: tuple[Any, ...] = ()) -> tuple[float, float]:
    """Return (p50_ms, p95_ms)."""
    with psycopg.connect(DB_URL) as conn:
        samples: list[float] = []
        with conn.cursor() as cur:
            for _ in range(ITERATIONS):
                t0 = time.perf_counter()
                cur.execute(sql, params)
                cur.fetchall()
                samples.append((time.perf_counter() - t0) * 1000)
    samples.sort()
    p50 = samples[len(samples) // 2]
    p95 = samples[int(len(samples) * 0.95)]
    return p50, p95


def main() -> None:
    queries: list[tuple[str, str, tuple[Any, ...]]] = [
        (
            "model_latency_1h",
            "SELECT model_id, AVG(latency_ms) FROM predictions "
            "WHERE created_at > NOW() - INTERVAL '1 hour' GROUP BY model_id",
            (),
        ),
        (
            "best_models_roc_auc",
            "SELECT id, version_tag FROM model_versions "
            "WHERE (metrics->>'roc_auc')::float > 0.9 "
            "ORDER BY (metrics->>'roc_auc')::float DESC LIMIT 10",
            (),
        ),
    ]
    for name, sql, params in queries:
        p50, p95 = time_query(sql, params)
        print(f"{name:30s} p50={p50:6.2f}ms  p95={p95:6.2f}ms")


if __name__ == "__main__":
    main()
