"""
Performance Benchmark: PostgreSQL vs MongoDB vs Redis

Demonstrates the performance characteristics of each database for different use cases.
"""

import time
import json
from typing import Callable, Dict, Any
from postgres_client import get_session as pg_session, engine
from mongodb_client import db
from redis_client import redis_client
from sqlalchemy import text
import statistics

def benchmark_operation(
    name: str,
    operation: Callable,
    num_iterations: int = 1000
) -> Dict[str, Any]:
    """Run benchmark and collect statistics."""
    print(f"\nBenchmarking: {name}")
    print(f"  Iterations: {num_iterations}")

    latencies = []
    for i in range(num_iterations):
        start = time.time()
        operation(i)
        latency = (time.time() - start) * 1000  # Convert to ms
        latencies.append(latency)

    total_time = sum(latencies)
    return {
        "name": name,
        "iterations": num_iterations,
        "total_time_ms": total_time,
        "avg_latency_ms": statistics.mean(latencies),
        "median_latency_ms": statistics.median(latencies),
        "min_latency_ms": min(latencies),
        "max_latency_ms": max(latencies),
        "p95_latency_ms": statistics.quantiles(latencies, n=20)[18] if len(latencies) > 20 else max(latencies),
        "p99_latency_ms": statistics.quantiles(latencies, n=100)[98] if len(latencies) > 100 else max(latencies),
        "ops_per_second": num_iterations / (total_time / 1000)
    }

# Benchmark 1: Simple Key-Value Writes
def benchmark_writes():
    print("\n" + "="*70)
    print("BENCHMARK 1: Simple Key-Value Writes")
    print("="*70)

    # PostgreSQL
    with pg_session() as session:
        session.execute(text("CREATE TABLE IF NOT EXISTS kv_test (key TEXT PRIMARY KEY, value TEXT)"))
        session.execute(text("TRUNCATE kv_test"))

    def pg_write(i):
        with pg_session() as session:
            session.execute(
                text("INSERT INTO kv_test (key, value) VALUES (:k, :v) ON CONFLICT (key) DO UPDATE SET value = :v"),
                {"k": f"key_{i}", "v": f"value_{i}"}
            )

    # MongoDB
    def mongo_write(i):
        db.kv_test.update_one(
            {"key": f"key_{i}"},
            {"$set": {"value": f"value_{i}"}},
            upsert=True
        )

    # Redis
    def redis_write(i):
        redis_client.set(f"key_{i}", f"value_{i}")

    pg_result = benchmark_operation("PostgreSQL Write", pg_write, 500)
    mongo_result = benchmark_operation("MongoDB Write", mongo_write, 500)
    redis_result = benchmark_operation("Redis Write", redis_write, 500)

    print_comparison([pg_result, mongo_result, redis_result])

# Benchmark 2: Simple Key-Value Reads
def benchmark_reads():
    print("\n" + "="*70)
    print("BENCHMARK 2: Simple Key-Value Reads")
    print("="*70)

    # Pre-populate data
    print("  Pre-populating data...")
    with pg_session() as session:
        for i in range(100):
            session.execute(text("INSERT INTO kv_test (key, value) VALUES (:k, :v) ON CONFLICT DO NOTHING"), {"k": f"key_{i}", "v": f"value_{i}"})

    for i in range(100):
        db.kv_test.update_one({"key": f"key_{i}"}, {"$set": {"value": f"value_{i}"}}, upsert=True)
        redis_client.set(f"key_{i}", f"value_{i}")

    # PostgreSQL
    def pg_read(i):
        with pg_session() as session:
            result = session.execute(text("SELECT value FROM kv_test WHERE key = :k"), {"k": f"key_{i % 100}"})
            result.fetchone()

    # MongoDB
    def mongo_read(i):
        db.kv_test.find_one({"key": f"key_{i % 100}"})

    # Redis
    def redis_read(i):
        redis_client.get(f"key_{i % 100}")

    pg_result = benchmark_operation("PostgreSQL Read", pg_read, 1000)
    mongo_result = benchmark_operation("MongoDB Read", mongo_read, 1000)
    redis_result = benchmark_operation("Redis Read", redis_read, 1000)

    print_comparison([pg_result, mongo_result, redis_result])

# Benchmark 3: Complex Queries
def benchmark_complex_queries():
    print("\n" + "="*70)
    print("BENCHMARK 3: Complex Analytical Queries")
    print("="*70)

    # PostgreSQL: JOIN + Aggregation
    def pg_complex(_):
        with pg_session() as session:
            session.execute(text("""
                SELECT
                    framework,
                    COUNT(*) as total,
                    AVG(accuracy) as avg_acc,
                    MAX(training_time_seconds) as max_time
                FROM training_runs
                WHERE status = 'completed'
                GROUP BY framework
                ORDER BY avg_acc DESC
            """)).fetchall()

    # MongoDB: Aggregation Pipeline
    def mongo_complex(_):
        list(db.model_configs.aggregate([
            {"$group": {
                "_id": "$framework",
                "count": {"$sum": 1},
                "avg_accuracy": {"$avg": "$metrics.accuracy"}
            }},
            {"$sort": {"avg_accuracy": -1}}
        ]))

    pg_result = benchmark_operation("PostgreSQL Complex Query", pg_complex, 100)
    mongo_result = benchmark_operation("MongoDB Aggregation", mongo_complex, 100)

    print_comparison([pg_result, mongo_result])
    print("\n  Note: PostgreSQL excels at complex JOINs and aggregations")
    print("        MongoDB better for flexible schemas and horizontal scaling")

# Benchmark 4: Batch Operations
def benchmark_batch_operations():
    print("\n" + "="*70)
    print("BENCHMARK 4: Batch Operations")
    print("="*70)

    batch_size = 100

    # PostgreSQL
    def pg_batch(_):
        with pg_session() as session:
            values = [f"('batch_key_{i}', 'batch_value_{i}')" for i in range(batch_size)]
            session.execute(text(f"INSERT INTO kv_test (key, value) VALUES {', '.join(values)} ON CONFLICT DO NOTHING"))

    # MongoDB
    def mongo_batch(_):
        docs = [{"key": f"batch_key_{i}", "value": f"batch_value_{i}"} for i in range(batch_size)]
        db.kv_test.insert_many(docs, ordered=False)

    # Redis (Pipeline)
    def redis_batch(_):
        pipe = redis_client.pipeline()
        for i in range(batch_size):
            pipe.set(f"batch_key_{i}", f"batch_value_{i}")
        pipe.execute()

    # Cleanup first
    with pg_session() as session:
        session.execute(text("DELETE FROM kv_test WHERE key LIKE 'batch_key_%'"))
    db.kv_test.delete_many({"key": {"$regex": "^batch_key_"}})

    pg_result = benchmark_operation("PostgreSQL Batch Insert", pg_batch, 50)

    db.kv_test.delete_many({"key": {"$regex": "^batch_key_"}})
    mongo_result = benchmark_operation("MongoDB Batch Insert", mongo_batch, 50)

    redis_result = benchmark_operation("Redis Pipeline", redis_batch, 50)

    print_comparison([pg_result, mongo_result, redis_result])

def print_comparison(results):
    """Print formatted benchmark comparison."""
    print("\n  Results:")
    print(f"  {'Database':<20} {'Avg (ms)':<12} {'P95 (ms)':<12} {'P99 (ms)':<12} {'QPS':<10}")
    print(f"  {'-'*66}")

    for result in results:
        print(f"  {result['name']:<20} {result['avg_latency_ms']:<12.4f} {result['p95_latency_ms']:<12.4f} {result['p99_latency_ms']:<12.4f} {result['ops_per_second']:<10.0f}")

    # Find fastest
    fastest = min(results, key=lambda x: x['avg_latency_ms'])
    print(f"\n  ✓ Fastest: {fastest['name']} ({fastest['ops_per_second']:.0f} QPS)")

    # Calculate speedups
    for result in results:
        if result != fastest:
            speedup = result['avg_latency_ms'] / fastest['avg_latency_ms']
            print(f"    {fastest['name']} is {speedup:.1f}x faster than {result['name']}")

def print_summary():
    """Print decision-making summary."""
    print("\n" + "="*70)
    print("SUMMARY: When to Use Each Database")
    print("="*70)

    summary = [
        ("PostgreSQL", [
            "Complex queries with JOINs and aggregations",
            "Data requiring ACID transactions",
            "Structured data with clear relationships",
            "Analytical workloads and reporting",
            "When strong consistency is required"
        ]),
        ("MongoDB", [
            "Flexible, rapidly evolving schemas",
            "Nested/hierarchical data structures",
            "Catalog systems with varied item types",
            "When horizontal scaling is needed",
            "ML experiments with varying parameters"
        ]),
        ("Redis", [
            "High-speed caching (<1ms latency)",
            "Real-time feature serving",
            "Session management",
            "Leaderboards and counters",
            "Rate limiting and pub/sub"
        ])
    ]

    for db, use_cases in summary:
        print(f"\n✓ {db}:")
        for use_case in use_cases:
            print(f"  - {use_case}")

if __name__ == "__main__":
    print("="*70)
    print("DATABASE PERFORMANCE BENCHMARKS")
    print("="*70)

    benchmark_writes()
    benchmark_reads()
    benchmark_complex_queries()
    benchmark_batch_operations()
    print_summary()

    print("\n" + "="*70)
    print("✓ All benchmarks complete")
    print("="*70)
