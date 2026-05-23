# Exercise 05: Query Optimization & Indexing — Solution

Reference solution for [learning exercise-05-optimization-indexing](https://github.com/ai-infra-curriculum/ai-infra-junior-engineer-learning/tree/main/lessons/mod-008-databases-sql/exercises/exercise-05-optimization-indexing.md).

This exercise has two outputs:

1. **A set of optimized queries** with EXPLAIN ANALYZE annotations showing the before/after.
2. **An indexing migration** that adds the right indexes for the workload.

## What this covers

- Reading EXPLAIN plans (`Seq Scan` vs `Index Scan` vs `Bitmap Heap Scan`)
- B-tree vs GIN vs BRIN index choice
- Compound index column order (the ESR rule)
- Partial and expression indexes
- Diagnosing missing statistics with `ANALYZE`
- Caveats of `IN`, `OR`, and implicit casts defeating indexes
- Keyset pagination vs `OFFSET`
- `pg_stat_statements` for production query profiling

## Files

- `solutions/before_after.sql` — Each problem query with its EXPLAIN plan before optimization, the change made, and the EXPLAIN plan after.
- `solutions/indexes.sql` — Index migration script.
- `solutions/benchmark.py` — Runs each query 100× and reports p50/p95.
- `tests/test_index_chosen.py` — Asserts the optimized queries use the expected index name in their plan.
