# Exercise 03: Advanced SQL — Joins, CTEs, Window Functions — Solution

Reference solution for [learning exercise-03-advanced-sql-joins](https://github.com/ai-infra-curriculum/ai-infra-junior-engineer-learning/tree/main/lessons/mod-008-databases-sql/exercises/exercise-03-advanced-sql-joins.md).

This is largely a SQL-only exercise — `solutions/queries.sql` is the deliverable. Each query is annotated with the technique it demonstrates.

## What this covers

- All join types (INNER, LEFT, FULL OUTER, CROSS, anti-join, semi-join)
- CTEs (recursive and non-recursive)
- Window functions (`ROW_NUMBER`, `RANK`, `LAG`, `LEAD`, running aggregates, partitioned windows)
- Subqueries vs CTEs vs joins — which to choose
- `LATERAL` joins for top-N per group

## Files

- `solutions/queries.sql` — Catalogue of solution queries, grouped by technique.
- `tests/test_query_shape.py` — Verifies each query returns the expected row shape against the seed data.
