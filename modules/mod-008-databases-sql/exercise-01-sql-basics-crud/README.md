# Exercise 01: SQL Basics & CRUD — Solution

Reference solution for [learning exercise-01-sql-basics-crud](https://github.com/ai-infra-curriculum/ai-infra-junior-engineer-learning/tree/main/lessons/mod-008-databases-sql/exercises/exercise-01-sql-basics-crud.md).

## What this covers

- `CREATE TABLE` with appropriate types and constraints
- `INSERT`, `SELECT`, `UPDATE`, `DELETE` against a Postgres database
- Filtering with `WHERE`, `IN`, `BETWEEN`, `LIKE`, `IS NULL`
- Sorting and paging with `ORDER BY`, `LIMIT`, `OFFSET`
- Grouping and aggregation with `GROUP BY`, `HAVING`
- Transactions and savepoints
- `INSERT ... ON CONFLICT` (upsert) and `RETURNING`

## Files

- `solutions/schema.sql` — DDL for the exercise's `predictions` and `models` tables.
- `solutions/queries.sql` — A catalogue of solution queries, organized by task.
- `solutions/crud.py` — Python implementation of the CRUD operations the exercise asks for, using `psycopg[binary]`.
- `tests/test_crud.py` — pytest-based behavioral tests. Run against a live Postgres instance via `DATABASE_URL`.

## Running

```bash
# Apply schema once
psql "$DATABASE_URL" -f solutions/schema.sql

# Run the Python CRUD layer
python solutions/crud.py --help

# Run tests (creates and drops its own schema)
pytest tests/test_crud.py
```
