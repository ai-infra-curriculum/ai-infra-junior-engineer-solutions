# Module 008: Databases & SQL — Solutions

Reference solutions for the 5 exercises in [learning module 008](https://github.com/ai-infra-curriculum/ai-infra-junior-engineer-learning/tree/main/lessons/mod-008-databases-sql).

## Exercises

| Exercise | Topic | Solutions |
|---|---|---|
| 01 | SQL basics & CRUD | `exercise-01-sql-basics-crud/` |
| 02 | Database design (ML model registry) | `exercise-02-database-design-ml-registry/` |
| 03 | Advanced SQL joins & window functions | `exercise-03-advanced-sql-joins/` |
| 04 | SQLAlchemy ORM integration | `exercise-04-sqlalchemy-orm-integration/` |
| 05 | Indexing & query optimization | `exercise-05-optimization-indexing/` |

## Setup

A single Postgres container backs all exercises. Bring it up once and reuse:

```bash
docker run -d \
  --name junior-db \
  -e POSTGRES_PASSWORD=devpass \
  -e POSTGRES_DB=ml \
  -p 5432:5432 \
  postgres:15

# Per-exercise solution code expects:
#   DATABASE_URL=postgresql://postgres:devpass@localhost:5432/ml
```

## How to use

1. Attempt the exercise in the learning repo first.
2. Compare your approach to the solution here.
3. Run the test file in each `tests/` directory to validate your own implementation against the same contract.
4. Read the inline comments — they call out the *why*, not just the *what*.
