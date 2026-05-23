# Exercise 02: Database Design — ML Model Registry — Solution

Reference solution for [learning exercise-02-database-design-ml-registry](https://github.com/ai-infra-curriculum/ai-infra-junior-engineer-learning/tree/main/lessons/mod-008-databases-sql/exercises/exercise-02-database-design-ml-registry.md).

Design a schema for an ML model registry that tracks experiments, model versions, deployments, and lineage.

## Design decisions made here

- **Normalized to 3NF** for the core entities (experiments, models, model_versions, deployments). One denormalized convenience column (`current_deployment_id` on `models`) is justified inline.
- **Soft delete** via `archived_at` on `model_versions` rather than physical deletes — model artifacts often need to be auditable.
- **JSONB for hyperparameters and metrics** — schemas vary per experiment, so this is the right shape. We add expression indexes on the hot-path JSON keys.
- **Lineage edges** modeled as a separate `model_version_lineage` table rather than a self-referencing FK, so a model can derive from multiple parents (ensembles, distillation).

See `solutions/schema.sql` for the full DDL with comments justifying each table and column.

## Files

- `solutions/schema.sql` — Full schema with comments.
- `solutions/seed.sql` — Sample data illustrating each entity.
- `solutions/queries.sql` — Common queries: latest version per experiment, deployment history, lineage walks.
- `tests/test_schema.py` — Property tests confirming constraints actually fire.
