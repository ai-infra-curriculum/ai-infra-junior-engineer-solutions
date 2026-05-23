# Exercise 04: SQLAlchemy ORM Integration — Solution

Reference solution for [learning exercise-04-sqlalchemy-orm-integration](https://github.com/ai-infra-curriculum/ai-infra-junior-engineer-learning/tree/main/lessons/mod-008-databases-sql/exercises/exercise-04-sqlalchemy-orm-integration.md).

Re-implement the model registry from exercise 02 using SQLAlchemy 2.0 ORM with typed mappings, Alembic migrations, and a clean repository layer.

## What this covers

- SQLAlchemy 2.0 declarative `Mapped[...]` typing
- Relationships and lazy/eager loading trade-offs
- Session management with `sessionmaker` and `scoped_session`
- Connection pooling configuration
- Alembic migrations from scratch
- Repository pattern to keep ORM details out of business logic
- Integration tests against a real Postgres

## Files

- `solutions/models.py` — Declarative models.
- `solutions/database.py` — Engine, session factory, pool configuration.
- `solutions/repositories.py` — Repository classes per aggregate.
- `solutions/alembic.ini` + `solutions/migrations/` — Migration setup.
- `tests/test_repositories.py` — End-to-end tests of the repository layer.

## Running

```bash
pip install 'sqlalchemy[asyncio]>=2.0' alembic 'psycopg[binary]>=3.1'

export DATABASE_URL=postgresql+psycopg://postgres:devpass@localhost:5432/ml

# Apply migrations
alembic -c solutions/alembic.ini upgrade head

# Run tests
pytest tests/
```
