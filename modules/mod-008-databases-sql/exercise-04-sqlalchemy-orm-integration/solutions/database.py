"""Engine + session factory.

Real services should isolate this in an app-level container (FastAPI deps,
Flask app factory, etc.). Kept simple here.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker


DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql+psycopg://postgres:devpass@localhost:5432/ml",
)

engine: Engine = create_engine(
    DATABASE_URL,
    # pool_pre_ping handles connections killed by the DB or load balancer.
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
    pool_recycle=1800,
    echo=False,
)


@event.listens_for(engine, "connect")
def _set_session_settings(dbapi_conn, _connection_record):
    """Per-connection settings: idle-in-transaction timeout, application_name."""
    with dbapi_conn.cursor() as cur:
        cur.execute("SET application_name = 'ml-registry'")
        cur.execute("SET idle_in_transaction_session_timeout = '30s'")


SessionFactory = sessionmaker(bind=engine, expire_on_commit=False)


@contextmanager
def session_scope() -> Iterator[Session]:
    """Yield a session, commit on success, rollback on error, always close."""
    session = SessionFactory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
