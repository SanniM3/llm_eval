"""Database connection and session management."""

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from evalab.storage.models import Base

# Default database path
DEFAULT_DB_PATH = "evalab.db"


def get_database_url(path: str | Path | None = None) -> str:
    """
    Get the database URL.

    Args:
        path: Optional path to SQLite database file

    Returns:
        SQLAlchemy database URL
    """
    if path is None:
        # Check environment variable first
        db_url = os.environ.get("DATABASE_URL")
        if db_url:
            return db_url
        path = DEFAULT_DB_PATH

    return f"sqlite:///{path}"


def get_engine(db_url: str | None = None, echo: bool = False) -> Engine:
    """
    Create or get a database engine.

    Args:
        db_url: Database URL (defaults to SQLite)
        echo: Whether to echo SQL statements

    Returns:
        SQLAlchemy Engine
    """
    if db_url is None:
        db_url = get_database_url()

    engine = create_engine(
        db_url,
        echo=echo,
        # SQLite-specific settings
        connect_args={"check_same_thread": False} if db_url.startswith("sqlite") else {},
    )

    # Enable foreign key support for SQLite
    if db_url.startswith("sqlite"):

        @event.listens_for(engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    return engine


# Global engine and session factory
_engine: Engine | None = None
_session_factory: sessionmaker | None = None


def init_db(db_url: str | None = None, echo: bool = False) -> Engine:
    """
    Initialize the database and create all tables.

    Args:
        db_url: Database URL
        echo: Whether to echo SQL statements

    Returns:
        SQLAlchemy Engine
    """
    global _engine, _session_factory

    _engine = get_engine(db_url, echo)
    _session_factory = sessionmaker(bind=_engine, autoflush=False, autocommit=False)

    # Create all tables
    Base.metadata.create_all(_engine)

    return _engine


def get_session() -> Session:
    """
    Get a new database session.

    Returns:
        SQLAlchemy Session
    """
    global _session_factory

    if _session_factory is None:
        init_db()

    return _session_factory()


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    """
    Provide a transactional scope around a series of operations.

    Usage:
        with session_scope() as session:
            session.add(item)
            # Commits automatically on success, rolls back on exception
    """
    session = get_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def reset_db(db_url: str | None = None) -> None:
    """
    Reset the database by dropping and recreating all tables.

    WARNING: This will delete all data!

    Args:
        db_url: Database URL
    """
    global _engine

    if _engine is None:
        _engine = get_engine(db_url)

    Base.metadata.drop_all(_engine)
    Base.metadata.create_all(_engine)
