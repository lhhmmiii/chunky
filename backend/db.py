"""
Async SQLAlchemy engine and session factory.

Usage
-----
In routers / services, use :func:`get_async_session` as a FastAPI dependency::

    from backend.db import get_async_session
    from sqlalchemy.ext.asyncio import AsyncSession

    @router.get("/example")
    async def example(session: AsyncSession = Depends(get_async_session)):
        ...

Tables are created automatically at application startup via
``async_engine.begin()`` in ``main.py``.
"""

from __future__ import annotations

from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from backend.config import get_settings


def _make_engine():
    settings = get_settings()
    return create_async_engine(
        settings.DATABASE_URL,
        echo=False,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
    )


# Module-level singletons — lazily initialised so worker processes that never
# touch the DB don't pay the connection overhead.
_engine = None
_session_factory = None


def get_engine():
    """Return the module-level async engine, creating it on first call."""
    global _engine
    if _engine is None:
        _engine = _make_engine()
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Return the module-level session factory, creating it on first call."""
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _session_factory


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency that yields a database session per request."""
    async with get_session_factory()() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


class Base(DeclarativeBase):
    """Declarative base shared by all ORM models."""
    pass
