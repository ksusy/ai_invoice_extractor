"""Async database connection and session management.

Provides the async engine, session factory, and dependency injection
for FastAPI routes.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src.config.settings import get_settings
from src.infrastructure.db.models import Base

# ────────────────────────────────────────────────────────────────────────────
# Engine and session factory (lazy initialization)
# ────────────────────────────────────────────────────────────────────────────

_engine: AsyncEngine | None = None
_async_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    """Return the async SQLAlchemy engine (singleton)."""
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_async_engine(
            str(settings.database_url),
            echo=settings.debug,
            future=True,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
        )
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Return the async session factory (singleton)."""
    global _async_session_factory
    if _async_session_factory is None:
        engine = get_engine()
        _async_session_factory = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )
    return _async_session_factory


# ────────────────────────────────────────────────────────────────────────────
# Session dependency for FastAPI
# ────────────────────────────────────────────────────────────────────────────


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """Yield an async database session (FastAPI dependency).

    Usage in routes:
        @router.get("/")
        async def get_invoices(session: AsyncSession = Depends(get_async_session)):
            ...
    """
    session_factory = get_session_factory()
    async with session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


@asynccontextmanager
async def get_session_context() -> AsyncGenerator[AsyncSession, None]:
    """Context manager for use outside of FastAPI routes.

    Usage:
        async with get_session_context() as session:
            result = await session.execute(...)
    """
    session_factory = get_session_factory()
    async with session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# ────────────────────────────────────────────────────────────────────────────
# Database lifecycle management
# ────────────────────────────────────────────────────────────────────────────


async def init_db() -> None:
    """Create all tables defined in models.

    WARNING: Use Alembic migrations in production. This is for
    development/testing only.
    """
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def drop_db() -> None:
    """Drop all tables. Use with caution!"""
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


async def close_db() -> None:
    """Close the database engine and release connections."""
    global _engine, _async_session_factory
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _async_session_factory = None
