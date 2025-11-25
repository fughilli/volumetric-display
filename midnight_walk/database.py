"""Database models and session management for the Midnight Walk server."""

from __future__ import annotations

import os
from typing import AsyncGenerator

from sqlalchemy import Column, Float, String
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, declared_attr


class Base(DeclarativeBase):
    """Base class for all database models."""

    @declared_attr
    def __tablename__(cls) -> str:
        """Generate table name from class name."""
        return cls.__name__.lower()


class BeaconModel(Base):
    """Database model for beacons."""

    id = Column(String, primary_key=True)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    search_radius_meters = Column(Float, nullable=False)

    def __repr__(self) -> str:
        return (
            f"<BeaconModel(id={self.id!r}, lat={self.latitude}, "
            f"lon={self.longitude}, radius={self.search_radius_meters})>"
        )


def get_database_url() -> str:
    """Get the database URL from environment variables or use SQLite default."""
    # Check for PostgreSQL (Heroku or explicit DATABASE_URL)
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        # Heroku provides postgres:// but SQLAlchemy needs postgresql+asyncpg://
        if database_url.startswith("postgres://"):
            database_url = database_url.replace("postgres://", "postgresql+asyncpg://", 1)
        elif not database_url.startswith("postgresql+asyncpg://"):
            database_url = f"postgresql+asyncpg://{database_url}"
        return database_url

    # Default to SQLite for local development
    db_path = os.getenv("MIDNIGHT_DB_PATH", "midnight_walk.db")
    return f"sqlite+aiosqlite:///{db_path}"


def create_async_engine_from_url(url: str):
    """Create an async SQLAlchemy engine from a URL."""
    return create_async_engine(
        url,
        echo=os.getenv("SQL_ECHO", "0") == "1",  # Enable SQL logging for debugging
        future=True,
    )


async def init_database(engine) -> None:
    """Initialize the database by creating all tables."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


# Global engine and session maker (initialized in create_app)
_engine = None
_async_session_maker = None


def set_db_engine(engine):
    """Set the global database engine and session maker."""
    global _engine, _async_session_maker
    _engine = engine
    _async_session_maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency function to get a database session."""
    if _async_session_maker is None:
        # Fallback: create engine if not set (for testing)
        engine = create_async_engine_from_url(get_database_url())
        async_session_maker = async_sessionmaker(
            engine, class_=AsyncSession, expire_on_commit=False
        )
    else:
        async_session_maker = _async_session_maker

    async with async_session_maker() as session:
        try:
            yield session
        finally:
            await session.close()
