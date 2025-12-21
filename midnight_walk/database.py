"""Database models and session management for the Midnight Walk server."""

from __future__ import annotations

import os
from typing import AsyncGenerator

from sqlalchemy import Column, Float, ForeignKey, String, Text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, declared_attr


class Base(DeclarativeBase):
    """Base class for all database models."""

    @declared_attr
    def __tablename__(cls) -> str:
        """Generate table name from class name."""
        return cls.__name__.lower()


class SpiritModel(Base):
    """Database model for spirits."""

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    blurb = Column(Text, nullable=False)  # Narrative text about the spirit
    image_url = Column(String, nullable=True)  # Optional image URL
    totem_image_url = Column(String, nullable=True)  # Totem silhouette/image URL
    current_activity = Column(Text, nullable=True)  # What they're up to in SF

    def __repr__(self) -> str:
        return f"<SpiritModel(id={self.id!r}, name={self.name!r})>"


class BeaconModel(Base):
    """Database model for beacons."""

    id = Column(String, primary_key=True)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    search_radius_meters = Column(Float, nullable=False)
    spirit_id = Column(String, ForeignKey("spiritmodel.id"), nullable=True)
    state = Column(String, nullable=False, server_default="undiscovered")

    def __repr__(self) -> str:
        return (
            f"<BeaconModel(id={self.id!r}, lat={self.latitude}, "
            f"lon={self.longitude}, radius={self.search_radius_meters}, "
            f"spirit_id={self.spirit_id!r}, state={self.state!r})>"
        )


class BeaconFindModel(Base):
    """Database model for logging when beacons are found via NFC."""

    id = Column(String, primary_key=True)
    beacon_id = Column(String, ForeignKey("beaconmodel.id"), nullable=False)
    found_at = Column(Float, nullable=False)  # Unix timestamp
    user_agent = Column(String, nullable=True)
    ip_address = Column(String, nullable=True)

    def __repr__(self) -> str:
        return (
            f"<BeaconFindModel(id={self.id!r}, beacon_id={self.beacon_id!r}, "
            f"found_at={self.found_at})>"
        )


class StoryEventModel(Base):
    """Database model for story log events."""

    id = Column(String, primary_key=True)
    created_at = Column(Float, nullable=False)
    event_type = Column(String, nullable=False)
    beacon_id = Column(String, ForeignKey("beaconmodel.id"), nullable=True)
    spirit_id = Column(String, ForeignKey("spiritmodel.id"), nullable=True)
    spirit_name = Column(String, nullable=True)
    player_name = Column(String, nullable=True)  # Name of player who triggered the event
    message = Column(Text, nullable=False)

    def __repr__(self) -> str:
        return (
            f"<StoryEventModel(id={self.id!r}, event={self.event_type!r}, "
            f"spirit={self.spirit_name!r}, player={self.player_name!r})>"
        )


class PlayerModel(Base):
    """Database model for tracking player names and their spirit encounters."""

    id = Column(String, primary_key=True)  # Player ID (stored in cookie)
    name = Column(String, nullable=False)
    created_at = Column(Float, nullable=False)
    # Track encounters: JSON array of spirit IDs
    encountered_spirits = Column(Text, nullable=True)  # JSON array of spirit IDs
    # Track vanquishes: JSON array of spirit IDs
    vanquished_spirits = Column(Text, nullable=True)  # JSON array of spirit IDs

    def __repr__(self) -> str:
        return (
            f"<PlayerModel(id={self.id!r}, name={self.name!r}, "
            f"encountered={self.encountered_spirits!r}, vanquished={self.vanquished_spirits!r})>"
        )


class TotemReturnModel(Base):
    """Database model for tracking which players have returned totems after vanquishing."""

    id = Column(String, primary_key=True)  # UUID
    player_id = Column(String, nullable=False)  # Player ID who returned
    beacon_id = Column(String, ForeignKey("beaconmodel.id"), nullable=False)
    spirit_id = Column(String, ForeignKey("spiritmodel.id"), nullable=False)
    returned_at = Column(Float, nullable=False)  # Unix timestamp
    new_latitude = Column(Float, nullable=False)
    new_longitude = Column(Float, nullable=False)

    def __repr__(self) -> str:
        return (
            f"<TotemReturnModel(id={self.id!r}, player_id={self.player_id!r}, "
            f"beacon_id={self.beacon_id!r}, spirit_id={self.spirit_id!r})>"
        )


class DisplayLocationModel(Base):
    """Database model for the rift gate (volumetric display) location."""

    id = Column(String, primary_key=True, default="rift_gate")
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    image_url = Column(String, nullable=True)

    def __repr__(self) -> str:
        return (
            f"<DisplayLocationModel(lat={self.latitude}, lon={self.longitude}, "
            f"image_url={self.image_url!r})>"
        )


class FakePlayerModel(Base):
    """Database model for fake players used for testing."""

    id = Column(String, primary_key=True)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    name = Column(String, nullable=True)  # Optional name/label

    def __repr__(self) -> str:
        return (
            f"<FakePlayerModel(id={self.id!r}, lat={self.latitude}, "
            f"lon={self.longitude}, name={self.name!r})>"
        )


class MinigameModel(Base):
    """Database model for spirit minigames."""

    id = Column(String, primary_key=True)
    spirit_id = Column(String, ForeignKey("spiritmodel.id"), nullable=False, unique=True)
    html_content = Column(Text, nullable=False)  # HTML content of the minigame

    def __repr__(self) -> str:
        return f"<MinigameModel(id={self.id!r}, spirit_id={self.spirit_id!r})>"


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
