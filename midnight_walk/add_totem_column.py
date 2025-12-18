#!/usr/bin/env python3
"""Quick script to add totem_image_url column to spiritmodel table."""
import asyncio

from sqlalchemy import text

from midnight_walk.database import create_async_engine_from_url, get_database_url


async def main():
    database_url = get_database_url()
    engine = create_async_engine_from_url(database_url)
    async with engine.begin() as conn:
        await conn.execute(
            text("ALTER TABLE spiritmodel ADD COLUMN IF NOT EXISTS totem_image_url TEXT")
        )
    await engine.dispose()
    print("Column added successfully!")


if __name__ == "__main__":
    asyncio.run(main())
