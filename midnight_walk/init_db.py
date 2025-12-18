#!/usr/bin/env python3
"""Initialize the database by creating all tables."""
import asyncio

from midnight_walk.database import (
    create_async_engine_from_url,
    get_database_url,
    init_database,
)


async def main():
    database_url = get_database_url()
    engine = create_async_engine_from_url(database_url)
    await init_database(engine)
    await engine.dispose()
    print("Database initialized successfully!")


if __name__ == "__main__":
    asyncio.run(main())
