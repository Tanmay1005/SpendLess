import logging

import asyncpg

from app.config import settings

logger = logging.getLogger("spendlens")

pool: asyncpg.Pool | None = None


async def init_db():
    """Create the asyncpg connection pool."""
    global pool
    logger.info(f"Connecting to database...")
    pool = await asyncpg.create_pool(
        settings.database_url,
        min_size=2,
        max_size=10,
    )
    logger.info("Database connection pool created")


async def close_db():
    """Close the connection pool."""
    global pool
    if pool:
        await pool.close()
        pool = None
        logger.info("Database connection pool closed")


def get_pool() -> asyncpg.Pool:
    """Get the connection pool. Raises if not initialized."""
    if pool is None:
        raise RuntimeError("Database pool not initialized")
    return pool
