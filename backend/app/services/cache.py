import hashlib
import json
import logging

import redis.asyncio as redis

from app.config import settings

logger = logging.getLogger("spendlens")

_redis: redis.Redis | None = None

DEFAULT_TTL = 3600  # 1 hour
ADVICE_TTL = 86400  # 24 hours


async def init_cache():
    global _redis
    _redis = redis.from_url(settings.redis_url, decode_responses=True)
    await _redis.ping()
    logger.info("Redis connection established")


async def close_cache():
    global _redis
    if _redis:
        await _redis.aclose()
        _redis = None
        logger.info("Redis connection closed")


def get_redis() -> redis.Redis:
    if _redis is None:
        raise RuntimeError("Redis not initialized")
    return _redis


def make_cache_key(prefix: str, **kwargs) -> str:
    """Generate a deterministic cache key from prefix + params."""
    raw = json.dumps(kwargs, sort_keys=True, default=str)
    h = hashlib.md5(raw.encode()).hexdigest()[:12]
    return f"spendlens:{prefix}:{h}"


async def cache_get(key: str) -> dict | None:
    r = get_redis()
    data = await r.get(key)
    if data:
        logger.debug(f"Cache HIT: {key}")
        return json.loads(data)
    logger.debug(f"Cache MISS: {key}")
    return None


async def cache_set(key: str, value: dict, ttl: int = DEFAULT_TTL):
    r = get_redis()
    await r.set(key, json.dumps(value, default=str), ex=ttl)


async def cache_invalidate(pattern: str):
    """Delete all keys matching a pattern."""
    r = get_redis()
    keys = []
    async for key in r.scan_iter(match=pattern):
        keys.append(key)
    if keys:
        await r.delete(*keys)
        logger.debug(f"Invalidated {len(keys)} cache keys matching {pattern}")
