from redis.asyncio import Redis

from app.core.config import get_settings

monitor_redis_client = Redis.from_url(get_settings().persistent.MONITOR_CACHE)


async def setup_cache_redis() -> Redis:
    redis = Redis.from_url(get_settings().persistent.API_CACHE, decode_responses=True)
    await redis.ping()
    return redis


async def close_cache_regis(redis: Redis):
    await redis.aclose()
