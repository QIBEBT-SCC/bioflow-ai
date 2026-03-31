from langchain_redis import RedisCache
from redis.asyncio import Redis

from app.core.config import get_settings

settings = get_settings().persistent.cache
monitor_redis_client = Redis.from_url(settings.MONITOR_CACHE)
llm_cache = RedisCache(redis_url=settings.LLM_CACHE)


async def setup_cache_redis() -> Redis:
    """初始化缓存."""
    redis = Redis.from_url(settings.API_CACHE, decode_responses=True)
    await redis.ping()
    return redis


async def close_cache_regis(redis: Redis) -> None:
    """关闭缓存链接."""
    await redis.aclose()
