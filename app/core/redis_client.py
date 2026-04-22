from typing import Annotated, TypeAlias

from fastapi import Depends
from langgraph.checkpoint.redis import AsyncRedisSaver, RedisSaver
from redis.asyncio import Redis
from starlette.requests import Request

from app.core.config import get_settings

settings = get_settings().persistent.cache
monitor_redis_client = Redis.from_url(settings.MONITOR_CACHE.unicode_string())


async def setup_cache_redis() -> Redis:
    """初始化缓存."""
    redis = Redis.from_url(settings.API_CACHE.unicode_string(), decode_responses=True)
    await redis.ping()

    async with AsyncRedisSaver.from_conn_string(settings.LLM_CACHE.unicode_string()) as checkpoint:
        await checkpoint.asetup()

    return redis


async def close_cache_regis(redis: Redis) -> None:
    """关闭缓存链接."""
    await redis.aclose()


async def _get_redis(request: Request) -> Redis:
    """全局请求缓存."""
    return request.app.state.redis


RedisDep: TypeAlias = Annotated[Redis, Depends(_get_redis)]
