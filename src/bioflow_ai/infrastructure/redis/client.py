import asyncio
from typing import Annotated, TypeAlias

from fastapi import Depends
from loguru import logger
from redis import Redis as SyncRedis
from redis.asyncio import Redis
from redis.exceptions import AuthenticationError, AuthorizationError
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError
from starlette.requests import Request

from bioflow_ai.core.config import get_settings

settings = get_settings().persistent.cache
monitor_redis_client = Redis.from_url(settings.MONITOR_CACHE.unicode_string())
# Celery tasks run synchronously, so they share the monitor cache through a blocking client.
sync_monitor_redis_client = SyncRedis.from_url(settings.MONITOR_CACHE.unicode_string())
notification_redis_client = Redis.from_url(settings.NOTIFICATION_CACHE.unicode_string(), decode_responses=True)


async def wait_for_cache_redis(*, timeout: float = 120, poll_interval: float = 2) -> None:
    """Wait for the API's Redis dependencies before startup initialization."""
    caches = {
        'API cache': settings.API_CACHE,
        'notifications': settings.NOTIFICATION_CACHE,
    }
    try:
        async with asyncio.timeout(timeout):
            for name, url in caches.items():
                async with Redis.from_url(url.unicode_string()) as redis:
                    while True:
                        try:
                            await asyncio.wait_for(redis.ping(), timeout=5)
                        except (AuthenticationError, AuthorizationError):
                            raise
                        except (RedisConnectionError, RedisTimeoutError, TimeoutError) as error:
                            logger.warning(
                                '{} Redis is not ready ({}); retrying in {} seconds.',
                                name,
                                type(error).__name__,
                                poll_interval,
                            )
                            await asyncio.sleep(poll_interval)
                        else:
                            break
    except TimeoutError as error:
        message = f'Timed out waiting for Redis readiness after {timeout:g} seconds.'
        raise RuntimeError(message) from error

    logger.info('Redis dependencies are ready; continuing API startup.')


async def setup_cache_redis() -> Redis:
    """初始化缓存."""
    redis = Redis.from_url(settings.API_CACHE.unicode_string(), decode_responses=True)
    await redis.ping()
    return redis


async def close_cache_regis(redis: Redis) -> None:
    """关闭缓存链接."""
    await redis.aclose()


async def _get_redis(request: Request) -> Redis:
    """全局请求缓存."""
    return request.app.state.redis


RedisDep: TypeAlias = Annotated[Redis, Depends(_get_redis)]
