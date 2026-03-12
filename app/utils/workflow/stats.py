"""Runtime Statistics.

Responsible for querying runtime instance statistics from Redis.
"""

from typing import Optional

from loguru import logger
from redis import Redis
from redis.asyncio import Redis as AsyncRedis

from app.schemas.common import Statistics


async def get_run_stats_from_redis(redis_client: AsyncRedis, run_uid: str) -> Optional[Statistics]:
    """Retrieve statistics of a running instance from Redis Hash (async version).

    Args:
        redis_client: Async Redis client
        run_uid: Running instance UID

    Returns:
        Statistics: Statistics object, or None if it does not exist
    """
    redis_key = f'run-stats-{run_uid}'

    try:
        if not await redis_client.exists(redis_key):
            return None

        stats = await redis_client.hgetall(redis_key)
        if not stats:
            return None

        return Statistics(
            total=int(stats.get(b'total', 0)),
            waiting=int(stats.get(b'waiting', 0)),
            running=int(stats.get(b'running', 0)),
            success=int(stats.get(b'success', 0)),
            error=int(stats.get(b'error', 0)),
        )
    except Exception as e:
        logger.error(f'从 Redis 读取统计信息失败: {e}')
        return None


def get_run_stats_from_redis_sync(redis_client: Redis, run_uid: str) -> Optional[Statistics]:
    """Retrieve statistics for a running instance from a Redis Hash (synchronous version).

    Args:
        redis_client: Synchronous Redis client
        run_uid: Running instance UID

    Returns:
        Statistics: Statistics object, or None if it does not exist
    """
    redis_key = f'run-stats-{run_uid}'

    try:
        if not redis_client.exists(redis_key):
            return None

        stats = redis_client.hgetall(redis_key)
        if not stats:
            return None

        return Statistics(
            total=int(stats.get(b'total', 0)),
            waiting=int(stats.get(b'waiting', 0)),
            running=int(stats.get(b'running', 0)),
            success=int(stats.get(b'success', 0)),
            error=int(stats.get(b'error', 0)),
        )
    except Exception as e:
        logger.error(f'从 Redis 读取统计信息失败: {e}')
        return None
