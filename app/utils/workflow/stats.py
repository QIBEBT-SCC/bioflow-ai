"""运行统计信息

负责从 Redis 查询运行实例的统计信息
"""

from typing import Optional

from loguru import logger
from redis import Redis
from redis.asyncio import Redis as AsyncRedis

from app.schemas.common import Statistics


async def get_run_stats_from_redis(redis_client: AsyncRedis, run_uid: str) -> Optional[Statistics]:
    """从 Redis Hash 中获取运行实例的统计信息（异步版本）

    Args:
        redis_client: 异步 Redis 客户端
        run_uid: 运行实例 UID

    Returns:
        Statistics: 统计信息对象，如果不存在则返回 None
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
    """从 Redis Hash 中获取运行实例的统计信息（同步版本）

    Args:
        redis_client: 同步 Redis 客户端
        run_uid: 运行实例 UID

    Returns:
        Statistics: 统计信息对象，如果不存在则返回 None
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
