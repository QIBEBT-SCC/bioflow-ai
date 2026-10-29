"""Process-local caching for reusable values and function results."""

import hashlib
import json
import time
from collections.abc import Callable
from functools import wraps
from typing import Any

from loguru import logger
from pydantic import BaseModel


class CachedItem(BaseModel):
    value: Any
    last_used: float
    ttl: int | None


class MemoryCache:
    """Cache values with an idle timeout renewed on each successful lookup."""

    def __init__(self) -> None:
        self._cache: dict[str, CachedItem] = {}

    def get(self, key: str) -> Any | None:
        """Return a cached value and renew its timeout, or return None on a miss."""
        if key not in self._cache:
            return None

        item = self._cache[key]
        current_time = time.time()

        # 检查是否过期
        if item.ttl is not None and current_time - item.last_used > item.ttl:
            del self._cache[key]
            return None

        # 更新最后使用时间
        item.last_used = current_time
        return item.value

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Store a value with an optional idle timeout in seconds."""
        self._cache[key] = CachedItem(value=value, last_used=time.time(), ttl=ttl)

    def clear(self) -> None:
        """Remove all cached values."""
        self._cache.clear()
        logger.info('已清除所有缓存')

    def clear_expired(self) -> None:
        """Remove cached values whose idle timeout has expired."""
        current_time = time.time()
        expired_keys = [
            key
            for key, item in self._cache.items()
            if item.ttl is not None and current_time - item.last_used > item.ttl
        ]
        for key in expired_keys:
            del self._cache[key]
        if expired_keys:
            logger.info(f'已清除 {len(expired_keys)} 个过期的缓存')


# 全局缓存实例
_cache = MemoryCache()


def cache_result(ttl: int | None = 3600) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Cache synchronous function results in the current process.

    Args:
        ttl: Idle timeout in seconds, renewed on each cache hit. None disables expiration.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # 创建缓存键
            cache_key_dict = {
                'func_name': func.__name__,
                'args': args,
                'kwargs': {k: str(v) for k, v in kwargs.items()},
            }
            cache_key = hashlib.md5(
                json.dumps(cache_key_dict, sort_keys=True).encode(), usedforsecurity=False
            ).hexdigest()

            # 尝试从缓存获取
            cached_value = _cache.get(cache_key)
            if cached_value is not None:
                logger.debug(f'使用缓存的函数结果: {func.__name__}')
                return cached_value

            # 未命中时执行原函数并缓存结果
            result = func(*args, **kwargs)
            _cache.set(cache_key, result, ttl)
            return result

        return wrapper

    return decorator
