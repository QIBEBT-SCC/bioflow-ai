"""Process-local caching for reusable values."""

import time
from collections.abc import Hashable

from loguru import logger
from pydantic import BaseModel


class CachedItem[ValueT](BaseModel):
    value: ValueT
    last_used: float
    ttl: int | None


class MemoryCache[KeyT: Hashable, ValueT]:
    """Cache values with an idle timeout renewed on each successful lookup."""

    def __init__(self) -> None:
        self._cache: dict[KeyT, CachedItem[ValueT]] = {}

    def get(self, key: KeyT) -> ValueT | None:
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

    def set(self, key: KeyT, value: ValueT, ttl: int | None = None) -> None:
        """Store a value with an optional idle timeout in seconds."""
        self._cache[key] = CachedItem(value=value, last_used=time.time(), ttl=ttl)

    def delete(self, key: KeyT) -> None:
        """Remove one cached value if it exists."""
        self._cache.pop(key, None)

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
