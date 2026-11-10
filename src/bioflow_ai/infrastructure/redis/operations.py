"""Shared Redis helpers and distributed coordination primitives."""

import asyncio
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager, suppress
from typing import Literal

from loguru import logger
from redis.asyncio import Redis
from redis.asyncio.lock import Lock
from redis.exceptions import LockError

from bioflow_ai.core.config import get_settings


async def _renew_lease(lock: Lock, kind: str, owner_task: asyncio.Task[object]) -> None:
    """Keep a long generation lease alive and cancel the owner if the lease is lost."""
    settings = get_settings().persistent.cache
    try:
        while True:
            await asyncio.sleep(settings.CATALOG_GENERATION_LOCK_RENEW_INTERVAL_SECONDS)
            if not await lock.extend(settings.CATALOG_GENERATION_LOCK_TIMEOUT_SECONDS, replace_ttl=True):
                logger.error('Lost Redis {} catalog generation lock', kind)
                owner_task.cancel()
                return
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception('Unable to renew {} catalog generation lock', kind)
        owner_task.cancel()


@asynccontextmanager
async def catalog_generation_lock(
    kind: Literal['tool'],
    *,
    on_wait: Callable[[], None] | None = None,
) -> AsyncGenerator[None]:
    """Serialize catalog generation across processes with a renewed Redis lease.

    Only creators take this lock. Searches stay lock-free; a creator re-checks the catalog after
    acquiring the lock, so anything committed by the previous holder is visible to it.

    Args:
        kind: Catalog kind whose generation is serialized.
        on_wait: Called once when the lock is held by another generator and this call must wait.
    """
    owner_task = asyncio.current_task()
    if owner_task is None:
        msg = 'Catalog generation lock must be held inside an asyncio task'
        raise RuntimeError(msg)

    settings = get_settings().persistent.cache
    redis = Redis.from_url(settings.API_CACHE.unicode_string(), decode_responses=True)
    try:
        lock = redis.lock(
            f'{settings.CATALOG_GENERATION_KEY_PREFIX}:{kind}:lock',
            timeout=settings.CATALOG_GENERATION_LOCK_TIMEOUT_SECONDS,
            thread_local=False,
        )
        if not await lock.acquire(blocking=False):
            if on_wait is not None:
                on_wait()
            await lock.acquire(blocking=True)

        renew_task = asyncio.create_task(_renew_lease(lock, kind, owner_task), name=f'{kind}-catalog-lock-renewal')
        try:
            yield
        finally:
            renew_task.cancel()
            with suppress(asyncio.CancelledError):
                await renew_task
            try:
                await lock.release()
            except LockError:
                logger.warning('{} catalog generation lock was no longer owned during release', kind)
    finally:
        await redis.aclose()


def decode_redis_value(value: bytes | str | None) -> str | None:
    """Decode a Redis value returned as bytes, str, or None into str or None."""
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode()
    return value


def get_bio_db_download_task_key(db_id: int) -> str:
    """Return the Redis key for the active BioDb download task."""
    return f'bio-db:{db_id}:download-task-id'
