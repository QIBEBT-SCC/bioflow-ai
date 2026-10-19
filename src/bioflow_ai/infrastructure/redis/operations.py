"""Shared Redis helpers and distributed coordination primitives."""

import asyncio
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager, suppress
from typing import Literal

from loguru import logger
from pydantic import BaseModel, ConfigDict
from redis.asyncio import Redis
from redis.asyncio.lock import Lock
from redis.exceptions import LockError

from bioflow_ai.core.config import get_settings


class CreationPermit(BaseModel):
    """Whether a workflow may create a tool catalog entry from its last search."""

    model_config = ConfigDict(frozen=True)

    allowed: bool
    reason: Literal['missing_search', 'stale_search'] | None = None


class CatalogGenerationGuard:
    """Coordinate tool catalog searches and workflow-initiated generation through Redis."""

    def __init__(
        self,
        redis: Redis,
        kind: Literal['tool'],
        thread_id: str,
        *,
        on_wait: Callable[[], None] | None = None,
    ) -> None:
        self.redis = redis
        self.kind = kind
        self.thread_id = thread_id
        self.on_wait = on_wait
        self.settings = get_settings().persistent.cache

    @property
    def lock_key(self) -> str:
        """Return the global generation-lock key for this catalog kind."""
        return f'{self.settings.CATALOG_GENERATION_KEY_PREFIX}:{self.kind}:lock'

    @property
    def epoch_key(self) -> str:
        """Return the global generation epoch key for this catalog kind."""
        return f'{self.settings.CATALOG_GENERATION_KEY_PREFIX}:{self.kind}:epoch'

    @property
    def seen_epoch_key(self) -> str:
        """Return the key recording the epoch observed by this workflow."""
        return f'{self.settings.CATALOG_GENERATION_KEY_PREFIX}:{self.kind}:seen:{self.thread_id}'

    async def _acquire(self, lock: Lock) -> None:
        if await lock.acquire(blocking=False):
            return
        if self.on_wait is not None:
            self.on_wait()
        await lock.acquire(blocking=True)

    async def _renew(self, lock: Lock, owner_task: asyncio.Task[object]) -> None:
        """Keep long-running tool generation leases alive and cancel work if ownership is lost."""
        try:
            while True:
                await asyncio.sleep(self.settings.CATALOG_GENERATION_LOCK_RENEW_INTERVAL_SECONDS)
                if not await lock.extend(
                    self.settings.CATALOG_GENERATION_LOCK_TIMEOUT_SECONDS,
                    replace_ttl=True,
                ):
                    logger.error('Lost Redis {} catalog generation lock', self.kind)
                    owner_task.cancel()
                    return
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception('Unable to renew {} catalog generation lock', self.kind)
            owner_task.cancel()

    @asynccontextmanager
    async def _locked(self) -> AsyncGenerator[None]:
        lock = self.redis.lock(
            self.lock_key,
            timeout=self.settings.CATALOG_GENERATION_LOCK_TIMEOUT_SECONDS,
            thread_local=False,
        )
        await self._acquire(lock)
        owner_task = asyncio.current_task()
        if owner_task is None:
            with suppress(LockError):
                await lock.release()
            msg = 'Catalog generation guard must run inside an asyncio task'
            raise RuntimeError(msg)

        renew_task = asyncio.create_task(
            self._renew(lock, owner_task),
            name=f'{self.kind}-catalog-lock-renewal',
        )
        try:
            yield
        finally:
            renew_task.cancel()
            with suppress(asyncio.CancelledError):
                await renew_task
            try:
                if await lock.owned():
                    await lock.release()
            except LockError:
                logger.warning('{} catalog generation lock was no longer owned during release', self.kind)

    async def _current_epoch(self) -> int:
        raw_epoch = await self.redis.get(self.epoch_key)
        return int(raw_epoch) if raw_epoch is not None else 0

    @asynccontextmanager
    async def search(self) -> AsyncGenerator[None]:
        """Run a catalog search atomically with recording the observed generation epoch."""
        async with self._locked():
            yield
            epoch = await self._current_epoch()
            await self.redis.set(
                self.seen_epoch_key,
                epoch,
                ex=self.settings.CATALOG_GENERATION_SEARCH_EPOCH_TTL_SECONDS,
            )

    @asynccontextmanager
    async def creation(self) -> AsyncGenerator[CreationPermit]:
        """Admit generation only when the workflow's most recent search is still current."""
        async with self._locked():
            raw_seen_epoch = await self.redis.get(self.seen_epoch_key)
            if raw_seen_epoch is None:
                yield CreationPermit(allowed=False, reason='missing_search')
                return

            current_epoch = await self._current_epoch()
            if int(raw_seen_epoch) != current_epoch:
                yield CreationPermit(allowed=False, reason='stale_search')
                return

            await self.redis.incr(self.epoch_key)
            yield CreationPermit(allowed=True)


@asynccontextmanager
async def open_catalog_generation_guard(
    kind: Literal['tool'],
    thread_id: str,
    *,
    on_wait: Callable[[], None] | None = None,
) -> AsyncGenerator[CatalogGenerationGuard]:
    """Open a loop-local Redis client for one guarded manager operation."""
    redis = Redis.from_url(
        get_settings().persistent.cache.API_CACHE.unicode_string(),
        decode_responses=True,
    )
    try:
        yield CatalogGenerationGuard(redis, kind, thread_id, on_wait=on_wait)
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
