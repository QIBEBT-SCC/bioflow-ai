"""Shared, generation-checked snapshots of persisted tool definitions."""

from collections.abc import Awaitable, Generator, Iterable, MutableMapping
from contextlib import contextmanager
from typing import Any, cast

from loguru import logger
from redis import Redis as SyncRedis
from redis.asyncio import Redis
from redis.exceptions import RedisError
from sqlalchemy.orm import Session

import bioflow_ai.crud.tools.tool as tool_crud
from bioflow_ai.schemas.tools import ToolNodePublic, ToolTaskInfo, WorkflowToolDefinition
from bioflow_ai.utils.identifiers import UUIDLike, as_uuid

TOOL_CACHE_TTL_SECONDS = 60 * 60 * 24
TOOL_DEFINITION_TTL_SECONDS = 300
_EPOCH_KEY = 'workflow-tool-definition:v1:epoch'
_PUBLISH = """
if (redis.call('GET', KEYS[1]) or '0') ~= ARGV[1] or
   (redis.call('GET', KEYS[2]) or '0') ~= ARGV[2] then return 0 end
redis.call('SET', KEYS[3], ARGV[3], 'EX', ARGV[4])
return 1
"""
_INVALIDATE = """
redis.call('INCR', KEYS[1])
for i = 2, #KEYS do redis.call('DEL', KEYS[i]) end
return 1
"""


def _uid(value: UUIDLike) -> str:
    return str(as_uuid(value))


def _definition_key(tool_uid: UUIDLike, epoch: str = '0') -> str:
    return f'workflow-tool-definition:v1:{epoch}:{_uid(tool_uid)}'


def _generation_key(tool_uid: UUIDLike) -> str:
    return f'workflow-tool-definition:v1:generation:{_uid(tool_uid)}'


def _tool_args_key(tool_uid: UUIDLike) -> str:
    return f'tool-args:{_uid(tool_uid)}'


def _tool_task_info_key(tool_uid: UUIDLike, epoch: str | None = None) -> str:
    prefix = f'tool-task-info:v2:{epoch}' if epoch is not None else 'tool-task-info'
    return f'{prefix}:{_uid(tool_uid)}'


def _text(value: Any) -> str:
    return value.decode() if isinstance(value, bytes) else str(value or '0')


async def _eval_async(redis: Redis, script: str, count: int, *args: str) -> Any:
    return await cast('Awaitable[Any]', redis.eval(script, count, *args))


def _warn(error: RedisError) -> None:
    logger.warning('Tool cache unavailable; using database ({})', type(error).__name__)


@contextmanager
def tool_cache_client() -> Generator[SyncRedis, None, None]:
    """Open a bounded synchronous client for synchronous builder operations."""
    from bioflow_ai.core.config import get_settings

    with SyncRedis.from_url(
        get_settings().persistent.cache.API_CACHE.unicode_string(),
        decode_responses=True,
        socket_connect_timeout=1,
        socket_timeout=1,
    ) as redis:
        yield redis


def _load(session: Session, uids: list[str], *, fresh: bool = False) -> dict[str, WorkflowToolDefinition]:
    if not uids:
        return {}
    tools = tool_crud.get_all_by_uid(session, uids, refresh=True) if fresh else tool_crud.get_all_by_uid(session, uids)
    return {str(tool.uid): WorkflowToolDefinition.model_validate(tool) for tool in tools}


def _decode(raw: str | bytes | None) -> WorkflowToolDefinition | None:
    if not raw:
        return None
    try:
        return WorkflowToolDefinition.model_validate_json(raw)
    except ValueError:
        return None


def get_tool_definitions_sync(
    session: Session,
    tool_uids: Iterable[UUIDLike],
    *,
    redis: SyncRedis | None = None,
    fresh: bool = False,
    snapshot: MutableMapping[str, WorkflowToolDefinition] | None = None,
) -> dict[str, WorkflowToolDefinition]:
    """Resolve each unique UID once; fresh reads never use cached snapshots."""
    uids = list(dict.fromkeys(_uid(uid) for uid in tool_uids))
    if not uids:
        return {}
    if fresh:
        return _load(session, uids, fresh=True)
    result = {uid: snapshot[uid] for uid in uids if snapshot is not None and uid in snapshot}
    missing = [uid for uid in uids if uid not in result]
    if not missing:
        return result
    if redis is None:
        with tool_cache_client() as client:
            result.update(get_tool_definitions_sync(session, missing, redis=client))
    else:
        epoch, generations = '0', {}
        available = True
        try:
            epoch = _text(redis.get(_EPOCH_KEY))
            generations = dict(
                zip(
                    missing,
                    map(_text, cast('list[Any]', redis.mget([_generation_key(uid) for uid in missing]))),
                    strict=True,
                )
            )
            raw_values = cast('list[Any]', redis.mget([_definition_key(uid, epoch) for uid in missing]))
            for uid, raw in zip(missing, raw_values, strict=True):
                definition = _decode(raw)
                if definition is not None and str(definition.uid) == uid:
                    result[uid] = definition
                elif raw:
                    redis.delete(_definition_key(uid, epoch))
        except RedisError as error:
            _warn(error)
            available = False
        loaded = _load(session, [uid for uid in missing if uid not in result])
        result.update(loaded)
        if available:
            try:
                for uid, definition in loaded.items():
                    redis.eval(
                        _PUBLISH,
                        3,
                        _EPOCH_KEY,
                        _generation_key(uid),
                        _definition_key(uid, epoch),
                        epoch,
                        generations[uid],
                        definition.model_dump_json(),
                        str(TOOL_DEFINITION_TTL_SECONDS),
                    )
            except RedisError as error:
                _warn(error)
    if snapshot is not None:
        snapshot.update(result)
    return result


async def get_tool_definitions(
    session: Session,
    redis: Redis,
    tool_uids: Iterable[UUIDLike],
    *,
    fresh: bool = False,
    snapshot: MutableMapping[str, WorkflowToolDefinition] | None = None,
) -> dict[str, WorkflowToolDefinition]:
    """Asynchronous cache adapter sharing DTOs, key format and database batching."""
    uids = list(dict.fromkeys(_uid(uid) for uid in tool_uids))
    if not uids:
        return {}
    if fresh:
        return _load(session, uids, fresh=True)
    result = {uid: snapshot[uid] for uid in uids if snapshot is not None and uid in snapshot}
    missing = [uid for uid in uids if uid not in result]
    if not missing:
        return result
    epoch, generations = '0', {}
    available = True
    try:
        epoch = _text(await redis.get(_EPOCH_KEY))
        generations = dict(
            zip(missing, map(_text, await redis.mget([_generation_key(uid) for uid in missing])), strict=True)
        )
        raw_values = await redis.mget([_definition_key(uid, epoch) for uid in missing])
        for uid, raw in zip(missing, raw_values, strict=True):
            definition = _decode(raw)
            if definition is not None and str(definition.uid) == uid:
                result[uid] = definition
            elif raw:
                await redis.delete(_definition_key(uid, epoch))
    except RedisError as error:
        _warn(error)
        available = False
    loaded = _load(session, [uid for uid in missing if uid not in result])
    result.update(loaded)
    if available:
        try:
            for uid, definition in loaded.items():
                await _eval_async(
                    redis,
                    _PUBLISH,
                    3,
                    _EPOCH_KEY,
                    _generation_key(uid),
                    _definition_key(uid, epoch),
                    epoch,
                    generations[uid],
                    definition.model_dump_json(),
                    str(TOOL_DEFINITION_TTL_SECONDS),
                )
        except RedisError as error:
            _warn(error)
    if snapshot is not None:
        snapshot.update(result)
    return result


async def invalidate_tool_projections(redis: Redis, tool_uid: UUIDLike) -> None:
    """Invalidate after commit, including both older public projection keys."""
    try:
        epoch = _text(await redis.get(_EPOCH_KEY))
        await _eval_async(
            redis,
            _INVALIDATE,
            5,
            _generation_key(tool_uid),
            _definition_key(tool_uid, epoch),
            _tool_args_key(tool_uid),
            _tool_task_info_key(tool_uid),
            _tool_task_info_key(tool_uid, epoch),
        )
    except RedisError as error:
        logger.error('Tool cache invalidation failed for {} ({})', tool_uid, type(error).__name__)


def invalidate_tool_projections_sync(tool_uid: UUIDLike, *, redis: SyncRedis | None = None) -> None:
    """Synchronous post-commit invalidation for generation and maintenance paths."""
    if redis is None:
        with tool_cache_client() as client:
            invalidate_tool_projections_sync(tool_uid, redis=client)
        return
    try:
        epoch = _text(redis.get(_EPOCH_KEY))
        redis.eval(
            _INVALIDATE,
            5,
            _generation_key(tool_uid),
            _definition_key(tool_uid, epoch),
            _tool_args_key(tool_uid),
            _tool_task_info_key(tool_uid),
            _tool_task_info_key(tool_uid, epoch),
        )
    except RedisError as error:
        logger.error('Tool cache invalidation failed for {} ({})', tool_uid, type(error).__name__)


def invalidate_all_tool_definitions_sync(*, redis: SyncRedis | None = None) -> None:
    """Rotate the definition namespace after a bulk catalog replacement."""
    if redis is None:
        with tool_cache_client() as client:
            invalidate_all_tool_definitions_sync(redis=client)
        return
    try:
        redis.incr(_EPOCH_KEY)
    except RedisError as error:
        logger.error('Bulk tool cache invalidation failed ({})', type(error).__name__)


async def get_tool_node(session: Session, redis: Redis, tool_uid: UUIDLike) -> ToolNodePublic | None:
    """Build the existing node projection from a complete cached definition."""
    definition = (await get_tool_definitions(session, redis, [tool_uid])).get(_uid(tool_uid))
    return definition.to_node() if definition is not None else None


async def get_tool_execution_config(session: Session, redis: Redis, tool_uid: UUIDLike) -> ToolTaskInfo | None:
    """Keep execution projection policy while tolerating cache outages."""
    epoch, generation = '0', '0'
    cache_key = _tool_task_info_key(tool_uid, epoch)
    available = True
    try:
        epoch = _text(await redis.get(_EPOCH_KEY))
        generation = _text(await redis.get(_generation_key(tool_uid)))
        cache_key = _tool_task_info_key(tool_uid, epoch)
        cached = await redis.get(cache_key)
        if cached:
            try:
                return ToolTaskInfo.model_validate_json(cached)
            except ValueError:
                await redis.delete(cache_key)
    except RedisError as error:
        _warn(error)
        available = False
    tool = tool_crud.get(session, tool_uid)
    if tool is None:
        return None
    projection = ToolTaskInfo.model_validate(tool)
    if available:
        try:
            await _eval_async(
                redis,
                _PUBLISH,
                3,
                _EPOCH_KEY,
                _generation_key(tool_uid),
                cache_key,
                epoch,
                generation,
                projection.model_dump_json(),
                str(TOOL_CACHE_TTL_SECONDS),
            )
        except RedisError as error:
            _warn(error)
    return projection
