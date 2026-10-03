"""Cached projections of persisted tool definitions."""

from redis.asyncio import Redis
from sqlalchemy.orm import Session

import bioflow_ai.crud.tools.tool as tool_crud
from bioflow_ai.schemas.tools import ToolNodePublic, ToolTaskInfo
from bioflow_ai.utils.identifiers import UUIDLike

TOOL_CACHE_TTL_SECONDS = 60 * 60 * 24


def _tool_args_key(tool_uid: UUIDLike) -> str:
    return f'tool-args:{tool_uid}'


def _tool_task_info_key(tool_uid: UUIDLike) -> str:
    return f'tool-task-info:{tool_uid}'


async def invalidate_tool_projections(redis: Redis, tool_uid: UUIDLike) -> None:
    """Remove every cached projection derived from a tool definition."""
    await redis.delete(_tool_args_key(tool_uid), _tool_task_info_key(tool_uid))


async def get_tool_node(
    session: Session,
    redis: Redis,
    tool_uid: UUIDLike,
) -> ToolNodePublic | None:
    """Return the workflow-node projection using a cache-aside strategy."""
    cache_key = _tool_args_key(tool_uid)
    cached = await redis.get(cache_key)
    if cached:
        try:
            return ToolNodePublic.model_validate_json(cached)
        except ValueError:
            await redis.delete(cache_key)

    tool = tool_crud.get(session, tool_uid)
    if tool is None:
        return None

    projection = ToolNodePublic.model_validate(tool)
    await redis.set(cache_key, projection.model_dump_json(), ex=TOOL_CACHE_TTL_SECONDS)
    return projection


async def get_tool_execution_config(
    session: Session,
    redis: Redis,
    tool_uid: UUIDLike,
) -> ToolTaskInfo | None:
    """Return the execution projection using a cache-aside strategy."""
    cache_key = _tool_task_info_key(tool_uid)
    cached = await redis.get(cache_key)
    if cached:
        try:
            return ToolTaskInfo.model_validate_json(cached)
        except ValueError:
            await redis.delete(cache_key)

    tool = tool_crud.get(session, tool_uid)
    if tool is None:
        return None

    projection = ToolTaskInfo.model_validate(tool)
    await redis.set(cache_key, projection.model_dump_json(), ex=TOOL_CACHE_TTL_SECONDS)
    return projection
