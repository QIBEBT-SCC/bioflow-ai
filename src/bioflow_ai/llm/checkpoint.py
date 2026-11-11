"""Postgres checkpointer and serialization shared by every agent."""

import enum
import typing
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from pydantic import BaseModel

from bioflow_ai.core.config import get_settings
from bioflow_ai.llm.middleware.todo import Todo
from bioflow_ai.llm.states import BuildEdge, BuildNode, WorkflowDraft
from bioflow_ai.llm.tools.test_run_report import TestRunReport
from bioflow_ai.schemas.tool_config import FileMount, ImageConfig, ParamDefine
from bioflow_ai.utils.uid_mapper import BidirectionalMapper

_STATE_TYPES: tuple[type, ...] = (
    BidirectionalMapper,
    BuildEdge,
    BuildNode,
    FileMount,
    ImageConfig,
    ParamDefine,
    TestRunReport,
    Todo,
    WorkflowDraft,
)


def _collect_types(annotation: object, seen: set[type]) -> None:
    for arg in typing.get_args(annotation):
        _collect_types(arg, seen)
    if not isinstance(annotation, type) or annotation in seen:
        return
    if issubclass(annotation, BaseModel):
        seen.add(annotation)
        for field in annotation.model_fields.values():
            _collect_types(field.annotation, seen)
    elif issubclass(annotation, enum.Enum):
        seen.add(annotation)


def checkpoint_serializer() -> JsonPlusSerializer:
    """Build the checkpoint serializer that can revive agent state types."""
    seen: set[type] = set()
    for state_type in _STATE_TYPES:
        _collect_types(state_type, seen)
    allowed_msgpack_modules = sorted((cls.__module__, cls.__name__) for cls in seen)

    return JsonPlusSerializer(
        allowed_json_modules=True,
        allowed_msgpack_modules=allowed_msgpack_modules,
    )


@asynccontextmanager
async def open_checkpointer(*, setup: bool = False) -> AsyncIterator[AsyncPostgresSaver]:
    """Open a Postgres checkpointer on the checkpoint database.

    Args:
        setup: Create or migrate the checkpoint tables before yielding.

    Yields:
        A checkpointer configured with the shared agent state serializer.
    """
    async with AsyncPostgresSaver.from_conn_string(
        get_settings().persistent.database.CHECKPOINT_URL.unicode_string(),
        serde=checkpoint_serializer(),
    ) as saver:
        if setup:
            await saver.setup()
        yield saver


def delete_checkpoint_thread(thread_id: str) -> None:
    """Delete every checkpoint of a finished thread, which no later run or resume reads again."""
    with PostgresSaver.from_conn_string(get_settings().persistent.database.CHECKPOINT_URL.unicode_string()) as saver:
        saver.delete_thread(thread_id)
