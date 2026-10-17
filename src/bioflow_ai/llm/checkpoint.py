"""Checkpoint serialization shared by every agent checkpointer."""

import enum
import typing

from langgraph.checkpoint.redis.jsonplus_redis import JsonPlusRedisSerializer
from pydantic import BaseModel

from bioflow_ai.llm.middleware.todo import Todo
from bioflow_ai.llm.states import BuildEdge, BuildNode, WorkflowDraft
from bioflow_ai.llm.tools.test_run_report import TestRunReport
from bioflow_ai.llm.utils.uid_mapper import BidirectionalMapper
from bioflow_ai.schemas.tool_config import FileMount, ImageConfig, ParamDefine

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


def checkpoint_serializer() -> JsonPlusRedisSerializer:
    """Build the Redis checkpoint serializer that can revive agent state types."""
    seen: set[type] = set()
    for state_type in _STATE_TYPES:
        _collect_types(state_type, seen)
    allowed_msgpack_modules = sorted((cls.__module__, cls.__name__) for cls in seen)

    return JsonPlusRedisSerializer(
        allowed_json_modules=True,
        allowed_msgpack_modules=allowed_msgpack_modules,
    )
