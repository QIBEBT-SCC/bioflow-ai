"""Persistent shared tool memories, with compact reads and replay-safe creation."""

import uuid

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from bioflow_ai.models.tools import ToolDefine, ToolMemory
from bioflow_ai.schemas.node_memory import ToolMemoryPublic, ToolMemorySummary
from bioflow_ai.services._transaction import committing
from bioflow_ai.utils.identifiers import UUIDLike, as_uuid


def list_tool_memories(session: Session, tool_uid: UUIDLike) -> list[ToolMemorySummary]:
    """List a tool's memory summaries without loading their deferred bodies."""
    stmt = (
        select(ToolMemory)
        .where(ToolMemory.tool_uid == as_uuid(tool_uid))
        .order_by(ToolMemory.created_at, ToolMemory.uid)
    )
    return [ToolMemorySummary.model_validate(memory) for memory in session.scalars(stmt)]


def read_tool_memory(session: Session, memory_id: uuid.UUID) -> ToolMemoryPublic | None:
    """Read a shared memory without exposing its creator's private run context."""
    memory = session.get(ToolMemory, memory_id)
    return ToolMemoryPublic.model_validate(memory) if memory is not None else None


def save_tool_memories(session: Session, memories: list[ToolMemory]) -> list[ToolMemoryPublic]:
    """Atomically save approved memories, tolerating retries with the same immutable IDs."""
    saved = []
    with committing(session):
        tool_uids = {memory.tool_uid for memory in memories}
        existing = set(session.scalars(select(ToolDefine.uid).where(ToolDefine.uid.in_(tool_uids))))
        if tool_uids - existing:
            msg = 'A target tool definition no longer exists. No memories were saved.'
            raise ValueError(msg)

        for memory in memories:
            stored = session.get(ToolMemory, memory.uid)
            if stored is None:
                try:
                    with session.begin_nested():
                        session.add(memory)
                        session.flush()
                    stored = memory
                except IntegrityError:
                    stored = session.get(ToolMemory, memory.uid)
                    if stored is None:
                        raise
            if (
                stored.tool_uid != memory.tool_uid
                or stored.description != memory.description
                or stored.content != memory.content
                or stored.created_by != memory.created_by
            ):
                msg = 'This memory write was already used for different content. No memories were saved.'
                raise ValueError(msg)
            saved.append(ToolMemoryPublic.model_validate(stored))
    return saved
