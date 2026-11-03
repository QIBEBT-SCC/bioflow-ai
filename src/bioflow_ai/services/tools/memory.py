"""Persistent shared tool memories, with compact reads and replay-safe creation."""

import uuid

from sqlalchemy import ColumnElement, Select, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import InstrumentedAttribute, Session

from bioflow_ai.models.tools import ToolDefine, ToolMemory
from bioflow_ai.models.user import User
from bioflow_ai.schemas.node_memory import (
    ToolMemoryCreate,
    ToolMemoryListItem,
    ToolMemoryPublic,
    ToolMemorySummary,
    ToolMemoryUpdate,
)
from bioflow_ai.services._transaction import committing
from bioflow_ai.utils.identifiers import UUIDLike, as_uuid


def _select_memories(*extra: InstrumentedAttribute | ColumnElement) -> Select:
    """Select memory fields with their tool name and creator; a creator's user row may no longer exist."""
    return (
        select(
            ToolMemory.uid,
            ToolMemory.description,
            ToolMemory.tool_uid,
            ToolDefine.name.label('tool_name'),
            User.username.label('creator_username'),
            ToolMemory.created_at,
            *extra,
        )
        .join_from(ToolMemory, ToolDefine, ToolMemory.tool_uid == ToolDefine.uid)
        .outerjoin(User, ToolMemory.created_by == User.id)
    )


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
    row = session.execute(_select_memories(ToolMemory.content).where(ToolMemory.uid == memory_id)).one_or_none()
    return ToolMemoryPublic.model_validate(row) if row is not None else None


def get_tool_memory_page(
    session: Session, *, tool_uid: uuid.UUID | None, offset: int, limit: int
) -> tuple[list[ToolMemoryListItem], int]:
    """List shared memories without loading their bodies."""
    stmt = _select_memories()
    count = select(func.count()).select_from(ToolMemory)
    if tool_uid is not None:
        stmt = stmt.where(ToolMemory.tool_uid == tool_uid)
        count = count.where(ToolMemory.tool_uid == tool_uid)
    rows = session.execute(stmt.order_by(ToolMemory.created_at.desc(), ToolMemory.uid).offset(offset).limit(limit))
    return [ToolMemoryListItem.model_validate(row) for row in rows], session.scalar(count) or 0


def create_tool_memory(session: Session, data: ToolMemoryCreate, *, user_id: int) -> ToolMemoryPublic | None:
    """Create a memory and return None if the target tool no longer exists."""
    with committing(session):
        tool_uid = session.scalar(
            select(ToolDefine.uid).where(ToolDefine.uid == data.tool_uid).with_for_update(read=True)
        )
        if tool_uid is None:
            return None
        memory = ToolMemory(tool_uid=tool_uid, description=data.description, content=data.content, created_by=user_id)
        session.add(memory)
        session.flush()
        return read_tool_memory(session, memory.uid)


def update_tool_memory(session: Session, memory_id: uuid.UUID, data: ToolMemoryUpdate) -> ToolMemoryPublic | None:
    """Update the provided fields while retaining the memory's tool and creator."""
    with committing(session):
        memory = session.get(ToolMemory, memory_id, with_for_update=True)
        if memory is None:
            return None
        if data.description is not None:
            memory.description = data.description
        if data.content is not None:
            memory.content = data.content
        session.flush()
        return read_tool_memory(session, memory_id)


def delete_tool_memory(session: Session, memory_id: uuid.UUID) -> bool:
    """Delete a memory and report whether it existed."""
    with committing(session):
        memory = session.get(ToolMemory, memory_id, with_for_update=True)
        if memory is None:
            return False
        session.delete(memory)
        return True


def save_tool_memories(session: Session, memories: list[ToolMemory]) -> list[ToolMemorySummary]:
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
            saved.append(ToolMemorySummary.model_validate(stored))
    return saved
