"""Node run queries and conditional writes; never commit here."""

import uuid
from collections.abc import Collection, Sequence
from datetime import datetime
from typing import Any

from sqlalchemy import delete, func, select, update
from sqlalchemy.orm import Session
from sqlalchemy.sql.elements import ColumnElement

from bioflow_ai.models.user import User
from bioflow_ai.models.workflow import NodeRun, WorkflowRun
from bioflow_ai.schemas.workflow_run import NodeRunStatus


def get(session: Session, uid: uuid.UUID, *, lock: bool = False) -> NodeRun | None:
    """Load a node with an optional row lock."""
    return session.get(NodeRun, uid, with_for_update=True if lock else None)


def insert(session: Session, node: NodeRun) -> None:
    """Stage a node without flushing."""
    session.add(node)


def delete_for_run(session: Session, run_uid: uuid.UUID) -> None:
    """Delete a run's nodes; database cascades remove their dependencies."""
    session.execute(delete(NodeRun).where(NodeRun.run_uid == run_uid))


def list_for_runs(session: Session, run_uids: Sequence[uuid.UUID]) -> list[NodeRun]:
    """Load nodes for multiple runs in definition order."""
    if not run_uids:
        return []
    return list(
        session.scalars(
            select(NodeRun).where(NodeRun.run_uid.in_(run_uids)).order_by(NodeRun.position_index, NodeRun.uid)
        )
    )


def get_all_by_uid(session: Session, uids: Collection[uuid.UUID], *, lock: bool = False) -> list[NodeRun]:
    """Load nodes in stable lock order."""
    statement = select(NodeRun).where(NodeRun.uid.in_(uids)).order_by(NodeRun.uid)
    if lock:
        statement = statement.with_for_update()
    return list(session.scalars(statement))


def _node_list_filters(
    *,
    owner_id: int | None,
    project_id: int | None,
    status: NodeRunStatus | None,
) -> list[ColumnElement[bool]]:
    filters: list[ColumnElement[bool]] = []
    if owner_id is not None:
        filters.append(WorkflowRun.owner_id == owner_id)
    if project_id is not None:
        filters.append(WorkflowRun.project_id == project_id)
    if status is not None:
        filters.append(NodeRun.status == status)
    return filters


def get_page(
    session: Session,
    *,
    owner_id: int | None,
    project_id: int | None = None,
    status: NodeRunStatus | None = None,
    offset: int = 0,
    limit: int = 20,
) -> tuple[list[tuple[NodeRun, WorkflowRun, str]], int]:
    """Load task-center rows and their total count."""
    filters = _node_list_filters(owner_id=owner_id, project_id=project_id, status=status)
    statement = (
        select(NodeRun, WorkflowRun, User.username)
        .join(WorkflowRun, WorkflowRun.uid == NodeRun.run_uid)
        .join(User, User.id == WorkflowRun.owner_id)
        .where(*filters)
        .order_by(NodeRun.create_time.desc(), NodeRun.uid.desc())
        .offset(offset)
        .limit(limit)
    )
    count_statement = (
        select(func.count()).select_from(NodeRun).join(WorkflowRun, WorkflowRun.uid == NodeRun.run_uid).where(*filters)
    )
    rows = session.execute(statement).all()
    total = session.scalar(count_statement) or 0
    return [tuple(row) for row in rows], total


def list_recent(
    session: Session, *, owner_id: int | None, since: datetime, limit: int
) -> list[tuple[NodeRun, WorkflowRun, str]]:
    """Load task-center rows created at or after the supplied cutoff."""
    filters = _node_list_filters(owner_id=owner_id, project_id=None, status=None)
    filters.append(NodeRun.create_time >= since)
    statement = (
        select(NodeRun, WorkflowRun, User.username)
        .join(WorkflowRun, WorkflowRun.uid == NodeRun.run_uid)
        .join(User, User.id == WorkflowRun.owner_id)
        .where(*filters)
        .order_by(NodeRun.create_time.desc(), NodeRun.uid.desc())
        .limit(limit)
    )
    return [tuple(row) for row in session.execute(statement)]


def count_by_status(session: Session, run_uid: uuid.UUID, statuses: Collection[NodeRunStatus]) -> int:
    """Count nodes matching any supplied status."""
    return (
        session.scalar(select(func.count(NodeRun.uid)).where(NodeRun.run_uid == run_uid, NodeRun.status.in_(statuses)))
        or 0
    )


def update_if_status(
    session: Session, uids: Collection[uuid.UUID], statuses: Collection[NodeRunStatus], values: dict[str, Any]
) -> int:
    """Atomically update nodes still in an expected state and return affected count."""
    return len(
        session.scalars(
            update(NodeRun)
            .where(NodeRun.uid.in_(uids), NodeRun.status.in_(statuses))
            .values(**values)
            .returning(NodeRun.uid)
        ).all()
    )


def update_by_status(session: Session, statuses: Collection[NodeRunStatus], values: dict[str, Any]) -> int:
    """Update every node in one of the supplied states and return affected count."""
    return len(
        session.scalars(
            update(NodeRun).where(NodeRun.status.in_(statuses)).values(**values).returning(NodeRun.uid)
        ).all()
    )


def list_uids_by_status(
    session: Session, statuses: Collection[NodeRunStatus], *, node_type: str | None = None
) -> list[uuid.UUID]:
    """Load node identities in creation order, optionally restricted by type."""
    statement = select(NodeRun.uid).where(NodeRun.status.in_(statuses))
    if node_type is not None:
        statement = statement.where(NodeRun.node_type == node_type)
    return list(session.scalars(statement.order_by(NodeRun.create_time)))


def lock_ready_uids(session: Session, limit: int) -> list[uuid.UUID]:
    """Select a dispatcher batch, skipping nodes locked by other dispatchers."""
    return list(
        session.scalars(
            select(NodeRun.uid)
            .where(NodeRun.status == NodeRunStatus.READY, NodeRun.node_type != 'foreach')
            .order_by(NodeRun.create_time, NodeRun.uid)
            .limit(limit)
            .with_for_update(skip_locked=True)
        )
    )


def list_stale_uids(session: Session, cutoff: datetime) -> list[uuid.UUID]:
    """Find running worker nodes with expired heartbeats."""
    return list(
        session.scalars(
            select(NodeRun.uid).where(
                NodeRun.status == NodeRunStatus.RUNNING, NodeRun.node_type != 'foreach', NodeRun.heartbeat_at < cutoff
            )
        )
    )
