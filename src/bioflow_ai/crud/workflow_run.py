"""Workflow run persistence; callers own transactions and business decisions."""

import uuid
from collections.abc import Sequence
from typing import Any

from sqlalchemy import delete, func, select, update
from sqlalchemy.orm import Session

from bioflow_ai.models.workflow import NodeRun, WorkflowRun
from bioflow_ai.schemas.workflow_run import RunStatus


def get(session: Session, uid: uuid.UUID, *, lock: bool = False) -> WorkflowRun | None:
    """Load a run, optionally taking a row lock in the caller's transaction."""
    return session.get(WorkflowRun, uid, with_for_update=True if lock else None)


def insert(session: Session, run: WorkflowRun) -> None:
    """Stage a run without flushing or committing the caller's transaction."""
    session.add(run)


def find_project_run(
    session: Session,
    project_id: int,
    workflow_uid: uuid.UUID,
    sample_uid: uuid.UUID | None,
) -> WorkflowRun | None:
    """Find and lock the reusable identity for a project/workflow/sample."""
    statement = select(WorkflowRun).where(
        WorkflowRun.project_id == project_id,
        WorkflowRun.workflow_uid == workflow_uid,
    )
    statement = (
        statement.where(WorkflowRun.sample_uid.is_(None))
        if sample_uid is None
        else statement.where(WorkflowRun.sample_uid == sample_uid)
    )
    return session.scalar(statement.with_for_update())


def get_page(
    session: Session,
    *,
    owner_id: int | None,
    project_id: int | None = None,
    status: RunStatus | None = None,
    offset: int = 0,
    limit: int = 20,
) -> tuple[list[WorkflowRun], int]:
    """Load a page of top-level runs and its total count."""
    filters = [WorkflowRun.parent_node_uid.is_(None)]
    if owner_id is not None:
        filters.append(WorkflowRun.owner_id == owner_id)
    if project_id is not None:
        filters.append(WorkflowRun.project_id == project_id)
    if status is not None:
        filters.append(WorkflowRun.status == status)
    statement = select(WorkflowRun).where(*filters).order_by(WorkflowRun.create_time.desc()).offset(offset).limit(limit)
    count_statement = select(func.count()).select_from(WorkflowRun).where(*filters)
    runs = list(session.scalars(statement))
    total = session.scalar(count_statement) or 0
    return runs, total


def count_by_status(session: Session, *, owner_id: int | None, project_id: int | None = None) -> dict[RunStatus, int]:
    """Count top-level runs by persisted status."""
    filters = [WorkflowRun.parent_node_uid.is_(None)]
    if owner_id is not None:
        filters.append(WorkflowRun.owner_id == owner_id)
    if project_id is not None:
        filters.append(WorkflowRun.project_id == project_id)
    statement = select(WorkflowRun.status, func.count()).where(*filters).group_by(WorkflowRun.status)
    return dict(session.execute(statement).tuples().all())


def list_for_workflow(
    session: Session, *, project_id: int, workflow_uid: uuid.UUID, owner_id: int | None
) -> list[WorkflowRun]:
    """Load current runs belonging to a project workflow."""
    filters = [WorkflowRun.project_id == project_id, WorkflowRun.workflow_uid == workflow_uid]
    if owner_id is not None:
        filters.append(WorkflowRun.owner_id == owner_id)
    return list(session.scalars(select(WorkflowRun).where(*filters).order_by(WorkflowRun.sample_uid)))


def list_children(session: Session, parent_node_uid: uuid.UUID, *, lock: bool = False) -> list[WorkflowRun]:
    """Load child runs in item order, optionally locking them."""
    statement = (
        select(WorkflowRun).where(WorkflowRun.parent_node_uid == parent_node_uid).order_by(WorkflowRun.item_index)
    )
    if lock:
        statement = statement.with_for_update()
    return list(session.scalars(statement))


def count_children_by_state(
    session: Session, parent_node_uids: Sequence[uuid.UUID]
) -> list[tuple[uuid.UUID, RunStatus, bool, bool, int]]:
    """Group children by parent, status, start presence and settlement."""
    if not parent_node_uids:
        return []
    started = WorkflowRun.start_time.is_not(None)
    rows = session.execute(
        select(WorkflowRun.parent_node_uid, WorkflowRun.status, started, WorkflowRun.settled, func.count())
        .where(WorkflowRun.parent_node_uid.in_(parent_node_uids))
        .group_by(WorkflowRun.parent_node_uid, WorkflowRun.status, started, WorkflowRun.settled)
    )
    return [tuple(row) for row in rows]


def count_unsettled_children(session: Session, run_uid: uuid.UUID) -> int:
    """Count unsettled children across a run's control nodes."""
    return (
        session.scalar(
            select(func.count(WorkflowRun.uid))
            .join(NodeRun, WorkflowRun.parent_node_uid == NodeRun.uid)
            .where(NodeRun.run_uid == run_uid, WorkflowRun.settled.is_(False))
        )
        or 0
    )


def list_unsettled_uids(session: Session) -> list[uuid.UUID]:
    """Load run identities awaiting settlement."""
    return list(session.scalars(select(WorkflowRun.uid).where(WorkflowRun.settled.is_(False))))


def update_unstarted_children(session: Session, parent_node_uid: uuid.UUID, values: dict[str, Any]) -> None:
    """Update only children that have not started."""
    session.execute(
        update(WorkflowRun)
        .where(WorkflowRun.parent_node_uid == parent_node_uid, WorkflowRun.start_time.is_(None))
        .values(**values)
    )


def get_parent_details(session: Session, node_uids: Sequence[uuid.UUID]) -> dict[uuid.UUID, tuple[uuid.UUID, str, str]]:
    """Batch-load parent node and run labels."""
    if not node_uids:
        return {}
    rows = session.execute(
        select(NodeRun.uid, NodeRun.run_uid, WorkflowRun.name, NodeRun.name)
        .join(WorkflowRun, WorkflowRun.uid == NodeRun.run_uid)
        .where(NodeRun.uid.in_(node_uids))
    )
    return {uid: (run_uid, run_name, node_name) for uid, run_uid, run_name, node_name in rows}


def delete_for_project(session: Session, project_id: int) -> None:
    """Delete a project's runs inside the caller's transaction."""
    session.execute(delete(WorkflowRun).where(WorkflowRun.project_id == project_id))


def delete_for_workflow(session: Session, project_id: int, workflow_uid: uuid.UUID) -> None:
    """Delete project workflow runs; database cascades remove execution records."""
    session.execute(
        delete(WorkflowRun).where(WorkflowRun.project_id == project_id, WorkflowRun.workflow_uid == workflow_uid)
    )
