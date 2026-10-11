"""Persistence and graph-edge queries for workflow node dependencies."""

import uuid
from collections.abc import Collection

from sqlalchemy import distinct, func, select
from sqlalchemy.orm import Session

from bioflow_ai.models.workflow import NodeDependency, NodeRun
from bioflow_ai.schemas.workflow_run import NodeRunStatus


def insert(session: Session, dependency: NodeDependency) -> None:
    """Stage a dependency after its nodes have been flushed."""
    session.add(dependency)


def list_successor_uids(session: Session, source_uids: Collection[uuid.UUID]) -> list[uuid.UUID]:
    """Load distinct direct successors of the supplied nodes."""
    return list(
        session.scalars(
            select(distinct(NodeDependency.target_node_uid)).where(NodeDependency.source_node_uid.in_(source_uids))
        )
    )


def count_incomplete_predecessors(session: Session, node_uid: uuid.UUID) -> int:
    """Count distinct predecessors that have not succeeded."""
    return (
        session.scalar(
            select(func.count(distinct(NodeDependency.source_node_uid)))
            .join(NodeRun, NodeRun.uid == NodeDependency.source_node_uid)
            .where(NodeDependency.target_node_uid == node_uid, NodeRun.status != NodeRunStatus.SUCCEEDED)
        )
        or 0
    )
