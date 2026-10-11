"""Transactional state machine for the DAG runtime."""

import platform
import socket
import uuid
from datetime import timedelta
from pathlib import Path

from loguru import logger
from sqlalchemy import distinct, func, select, update
from sqlalchemy.orm import Session

from bioflow_ai.infrastructure.database import engine
from bioflow_ai.models.workflow import NodeDependency, NodeRun, WorkflowRun
from bioflow_ai.schemas.workflow_config import ToolOutput
from bioflow_ai.schemas.workflow_run import ACTIVE_NODE_STATUSES, NodeRunStatus, RunStatus
from bioflow_ai.utils.system import utc_now


class WorkflowRuntimeInvariantError(RuntimeError):
    """Durable workflow state violates an execution invariant."""


def get_node_for_execution(node_run_uid: uuid.UUID) -> NodeRun:
    """Load the frozen plan after a successful claim."""
    with Session(engine) as session:
        node = session.get(NodeRun, node_run_uid)
        if node is None or node.status != NodeRunStatus.RUNNING:
            message = f'Node {node_run_uid} is no longer running'
            raise WorkflowRuntimeInvariantError(message)
        session.expunge(node)
        return node


def persist_failure_log(node_run_uid: uuid.UUID, error: BaseException) -> None:
    """Write an execution error beside node outputs when a directory exists."""
    with Session(engine) as session:
        node = session.get(NodeRun, node_run_uid)
        if node is None or not node.base_dir:
            return
        output_dir = Path(node.base_dir)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / '11_run.log').open('a', encoding='utf-8') as log_file:
            log_file.write(f'\n[workflow failure]\n{type(error).__name__}: {error}\n')
    except OSError:
        logger.exception(f'Failed to persist node error log for {node_run_uid}')


def claim_node(node_run_uid: uuid.UUID, worker_id: str | None) -> bool:
    """Atomically claim QUEUED -> RUNNING; duplicate or stale messages are no-ops."""
    now = utc_now()
    with Session(engine) as session, session.begin():
        node = session.get(NodeRun, node_run_uid)
        if node is None or node.status != NodeRunStatus.QUEUED:
            return False
        claimed = session.execute(
            update(NodeRun)
            .where(NodeRun.uid == node_run_uid, NodeRun.status == NodeRunStatus.QUEUED)
            .values(
                status=NodeRunStatus.RUNNING,
                start_time=now,
                heartbeat_at=now,
                worker_id=worker_id,
                system=platform.platform(),
                hostname=socket.gethostname(),
            )
            .returning(NodeRun.uid)
        ).scalar_one_or_none()
        if claimed is None:
            return False
        run = session.get(WorkflowRun, node.run_uid, with_for_update=True)
        if run is None:
            raise ValueError

        if run.status == RunStatus.PENDING:
            run.status = RunStatus.RUNNING
            run.start_time = now
        run.settled = False
    return True


def heartbeat_node(node_run_uid: uuid.UUID) -> bool:
    """Refresh the heartbeat only while the node is still running."""
    with Session(engine) as session, session.begin():
        updated = session.execute(
            update(NodeRun)
            .where(NodeRun.uid == node_run_uid, NodeRun.status == NodeRunStatus.RUNNING)
            .values(heartbeat_at=utc_now())
            .returning(NodeRun.uid)
        ).scalar_one_or_none()
    return updated is not None


def _direct_successors(session: Session, node_uid: uuid.UUID) -> list[uuid.UUID]:
    return list(
        session.scalars(
            select(distinct(NodeDependency.target_node_uid)).where(NodeDependency.source_node_uid == node_uid)
        )
    )


def _all_predecessors_succeeded(session: Session, node_uid: uuid.UUID) -> bool:
    incomplete = session.scalar(
        select(func.count(distinct(NodeDependency.source_node_uid)))
        .join(NodeRun, NodeRun.uid == NodeDependency.source_node_uid)
        .where(
            NodeDependency.target_node_uid == node_uid,
            NodeRun.status != NodeRunStatus.SUCCEEDED,
        )
    )
    return not incomplete


def _settle_run_if_complete(session: Session, run_uid: uuid.UUID) -> bool:
    run = session.get(WorkflowRun, run_uid, with_for_update=True)
    if run is None:
        raise ValueError

    unsettled_children = session.scalar(
        select(func.count(WorkflowRun.uid))
        .join(NodeRun, WorkflowRun.parent_node_uid == NodeRun.uid)
        .where(NodeRun.run_uid == run_uid, WorkflowRun.settled.is_(False))
    )
    if unsettled_children:
        run.settled = False
        return False
    active = session.scalar(
        select(func.count(NodeRun.uid)).where(
            NodeRun.run_uid == run_uid,
            NodeRun.status.in_(ACTIVE_NODE_STATUSES),
        )
    )
    if active:
        run.settled = False
        return False
    run.settled = True
    run.end_time = utc_now()
    if run.status != RunStatus.FAILED:
        failed = session.scalar(
            select(func.count(NodeRun.uid)).where(
                NodeRun.run_uid == run_uid,
                NodeRun.status == NodeRunStatus.FAILED,
            )
        )
        run.status = RunStatus.FAILED if failed else RunStatus.SUCCEEDED
    return True


def mark_node_succeeded(node_run_uid: uuid.UUID, output: ToolOutput) -> bool:
    """Commit output, then independently unlock every newly-ready direct successor."""
    now = utc_now()
    with Session(engine) as session, session.begin():
        node = session.get(NodeRun, node_run_uid, with_for_update=True)
        if node is None or node.status != NodeRunStatus.RUNNING:
            return False
        # Lock the run before successors, matching mark_node_failed's lock order.
        session.get(WorkflowRun, node.run_uid, with_for_update=True)
        node.tool_output = output
        node.status = NodeRunStatus.SUCCEEDED
        node.heartbeat_at = now
        node.end_time = now

        successor_ids = sorted(_direct_successors(session, node.uid), key=str)
        if successor_ids:
            successors = list(
                session.scalars(
                    select(NodeRun).where(NodeRun.uid.in_(successor_ids)).order_by(NodeRun.uid).with_for_update()
                )
            )
            for successor in successors:
                if successor.status != NodeRunStatus.PENDING:
                    continue
                if _all_predecessors_succeeded(session, successor.uid):
                    successor.status = NodeRunStatus.READY
        _settle_run_if_complete(session, node.run_uid)
    return True


def _descendant_ids(session: Session, root_uid: uuid.UUID) -> set[uuid.UUID]:
    descendants: set[uuid.UUID] = set()
    frontier = {root_uid}
    while frontier:
        next_nodes = set(
            session.scalars(
                select(distinct(NodeDependency.target_node_uid)).where(NodeDependency.source_node_uid.in_(frontier))
            )
        )
        frontier = next_nodes - descendants
        descendants.update(frontier)
    descendants.discard(root_uid)
    return descendants


def mark_node_failed(node_run_uid: uuid.UUID, error: BaseException | str) -> bool:
    """Fail one node, block only its descendants, and leave independent branches active."""
    now = utc_now()
    with Session(engine) as session, session.begin():
        node = session.get(NodeRun, node_run_uid, with_for_update=True)
        if node is None or node.status not in (
            {NodeRunStatus.RUNNING, NodeRunStatus.READY}
            if node is not None and node.node_type == 'foreach'
            else {NodeRunStatus.RUNNING}
        ):
            return False
        run = session.get(WorkflowRun, node.run_uid, with_for_update=True)
        if run is None:
            raise ValueError

        node.status = NodeRunStatus.FAILED
        node.error_message = str(error)[:4000]
        node.heartbeat_at = now
        node.end_time = now
        run.status = RunStatus.FAILED
        run.settled = False
        if node.node_type == 'foreach':
            session.execute(
                update(WorkflowRun)
                .where(WorkflowRun.parent_node_uid == node.uid, WorkflowRun.start_time.is_(None))
                .values(status=RunStatus.FAILED, settled=True, end_time=now)
            )

        descendants = _descendant_ids(session, node.uid)
        if descendants:
            session.execute(
                update(NodeRun)
                .where(
                    NodeRun.uid.in_(descendants),
                    NodeRun.status.in_(
                        {
                            NodeRunStatus.PENDING,
                            NodeRunStatus.READY,
                            NodeRunStatus.QUEUED,
                            NodeRunStatus.RUNNING,
                        }
                    ),
                )
                .values(status=NodeRunStatus.BLOCKED, end_time=now)
            )
        _settle_run_if_complete(session, node.run_uid)
    return True


def recover_runtime(*, stale_after: timedelta) -> dict[str, int]:
    """Fail stale workers and settle converged runs."""
    stale = 0
    settled = 0
    cutoff = utc_now() - stale_after

    with Session(engine) as session:
        stale_statement = select(NodeRun.uid).where(
            NodeRun.status == NodeRunStatus.RUNNING,
            NodeRun.node_type != 'foreach',
            NodeRun.heartbeat_at < cutoff,
        )
        stale_nodes = list(session.scalars(stale_statement))

    for node_uid in stale_nodes:
        if mark_node_failed(node_uid, f'Worker heartbeat timed out after {stale_after}'):
            stale += 1

    with Session(engine) as session, session.begin():
        run_statement = select(WorkflowRun.uid).where(WorkflowRun.settled.is_(False))
        for run_uid in session.scalars(run_statement):
            settled += int(_settle_run_if_complete(session, run_uid))
    if stale or settled:
        logger.info(f'Workflow recovery: stale={stale}, settled={settled}')
    return {'stale': stale, 'settled': settled}
