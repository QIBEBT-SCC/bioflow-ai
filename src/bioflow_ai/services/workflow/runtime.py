"""Transactional state machine for the DAG runtime."""

import platform
import socket
import uuid
from collections.abc import Iterable
from datetime import timedelta
from pathlib import Path

from loguru import logger
from sqlalchemy import distinct, func, select, update
from sqlalchemy.orm import Session

from bioflow_ai.core.session import engine
from bioflow_ai.models.workflow import NodeDependency, NodeRun, WorkflowOutbox, WorkflowRun
from bioflow_ai.schemas.workflow_config import ToolOutput
from bioflow_ai.schemas.workflow_run import ACTIVE_NODE_STATUSES, NodeRunStatus, NodeRunUpdate, RunStatus
from bioflow_ai.utils.system import utc_now


class WorkflowRuntimeInvariantError(RuntimeError):
    """Durable workflow state violates an execution invariant."""


def get_node_for_execution(node_run_uid: uuid.UUID, generation: int) -> NodeRun:
    """Load the frozen plan after a successful claim."""
    with Session(engine) as session:
        node = _get_current_node(session, node_run_uid, generation)
        if node is None or node.status != NodeRunStatus.RUNNING:
            message = f'Node {node_run_uid} is no longer running in generation {generation}'
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


def _get_current_node(session: Session, node_run_uid: uuid.UUID, generation: int) -> NodeRun | None:
    statement = (
        select(NodeRun)
        .join(WorkflowRun, WorkflowRun.uid == NodeRun.run_uid)
        .where(
            NodeRun.uid == node_run_uid,
            NodeRun.generation == generation,
            WorkflowRun.generation == generation,
        )
    )
    return session.scalar(statement)


def claim_node(node_run_uid: uuid.UUID, generation: int, worker_id: str | None) -> bool:
    """Atomically claim QUEUED -> RUNNING; duplicate or stale messages are no-ops."""
    now = utc_now()
    with Session(engine) as session, session.begin():
        node = _get_current_node(session, node_run_uid, generation)
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
        if run is None or run.generation != generation:
            return False
        if run.status == RunStatus.PENDING:
            run.status = RunStatus.RUNNING
            run.start_time = now
        run.settled = False
    return True


def heartbeat_node(node_run_uid: uuid.UUID, generation: int) -> bool:
    """Refresh the heartbeat only while this generation still owns the running node."""
    with Session(engine) as session, session.begin():
        updated = session.execute(
            update(NodeRun)
            .where(
                NodeRun.uid == node_run_uid,
                NodeRun.generation == generation,
                NodeRun.status == NodeRunStatus.RUNNING,
            )
            .values(heartbeat_at=utc_now())
            .returning(NodeRun.uid)
        ).scalar_one_or_none()
    return updated is not None


def update_node_from_executor(node_run_uid: uuid.UUID, generation: int, uid: str, data: NodeRunUpdate) -> bool:
    """Persist non-lifecycle metadata emitted by a shared execution implementation."""
    if uid != str(node_run_uid):
        message = 'Executor attempted to update a different node'
        raise WorkflowRuntimeInvariantError(message)
    values = data.model_dump(exclude_none=True)
    if not values:
        return True
    with Session(engine) as session, session.begin():
        updated = session.execute(
            update(NodeRun)
            .where(
                NodeRun.uid == node_run_uid,
                NodeRun.generation == generation,
                NodeRun.status == NodeRunStatus.RUNNING,
            )
            .values(**values)
            .returning(NodeRun.uid)
        ).scalar_one_or_none()
    return updated is not None


def resolve_dependency_output(
    current_node_uid: uuid.UUID,
    generation: int,
    source_uid: str,
    output_key: str,
) -> tuple[str, str | None]:
    """Resolve one declared successful predecessor without polling."""
    try:
        source_node_uid = uuid.UUID(source_uid)
    except ValueError as error:
        message = f'Invalid dependency node UID: {source_uid}'
        raise WorkflowRuntimeInvariantError(message) from error

    with Session(engine) as session:
        dependency = session.scalar(
            select(NodeDependency.uid).where(
                NodeDependency.source_node_uid == source_node_uid,
                NodeDependency.target_node_uid == current_node_uid,
                NodeDependency.generation == generation,
                NodeDependency.source_handle == output_key,
            )
        )
        if dependency is None:
            message = f'Node {source_uid} output {output_key!r} is not a declared dependency of {current_node_uid}'
            raise WorkflowRuntimeInvariantError(message)
        source = session.get(NodeRun, source_node_uid)
        if source is None or source.generation != generation or source.status != NodeRunStatus.SUCCEEDED:
            message = f'Dependency {source_uid} is not successfully completed'
            raise WorkflowRuntimeInvariantError(message)
        if source.tool_output is None:
            message = f'Dependency {source_uid} succeeded without output'
            raise WorkflowRuntimeInvariantError(message)
        if output_key == 'workdir' and source.base_dir:
            return str(Path(source.base_dir).absolute()), source.tool_output.sample_name
        values = source.tool_output.all_files()
        if output_key not in values:
            message = f'Dependency {source_uid} has no output named {output_key!r}'
            raise WorkflowRuntimeInvariantError(message)
        return str(values[output_key]), source.tool_output.sample_name


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


def _ensure_outbox(session: Session, node: NodeRun) -> None:
    if node.node_type == 'foreach':
        return
    outbox = session.scalar(select(WorkflowOutbox).where(WorkflowOutbox.node_run_uid == node.uid))
    if outbox is None:
        session.add(
            WorkflowOutbox(
                run_uid=node.run_uid,
                node_run_uid=node.uid,
                generation=node.generation,
            )
        )


def _settle_run_if_complete(session: Session, run_uid: uuid.UUID, generation: int) -> bool:
    run = session.get(WorkflowRun, run_uid, with_for_update=True)
    if run is None or run.generation != generation:
        return False
    unsettled_children = session.scalar(
        select(func.count(WorkflowRun.uid))
        .join(NodeRun, WorkflowRun.parent_node_uid == NodeRun.uid)
        .where(NodeRun.run_uid == run_uid, NodeRun.generation == generation, WorkflowRun.settled.is_(False))
    )
    if unsettled_children:
        run.settled = False
        return False
    active = session.scalar(
        select(func.count(NodeRun.uid)).where(
            NodeRun.run_uid == run_uid,
            NodeRun.generation == generation,
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
                NodeRun.generation == generation,
                NodeRun.status == NodeRunStatus.FAILED,
            )
        )
        run.status = RunStatus.FAILED if failed else RunStatus.SUCCEEDED
    return True


def mark_node_succeeded(node_run_uid: uuid.UUID, generation: int, output: ToolOutput) -> bool:
    """Commit output, then independently unlock every newly-ready direct successor."""
    now = utc_now()
    with Session(engine) as session, session.begin():
        node = session.get(NodeRun, node_run_uid, with_for_update=True)
        if node is None or node.generation != generation or node.status != NodeRunStatus.RUNNING:
            return False
        run = session.get(WorkflowRun, node.run_uid, with_for_update=True)
        if run is None or run.generation != generation:
            return False
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
                if successor.generation != generation or successor.status != NodeRunStatus.PENDING:
                    continue
                if _all_predecessors_succeeded(session, successor.uid):
                    successor.status = NodeRunStatus.READY
                    _ensure_outbox(session, successor)
        _settle_run_if_complete(session, node.run_uid, generation)
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


def mark_node_failed(node_run_uid: uuid.UUID, generation: int, error: BaseException | str) -> bool:
    """Fail one node, block only its descendants, and leave independent branches active."""
    now = utc_now()
    with Session(engine) as session, session.begin():
        node = session.get(NodeRun, node_run_uid, with_for_update=True)
        if (
            node is None
            or node.generation != generation
            or node.status
            not in (
                {NodeRunStatus.RUNNING, NodeRunStatus.READY}
                if node is not None and node.node_type == 'foreach'
                else {NodeRunStatus.RUNNING}
            )
        ):
            return False
        run = session.get(WorkflowRun, node.run_uid, with_for_update=True)
        if run is None or run.generation != generation:
            return False
        node.status = NodeRunStatus.FAILED
        node.error_message = str(error)[:4000]
        node.heartbeat_at = now
        node.end_time = now
        run.status = RunStatus.FAILED
        run.settled = False
        if node.node_type == 'foreach':
            session.execute(
                update(WorkflowRun)
                .where(
                    WorkflowRun.parent_node_uid == node.uid,
                    WorkflowRun.parent_generation == generation,
                    WorkflowRun.start_time.is_(None),
                )
                .values(status=RunStatus.FAILED, settled=True, end_time=now)
            )

        descendants = _descendant_ids(session, node.uid)
        if descendants:
            session.execute(
                update(NodeRun)
                .where(
                    NodeRun.uid.in_(descendants),
                    NodeRun.generation == generation,
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
        _settle_run_if_complete(session, node.run_uid, generation)
    return True


def recover_runtime(*, stale_after: timedelta, run_uids: Iterable[uuid.UUID] | None = None) -> dict[str, int]:
    """Repair READY outboxes, fail stale workers and settle converged runs."""
    repaired = 0
    stale = 0
    settled = 0
    now = utc_now()
    cutoff = now - stale_after

    with Session(engine) as session, session.begin():
        ready_statement = select(NodeRun).where(NodeRun.status == NodeRunStatus.READY)
        if run_uids is not None:
            ready_statement = ready_statement.where(NodeRun.run_uid.in_(list(run_uids)))
        for node in session.scalars(ready_statement.with_for_update(skip_locked=True)):
            before = session.scalar(select(WorkflowOutbox.uid).where(WorkflowOutbox.node_run_uid == node.uid))
            _ensure_outbox(session, node)
            repaired += int(before is None)

        stale_statement = select(NodeRun.uid, NodeRun.generation).where(
            NodeRun.status == NodeRunStatus.RUNNING,
            NodeRun.node_type != 'foreach',
            NodeRun.heartbeat_at < cutoff,
        )
        stale_nodes = list(session.execute(stale_statement))

    for node_uid, generation in stale_nodes:
        if mark_node_failed(node_uid, generation, f'Worker heartbeat timed out after {stale_after}'):
            stale += 1

    with Session(engine) as session, session.begin():
        run_statement = select(WorkflowRun.uid, WorkflowRun.generation).where(WorkflowRun.settled.is_(False))
        if run_uids is not None:
            run_statement = run_statement.where(WorkflowRun.uid.in_(list(run_uids)))
        for run_uid, generation in session.execute(run_statement):
            settled += int(_settle_run_if_complete(session, run_uid, generation))
    if repaired or stale or settled:
        logger.info(f'Workflow recovery: repaired={repaired}, stale={stale}, settled={settled}')
    return {'repaired': repaired, 'stale': stale, 'settled': settled}
