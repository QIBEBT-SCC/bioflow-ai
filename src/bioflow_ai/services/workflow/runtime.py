"""Transactional state machine for the DAG runtime."""

import platform
import socket
import uuid
from datetime import timedelta
from pathlib import Path
from typing import Any

from loguru import logger
from sqlalchemy.orm import Session

from bioflow_ai.core.config import get_settings
from bioflow_ai.crud import node_dependency as dependency_crud
from bioflow_ai.crud import node_run as node_run_crud
from bioflow_ai.crud import workflow_run as run_crud
from bioflow_ai.infrastructure.database import engine
from bioflow_ai.models.workflow import NodeRun
from bioflow_ai.schemas.passthrough import inherited_log_path
from bioflow_ai.schemas.workflow_config import ToolOutput
from bioflow_ai.schemas.workflow_run import ACTIVE_NODE_STATUSES, NodeRunStatus, RunStatus
from bioflow_ai.utils.system import utc_now


class WorkflowRuntimeInvariantError(RuntimeError):
    """Durable workflow state violates an execution invariant."""


def node_log_path(node: NodeRun) -> Path | None:
    """Locate a node's durable log, including a pass-through Tool writing into an inherited workspace."""
    if node.tool_output is not None and node.tool_output.log:
        return Path(node.tool_output.log)
    if not node.base_dir:
        return None
    # The executor replaces base_dir with the inherited workspace; the frozen plan keeps the node's own directory.
    node_dir = (node.prepared_data or {}).get('kwargs', {}).get('output_dir')
    if node.node_type == 'tool' and node_dir and Path(node_dir).resolve() != Path(node.base_dir).resolve():
        return inherited_log_path(node.base_dir, node_dir)
    return Path(node.base_dir) / '11_run.log'


def persist_failure_log(node_run_uid: uuid.UUID, error: BaseException) -> None:
    """Write an execution error into the node log when a directory exists."""
    with Session(engine) as session:
        node = node_run_crud.get(session, node_run_uid)
        log_path = node_log_path(node) if node is not None else None
    if log_path is None:
        return
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open('a', encoding='utf-8') as log_file:
            log_file.write(f'\n[workflow failure]\n{type(error).__name__}: {error}\n')
    except OSError:
        logger.exception(f'Failed to persist node error log for {node_run_uid}')


def claim_node(node_run_uid: uuid.UUID, worker_id: str | None) -> dict[str, Any] | None:
    """Atomically claim QUEUED -> RUNNING and return the frozen plan.

    Duplicate or stale messages return ``None``.
    """
    now = utc_now()
    with Session(engine) as session, session.begin():
        claimed = node_run_crud.update_if_status(
            session,
            [node_run_uid],
            [NodeRunStatus.QUEUED],
            {
                'status': NodeRunStatus.RUNNING,
                'start_time': now,
                'heartbeat_at': now,
                'worker_id': worker_id,
                'system': platform.platform(),
                'hostname': socket.gethostname(),
            },
        )
        if not claimed:
            return None
        node = node_run_crud.get(session, node_run_uid)
        if node is None:
            raise ValueError
        run = run_crud.get(session, node.run_uid, lock=True)
        if run is None:
            raise ValueError

        if run.status == RunStatus.PENDING:
            run.status = RunStatus.RUNNING
            run.start_time = now
        return node.prepared_data


def heartbeat_node(node_run_uid: uuid.UUID) -> bool:
    """Refresh the heartbeat only while the node is still running."""
    with Session(engine) as session, session.begin():
        return bool(
            node_run_crud.update_if_status(
                session, [node_run_uid], [NodeRunStatus.RUNNING], {'heartbeat_at': utc_now()}
            )
        )


def _settle_run_if_complete(session: Session, run_uid: uuid.UUID) -> bool:
    run = run_crud.get(session, run_uid, lock=True)
    if run is None:
        raise ValueError

    if run_crud.count_unsettled_children(session, run_uid):
        return False
    if node_run_crud.count_by_status(session, run_uid, ACTIVE_NODE_STATUSES):
        return False
    run.settled = True
    run.end_time = utc_now()
    # mark_node_failed already marks the run FAILED, so any other run succeeded.
    if run.status != RunStatus.FAILED:
        run.status = RunStatus.SUCCEEDED
    return True


def mark_node_succeeded(node_run_uid: uuid.UUID, output: ToolOutput) -> bool:
    """Commit output, then independently unlock every newly-ready direct successor."""
    now = utc_now()
    with Session(engine) as session, session.begin():
        node = node_run_crud.get(session, node_run_uid, lock=True)
        if node is None or node.status != NodeRunStatus.RUNNING:
            return False
        # Lock the run before successors, matching mark_node_failed's lock order.
        run_crud.get(session, node.run_uid, lock=True)
        node.tool_output = output
        node.status = NodeRunStatus.SUCCEEDED
        node.heartbeat_at = now
        node.end_time = now

        successor_ids = dependency_crud.list_successor_uids(session, [node.uid])
        if successor_ids:
            for successor in node_run_crud.get_all_by_uid(session, successor_ids, lock=True):
                if successor.status != NodeRunStatus.PENDING:
                    continue
                if not dependency_crud.count_incomplete_predecessors(session, successor.uid):
                    successor.status = NodeRunStatus.READY
        _settle_run_if_complete(session, node.run_uid)
    return True


def _descendant_ids(session: Session, root_uid: uuid.UUID) -> set[uuid.UUID]:
    descendants: set[uuid.UUID] = set()
    frontier = {root_uid}
    while frontier:
        next_nodes = set(dependency_crud.list_successor_uids(session, frontier))
        frontier = next_nodes - descendants
        descendants.update(frontier)
    descendants.discard(root_uid)
    return descendants


def mark_node_failed(node_run_uid: uuid.UUID, error: BaseException | str) -> bool:
    """Fail one node, block only its descendants, and leave independent branches active."""
    now = utc_now()
    with Session(engine) as session, session.begin():
        node = node_run_crud.get(session, node_run_uid, lock=True)
        if node is None:
            return False
        # ForEach nodes are coordinated by the dispatcher and can fail before they start running.
        failable = (
            {NodeRunStatus.RUNNING, NodeRunStatus.READY} if node.node_type == 'foreach' else {NodeRunStatus.RUNNING}
        )
        if node.status not in failable:
            return False
        run = run_crud.get(session, node.run_uid, lock=True)
        if run is None:
            raise ValueError

        node.status = NodeRunStatus.FAILED
        node.error_message = str(error)[:4000]
        node.heartbeat_at = now
        node.end_time = now
        run.status = RunStatus.FAILED
        if node.node_type == 'foreach':
            run_crud.update_unstarted_children(
                session, node.uid, {'status': RunStatus.FAILED, 'settled': True, 'end_time': now}
            )

        # A descendant needs this node to succeed before it can leave PENDING.
        descendants = _descendant_ids(session, node.uid)
        if descendants:
            node_run_crud.update_if_status(
                session, descendants, [NodeRunStatus.PENDING], {'status': NodeRunStatus.BLOCKED, 'end_time': now}
            )
        _settle_run_if_complete(session, node.run_uid)
    return True


def recover_runtime() -> None:
    """Fail stale workers and settle converged runs.

    The settle sweep also catches parents of a failed ForEach whose started children finish later.
    """
    stale = 0
    settled = 0
    stale_after = timedelta(seconds=get_settings().workflow.HEARTBEAT_TIMEOUT)
    cutoff = utc_now() - stale_after

    with Session(engine) as session:
        stale_nodes = node_run_crud.list_stale_uids(session, cutoff)

    for node_uid in stale_nodes:
        if mark_node_failed(node_uid, f'Worker heartbeat timed out after {stale_after}'):
            stale += 1

    with Session(engine) as session, session.begin():
        for run_uid in run_crud.list_unsettled_uids(session):
            settled += int(_settle_run_if_complete(session, run_uid))
    if stale or settled:
        logger.info(f'Workflow recovery: stale={stale}, settled={settled}')
