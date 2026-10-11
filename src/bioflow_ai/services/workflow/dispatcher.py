"""Publisher that moves READY workflow nodes onto the Celery queue."""

import uuid

from loguru import logger
from sqlalchemy.orm import Session

from bioflow_ai.crud import node_run as node_run_crud
from bioflow_ai.infrastructure.database import engine
from bioflow_ai.schemas.workflow_run import NodeRunStatus
from bioflow_ai.tasks.workflow import execute_node
from bioflow_ai.utils.system import utc_now


def _queue_ready_nodes(limit: int) -> list[uuid.UUID]:
    """Commit READY -> QUEUED before any message can reach a worker."""
    with Session(engine) as session, session.begin():
        node_uids = node_run_crud.lock_ready_uids(session, limit)
        if node_uids:
            node_run_crud.update_if_status(
                session, node_uids, [NodeRunStatus.READY], {'status': NodeRunStatus.QUEUED, 'queued_at': utc_now()}
            )
    return node_uids


def _return_to_ready(node_uids: list[uuid.UUID]) -> int:
    with Session(engine) as session, session.begin():
        return node_run_crud.update_if_status(
            session, node_uids, [NodeRunStatus.QUEUED], {'status': NodeRunStatus.READY, 'queued_at': None}
        )


def requeue_unpublished() -> int:
    """Return QUEUED nodes to READY when the dispatcher starts.

    A dispatcher that stopped between committing QUEUED and publishing leaves
    nodes without a message. Republishing a node whose message did reach the
    broker is harmless: ``claim_node`` lets only one message start it.
    """
    with Session(engine) as session, session.begin():
        return node_run_crud.update_by_status(
            session, [NodeRunStatus.QUEUED], {'status': NodeRunStatus.READY, 'queued_at': None}
        )


def dispatch_pending(*, limit: int = 100) -> int:
    """Publish one batch of READY nodes and return how many were published.

    Queue, serializer and publish retries come from the task and Celery configuration.
    """
    published = 0
    for node_uid in _queue_ready_nodes(limit):
        try:
            execute_node.apply_async(kwargs={'node_run_id': str(node_uid)}, task_id=str(node_uid))
        except Exception:
            _return_to_ready([node_uid])
            logger.exception(f'Failed to publish workflow node {node_uid}')
        else:
            published += 1
    return published
