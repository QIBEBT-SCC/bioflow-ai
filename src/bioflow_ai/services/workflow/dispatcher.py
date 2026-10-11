"""Publisher that moves READY workflow nodes onto the Celery queue."""

import uuid

from loguru import logger
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from bioflow_ai.infrastructure.database import engine
from bioflow_ai.models.workflow import NodeRun
from bioflow_ai.schemas.workflow_run import NodeRunStatus
from bioflow_ai.utils.system import utc_now

WORKFLOW_QUEUE = 'workflow-v2'


def _queue_ready_nodes(limit: int) -> list[uuid.UUID]:
    """Commit READY -> QUEUED before any message can reach a worker."""
    with Session(engine) as session, session.begin():
        node_uids = list(
            session.scalars(
                select(NodeRun.uid)
                .where(NodeRun.status == NodeRunStatus.READY, NodeRun.node_type != 'foreach')
                .order_by(NodeRun.create_time, NodeRun.uid)
                .limit(limit)
                .with_for_update(skip_locked=True)
            )
        )
        if node_uids:
            session.execute(
                update(NodeRun)
                .where(NodeRun.uid.in_(node_uids))
                .values(status=NodeRunStatus.QUEUED, queued_at=utc_now())
            )
    return node_uids


def _return_to_ready(node_uids: list[uuid.UUID]) -> int:
    with Session(engine) as session, session.begin():
        return session.execute(
            update(NodeRun)
            .where(NodeRun.uid.in_(node_uids), NodeRun.status == NodeRunStatus.QUEUED)
            .values(status=NodeRunStatus.READY, queued_at=None)
        ).rowcount


def requeue_unpublished() -> int:
    """Return QUEUED nodes to READY when the dispatcher starts.

    A dispatcher that stopped between committing QUEUED and publishing leaves
    nodes without a message. Republishing a node whose message did reach the
    broker is harmless: ``claim_node`` lets only one message start it.
    """
    with Session(engine) as session, session.begin():
        node_uids = list(session.scalars(select(NodeRun.uid).where(NodeRun.status == NodeRunStatus.QUEUED)))
    return _return_to_ready(node_uids) if node_uids else 0


def dispatch_pending(*, limit: int = 100) -> int:
    """Publish one batch of READY nodes and return how many were published."""
    from bioflow_ai.tasks.workflow import execute_node

    published = 0
    for node_uid in _queue_ready_nodes(limit):
        try:
            execute_node.apply_async(
                kwargs={'node_run_id': str(node_uid)},
                task_id=str(node_uid),
                queue=WORKFLOW_QUEUE,
                serializer='json',
                ignore_result=True,
                retry=True,
                retry_policy={'max_retries': 3, 'interval_start': 0, 'interval_step': 0.2, 'interval_max': 1},
            )
        except Exception:
            _return_to_ready([node_uid])
            logger.exception(f'Failed to publish workflow node {node_uid}')
        else:
            published += 1
    return published
