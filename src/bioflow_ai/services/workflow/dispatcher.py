"""Publisher that moves READY workflow nodes onto the Celery queue."""

import threading
import time
import uuid

from loguru import logger
from sqlalchemy.orm import Session

from bioflow_ai.crud import node_run as node_run_crud
from bioflow_ai.infrastructure.database import engine
from bioflow_ai.schemas.workflow_run import NodeRunStatus
from bioflow_ai.services.workflow.foreach import advance_foreach
from bioflow_ai.services.workflow.runtime import recover_runtime
from bioflow_ai.tasks.workflow import execute_node
from bioflow_ai.utils.system import utc_now

POLL_INTERVAL_SECONDS = 0.25
ERROR_BACKOFF_SECONDS = 5.0
RECOVERY_INTERVAL_SECONDS = 30.0
FOREACH_INTERVAL_SECONDS = 2.0


def _queue_ready_nodes(limit: int) -> list[uuid.UUID]:
    """Commit READY -> QUEUED before any message can reach a worker."""
    with Session(engine) as session, session.begin():
        node_uids = node_run_crud.lock_ready_uids(session, limit)
        if node_uids:
            node_run_crud.update_if_status(
                session, node_uids, [NodeRunStatus.READY], {'status': NodeRunStatus.QUEUED, 'queued_at': utc_now()}
            )
    return node_uids


def _return_to_ready(node_uids: list[uuid.UUID]) -> None:
    with Session(engine) as session, session.begin():
        node_run_crud.update_if_status(
            session, node_uids, [NodeRunStatus.QUEUED], {'status': NodeRunStatus.READY, 'queued_at': None}
        )


def requeue_unpublished() -> None:
    """Return QUEUED nodes to READY when the dispatcher starts.

    A dispatcher that stopped between committing QUEUED and publishing leaves
    nodes without a message. Republishing a node whose message did reach the
    broker is harmless: ``claim_node`` lets only one message start it.
    """
    with Session(engine) as session, session.begin():
        requeued = node_run_crud.update_by_status(
            session, [NodeRunStatus.QUEUED], {'status': NodeRunStatus.READY, 'queued_at': None}
        )
    if requeued:
        logger.info(f'Requeued {requeued} workflow nodes left QUEUED by a previous dispatcher')


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


class WorkflowDispatcher:
    """Publish pending nodes and periodically repair durable state.

    Every step locks the rows it touches, so several API instances may each run a dispatcher.
    """

    def __init__(self) -> None:
        self._stopping = threading.Event()
        self._thread = threading.Thread(target=self._run, name='workflow-dispatcher', daemon=True)

    def start(self) -> None:
        """Start the dispatcher thread."""
        self._thread.start()

    def stop(self, timeout: float = 10.0) -> None:
        """Signal the loop to stop and wait for the current iteration to finish."""
        self._stopping.set()
        self._thread.join(timeout)

    def _run(self) -> None:
        try:
            requeue_unpublished()
        except Exception:
            logger.exception('Failed to requeue unpublished workflow nodes')
        last_recovery = 0.0
        last_foreach = 0.0
        logger.info('workflow dispatcher started')
        while not self._stopping.is_set():
            try:
                published = dispatch_pending()
                now = time.monotonic()
                if now - last_foreach >= FOREACH_INTERVAL_SECONDS:
                    advance_foreach()
                    last_foreach = now
                if now - last_recovery >= RECOVERY_INTERVAL_SECONDS:
                    recover_runtime()
                    last_recovery = now
            except Exception:
                # A dead thread would silently stop all workflows while the API keeps serving.
                logger.exception('Workflow dispatcher iteration failed')
                self._stopping.wait(ERROR_BACKOFF_SECONDS)
                continue
            if published == 0:
                self._stopping.wait(POLL_INTERVAL_SECONDS)
        logger.info('workflow dispatcher stopped')
