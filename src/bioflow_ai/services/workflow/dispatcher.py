"""Transactional Outbox publisher for the workflow queue."""

import uuid
from datetime import datetime, timedelta

from loguru import logger
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from bioflow_ai.infrastructure.database import engine
from bioflow_ai.models.workflow import NodeRun, WorkflowOutbox, WorkflowRun
from bioflow_ai.schemas.workflow_run import NodeRunStatus, OutboxStatus
from bioflow_ai.services.workflow.runtime import recover_runtime
from bioflow_ai.utils.system import utc_now

WORKFLOW_QUEUE = 'workflow-v2'
DISPATCH_CLAIM_TIMEOUT = timedelta(seconds=30)


def _has_active_claim(event: WorkflowOutbox, now: datetime) -> bool:
    if event.claim_token is None or event.claimed_at is None:
        return False
    claimed_at = event.claimed_at
    if claimed_at.tzinfo is None:
        now = now.astimezone().replace(tzinfo=None)
    return claimed_at >= now - DISPATCH_CLAIM_TIMEOUT


def _reserve_event(event_uid: uuid.UUID) -> tuple[uuid.UUID, str, int] | None:
    """Lease one event and commit QUEUED before a message can reach a worker."""
    now = utc_now()
    with Session(engine) as session, session.begin():
        event = session.get(WorkflowOutbox, event_uid, with_for_update=True)
        if event is None or event.status != OutboxStatus.PENDING:
            return None
        if _has_active_claim(event, now):
            return None

        node = session.get(NodeRun, event.node_run_uid, with_for_update=True)
        run = session.get(WorkflowRun, event.run_uid)
        if node is None or run is None or node.generation != event.generation or run.generation != event.generation:
            event.status = OutboxStatus.PUBLISHED
            event.published_at = now
            event.claim_token = None
            event.claimed_at = None
            event.last_error = 'Discarded stale generation event'
            return None
        if node.status == NodeRunStatus.READY:
            node.status = NodeRunStatus.QUEUED
            node.queued_at = now
        elif node.status != NodeRunStatus.QUEUED:
            event.status = OutboxStatus.PUBLISHED
            event.published_at = now
            event.claim_token = None
            event.claimed_at = None
            event.last_error = f'Discarded event for node in {node.status.value}'
            return None

        token = uuid.uuid4()
        event.claim_token = token
        event.claimed_at = now
        event.publish_attempts += 1
        event.last_error = None
        return token, str(node.uid), event.generation


def _record_publish_failure(event_uid: uuid.UUID, token: uuid.UUID, error: BaseException) -> None:
    with Session(engine) as session, session.begin():
        event = session.get(WorkflowOutbox, event_uid, with_for_update=True)
        if event is None or event.status != OutboxStatus.PENDING or event.claim_token != token:
            return
        event.claim_token = None
        event.claimed_at = None
        event.last_error = str(error)[:4000]
        node = session.get(NodeRun, event.node_run_uid, with_for_update=True)
        if node is not None and node.generation == event.generation and node.status == NodeRunStatus.QUEUED:
            node.status = NodeRunStatus.READY
            node.queued_at = None


def _record_publish_success(event_uid: uuid.UUID, token: uuid.UUID) -> None:
    with Session(engine) as session, session.begin():
        event = session.get(WorkflowOutbox, event_uid, with_for_update=True)
        if event is None or event.status != OutboxStatus.PENDING or event.claim_token != token:
            return
        event.status = OutboxStatus.PUBLISHED
        event.published_at = utc_now()
        event.claim_token = None
        event.claimed_at = None
        event.last_error = None


def dispatch_pending(*, limit: int = 100) -> int:
    """Publish one leased batch after selecting event IDs with SKIP LOCKED."""
    expired_before = utc_now() - DISPATCH_CLAIM_TIMEOUT
    with Session(engine) as session, session.begin():
        event_ids = list(
            session.scalars(
                select(WorkflowOutbox.uid)
                .where(
                    WorkflowOutbox.status == OutboxStatus.PENDING,
                    or_(
                        WorkflowOutbox.claim_token.is_(None),
                        WorkflowOutbox.claimed_at.is_(None),
                        WorkflowOutbox.claimed_at < expired_before,
                    ),
                )
                .order_by(WorkflowOutbox.create_time)
                .limit(limit)
                .with_for_update(skip_locked=True)
            )
        )

    published = 0
    for event_uid in event_ids:
        reserved = _reserve_event(event_uid)
        if reserved is None:
            continue
        token, node_run_id, generation = reserved
        try:
            from bioflow_ai.tasks.workflow import execute_node

            execute_node.apply_async(
                kwargs={'node_run_id': node_run_id, 'generation': generation},
                task_id=node_run_id,
                queue=WORKFLOW_QUEUE,
                serializer='json',
                ignore_result=True,
                retry=True,
                retry_policy={'max_retries': 3, 'interval_start': 0, 'interval_step': 0.2, 'interval_max': 1},
            )
        except Exception as error:
            _record_publish_failure(event_uid, token, error)
            logger.exception(f'Failed to publish workflow outbox event {event_uid}')
        else:
            _record_publish_success(event_uid, token)
            published += 1
    return published


def recover(*, stale_after_seconds: int = 300) -> dict[str, int]:
    """Run the dispatcher recovery scan."""
    return recover_runtime(stale_after=timedelta(seconds=stale_after_seconds))
