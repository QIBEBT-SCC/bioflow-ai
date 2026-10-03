"""Single generic Celery task for workflow-node execution."""

import threading
import uuid
from functools import partial

from celery import Task
from loguru import logger

from bioflow_ai.cli.worker import celery_app
from bioflow_ai.services.workflow import runtime
from bioflow_ai.services.workflow.execution_context import node_execution_context
from bioflow_ai.services.workflow.executors import execute_prepared_node

HEARTBEAT_INTERVAL_SECONDS = 30.0


def _heartbeat_until_stopped(node_uid: uuid.UUID, generation: int, stop: threading.Event) -> None:
    while not stop.wait(HEARTBEAT_INTERVAL_SECONDS):
        try:
            if not runtime.heartbeat_node(node_uid, generation):
                return
        except Exception:
            logger.exception(f'Failed to heartbeat node {node_uid}')


@celery_app.task(
    bind=True,
    name='bioflow_ai.tasks.workflow.execute_node',
    serializer='json',
    ignore_result=True,
    acks_late=True,
    reject_on_worker_lost=True,
    max_retries=0,
)
def execute_node(self: Task, node_run_id: str, generation: int) -> None:
    """Claim and execute one ready node; duplicates and stale generations ACK as no-ops."""
    node_uid = uuid.UUID(node_run_id)
    worker_id = getattr(self.request, 'hostname', None)
    if not runtime.claim_node(node_uid, generation, worker_id):
        logger.info(f'Ignoring duplicate or stale message for node {node_uid}, generation {generation}')
        return

    stop_heartbeat = threading.Event()
    heartbeat = threading.Thread(
        target=_heartbeat_until_stopped,
        args=(node_uid, generation, stop_heartbeat),
        name=f'workflow-heartbeat-{node_uid}',
        daemon=True,
    )
    heartbeat.start()
    try:
        node = runtime.get_node_for_execution(node_uid, generation)
        with node_execution_context(
            output_resolver=partial(runtime.resolve_dependency_output, node_uid, generation),
            node_updater=partial(runtime.update_node_from_executor, node_uid, generation),
        ):
            output = execute_prepared_node(node_uid, node.prepared_data)
        runtime.mark_node_succeeded(node_uid, generation, output)
    except Exception as error:
        runtime.persist_failure_log(node_uid, error)
        runtime.mark_node_failed(node_uid, generation, error)
        logger.exception(f'Node {node_uid} failed')
        raise
    finally:
        stop_heartbeat.set()
        heartbeat.join(timeout=1)
