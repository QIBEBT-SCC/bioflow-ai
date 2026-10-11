"""Long-running workflow node dispatcher."""

import signal
import threading
import time
from datetime import timedelta

from loguru import logger

from bioflow_ai.infrastructure.database import create_db_and_tables
from bioflow_ai.services.workflow.dispatcher import dispatch_pending, requeue_unpublished
from bioflow_ai.services.workflow.foreach import advance_foreach
from bioflow_ai.services.workflow.runtime import recover_runtime

POLL_INTERVAL_SECONDS = 0.25
RECOVERY_INTERVAL_SECONDS = 30.0
FOREACH_INTERVAL_SECONDS = 2.0


def main(*, stale_after_seconds: int = 300) -> None:
    """Publish pending nodes and periodically repair durable state."""
    create_db_and_tables()
    stopping = threading.Event()

    def stop(_signum: int, _frame: object) -> None:
        stopping.set()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    stale_after = timedelta(seconds=stale_after_seconds)
    requeued = requeue_unpublished()
    if requeued:
        logger.info(f'Requeued {requeued} workflow nodes left QUEUED by a previous dispatcher')
    last_recovery = 0.0
    last_foreach = 0.0
    logger.info('workflow dispatcher started')
    while not stopping.is_set():
        published = dispatch_pending()
        now = time.monotonic()
        if now - last_foreach >= FOREACH_INTERVAL_SECONDS:
            advance_foreach()
            last_foreach = now
        if now - last_recovery >= RECOVERY_INTERVAL_SECONDS:
            recover_runtime(stale_after=stale_after)
            last_recovery = now
        if published == 0:
            stopping.wait(POLL_INTERVAL_SECONDS)
    logger.info('workflow dispatcher stopped')
