"""Long-running V2 transactional Outbox dispatcher."""

import signal
import threading
import time

from loguru import logger

from app.core.session import create_db_and_tables
from app.v2.services.dispatcher import dispatch_pending, recover

POLL_INTERVAL_SECONDS = 0.25
RECOVERY_INTERVAL_SECONDS = 30.0


def main(*, stale_after_seconds: int = 300) -> None:
    """Publish pending V2 nodes and periodically repair durable state."""
    create_db_and_tables()
    stopping = threading.Event()

    def stop(_signum: int, _frame: object) -> None:
        stopping.set()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    last_recovery = 0.0
    logger.info('workflow-v2 dispatcher started')
    while not stopping.is_set():
        published = dispatch_pending()
        now = time.monotonic()
        if now - last_recovery >= RECOVERY_INTERVAL_SECONDS:
            recover(stale_after_seconds=stale_after_seconds)
            last_recovery = now
        if published == 0:
            stopping.wait(POLL_INTERVAL_SECONDS)
    logger.info('workflow-v2 dispatcher stopped')
