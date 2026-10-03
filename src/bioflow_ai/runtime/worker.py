import os

from bioflow_ai.core.config import get_settings
from bioflow_ai.runtime.celery_app import celery_app


def agent_main(*, concurrency: int = 1) -> None:
    """Start the shared long-running agent worker for LangGraph and ACP queues."""
    args = [
        'worker',
        f'--loglevel={get_settings().logger.CELERY_LEVEL}',
        '--queues=agent,acp',
        f'--concurrency={concurrency}',
    ]
    celery_app.worker_main(argv=args)


def execution_main(*, concurrency: int = 4) -> None:
    """Start the shared worker for workflow nodes and one-off resource jobs."""
    args = [
        'worker',
        f'--loglevel={get_settings().logger.CELERY_LEVEL}',
        '--queues=workflow-v2,resource',
        f'--concurrency={concurrency}',
        '--prefetch-multiplier=1',
        f'--hostname=execution-{os.getpid()}@%h',
    ]
    celery_app.worker_main(argv=args)
