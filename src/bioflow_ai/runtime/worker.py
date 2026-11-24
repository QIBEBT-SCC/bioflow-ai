import os
from pathlib import Path

from celery.signals import worker_ready

from bioflow_ai.core.config import get_settings
from bioflow_ai.runtime.celery_app import celery_app


@worker_ready.connect
def _mark_worker_ready(**kwargs: object) -> None:  # noqa: ARG001
    """Write the ready marker file once the Celery worker finishes booting."""
    ready_file = os.environ.get('BIOFLOW_READY_FILE')
    if not ready_file:
        return
    path = Path(ready_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    tmp_path.write_text(f'{os.getpid()}\n', encoding='utf-8')
    tmp_path.replace(path)


def agent_main(*, concurrency: int | None = None) -> None:
    """Start the long-running LangGraph agent worker."""
    settings = get_settings()
    resolved_concurrency = settings.celery.AGENT_CONCURRENCY if concurrency is None else concurrency
    args = [
        'worker',
        f'--loglevel={settings.logger.CELERY_LEVEL}',
        '--queues=agent',
        f'--concurrency={resolved_concurrency}',
        f'--hostname=agent-{os.getpid()}@%h',
    ]
    celery_app.worker_main(argv=args)


def execution_main(*, concurrency: int | None = None) -> None:
    """Start the shared worker for workflow nodes and one-off resource jobs."""
    settings = get_settings()
    resolved_concurrency = settings.celery.EXECUTION_CONCURRENCY if concurrency is None else concurrency
    args = [
        'worker',
        f'--loglevel={settings.logger.CELERY_LEVEL}',
        '--queues=workflow-v2,resource',
        f'--concurrency={resolved_concurrency}',
        '--prefetch-multiplier=1',
        f'--hostname=execution-{os.getpid()}@%h',
    ]
    celery_app.worker_main(argv=args)
