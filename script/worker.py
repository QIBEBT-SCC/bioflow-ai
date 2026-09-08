import os

from celery import Celery
from celery.signals import worker_process_init
from redis import Redis

from app.core.config import get_settings
from app.core.session import engine

monitor_redis_client = Redis.from_url(get_settings().persistent.cache.MONITOR_CACHE.unicode_string())
celery_app = Celery(
    'tasks',
    broker=get_settings().celery.BROKER.unicode_string(),
    backend=get_settings().celery.BACKEND.unicode_string(),
)

celery_app.conf.update(
    worker_concurrency=4,
    task_serializer='pickle',
    result_serializer='pickle',
    accept_content=['pickle', 'json'],
    broker_transport_options={'confirm_publish': True},
    task_publish_retry=True,
    task_track_started=True,
    task_routes={
        'app.tasks.agent_run.execute_agent_run': {'queue': 'agent'},
        'app.tasks.code_agent.execute_code_agent_session': {'queue': 'acp'},
        'app.tasks.code_agent.login_codex_account': {'queue': 'acp'},
        'app.v2.tasks.workflow.execute_node': {'queue': 'workflow-v2'},
    },
    enable_utc=True,
    timezone='Asia/Shanghai',
)
celery_app.autodiscover_tasks(
    [
        'app.tasks.load_tasks.load_bio_db',
        'app.tasks.load_tasks.load_file',
        'app.tasks.load_tasks.load_genome',
        'app.tasks.load_tasks.static_value',
        'app.tasks.summary_task.summary',
        'app.tasks.archive_operation',
        'app.tasks.code',
        'app.tasks.file_operation',
        'app.tasks.run_tool',
        'app.tasks.agent_run',
        'app.tasks.code_agent',
        'app.v2.tasks.workflow',
    ],
    # pyrefly: ignore [bad-argument-type]
    None,
)


@worker_process_init.connect
def on_worker_process_init(**kwargs) -> None:  # noqa: ARG001
    """Fork 后重置数据库连接池, 防止多进程共享同一连接."""
    engine.dispose()


def main() -> None:
    """Start the Celery worker."""
    args = ['worker', f'--loglevel={get_settings().logger.CELERY_LEVEL}']
    celery_app.worker_main(argv=args)


def agent_main(*, concurrency: int = 1) -> None:
    """Start the shared long-running agent worker for LangGraph and ACP queues."""
    args = [
        'worker',
        f'--loglevel={get_settings().logger.CELERY_LEVEL}',
        '--queues=agent,acp',
        f'--concurrency={concurrency}',
    ]
    celery_app.worker_main(argv=args)


def workflow_v2_main(*, concurrency: int = 4) -> None:
    """Start the isolated worker that only consumes durable V2 node events."""
    args = [
        'worker',
        f'--loglevel={get_settings().logger.CELERY_LEVEL}',
        '--queues=workflow-v2',
        f'--concurrency={concurrency}',
        '--prefetch-multiplier=1',
        f'--hostname=workflow-v2-{os.getpid()}@%h',
    ]
    celery_app.worker_main(argv=args)
