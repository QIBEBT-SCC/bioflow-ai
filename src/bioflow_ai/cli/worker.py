import os

from celery import Celery
from celery.signals import worker_process_init
from redis import Redis

from bioflow_ai.core.config import get_settings
from bioflow_ai.core.session import engine

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
        'bioflow_ai.tasks.agent_run.execute_agent_run': {'queue': 'agent'},
        'bioflow_ai.tasks.code_agent.execute_code_agent_session': {'queue': 'acp'},
        'bioflow_ai.tasks.code_agent.login_codex_account': {'queue': 'acp'},
        'bioflow_ai.tasks.load_tasks.load_bio_db.download_bio_db_task': {'queue': 'resource'},
        'bioflow_ai.tasks.load_tasks.load_genome.download_genome_admin_task': {'queue': 'resource'},
        'bioflow_ai.tasks.load_tasks.load_genome.build_genome_index_admin_task': {'queue': 'resource'},
        'bioflow_ai.tasks.workflow.execute_node': {'queue': 'workflow-v2'},
    },
    enable_utc=True,
    timezone='Asia/Shanghai',
)
celery_app.autodiscover_tasks(
    [
        'bioflow_ai.tasks.load_tasks.load_bio_db',
        'bioflow_ai.tasks.load_tasks.load_genome',
        'bioflow_ai.tasks.agent_run',
        'bioflow_ai.tasks.code_agent',
        'bioflow_ai.tasks.workflow',
    ],
    # pyrefly: ignore [bad-argument-type]
    None,
)


@worker_process_init.connect
def on_worker_process_init(**kwargs) -> None:  # noqa: ARG001
    """Fork 后重置数据库连接池, 防止多进程共享同一连接."""
    engine.dispose()


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
