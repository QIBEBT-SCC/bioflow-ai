from celery import Celery
from redis import Redis

from app.core.config import get_settings
from app.models.chat import ChatMessages, ChatSessions  # noqa: F401
from app.models.document import ParentDocument  # noqa: F401
from app.models.project import Project, ProjectStar, ProjectTag, ProjectTagLink  # noqa: F401
from app.models.project_workflow import ProjectWorkflow  # noqa: F401
from app.models.resource import BioDb, File, ReferenceGenomes, Sample, SampleFiles  # noqa: F401
from app.models.setting import LLMModel, LLMProvider, LLMSetting, LLMStatistic  # noqa: F401
from app.models.tools import (  # noqa: F401
    ToolDefine,
    ToolGroup,
    ToolHelpDoc,
    ToolImage,
    ToolTag,
    ToolTagLink,
)
from app.models.user import User  # noqa: F401
from app.models.workflow import RunInstance, TaskInstance, TaskMonitor, Workflow  # noqa: F401

monitor_redis_client = Redis.from_url(get_settings().persistent.MONITOR_CACHE)
celery_app = Celery('tasks', broker=get_settings().celery.BROKER, backend=get_settings().celery.BACKEND)

celery_app.conf.update(
    worker_concurrency=4,
    task_serializer='pickle',
    result_serializer='pickle',
    accept_content=['pickle'],
    enable_utc=True,
    timezone='Asia/Shanghai',
)
celery_app.autodiscover_tasks(
    [
        'app.tasks.load_tasks.load_file',
        'app.tasks.load_tasks.load_genome',
        'app.tasks.load_tasks.static_value',
        'app.tasks.code',
        'app.tasks.file_operation',
        'app.tasks.run_tool',
    ],
    None,
)


if __name__ == '__main__':
    args = ['worker', f'--loglevel={get_settings().logger.CELERY_LEVEL}']
    celery_app.worker_main(argv=args)
