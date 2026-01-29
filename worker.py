from celery import Celery
from redis import Redis

from app.core.config import get_settings

# ==================== 重要：导入所有模型以加载 SQLAlchemy 元数据 ====================
# 必须在 Celery worker 启动时导入所有模型，否则外键约束无法解析
from app.models.user import User  # noqa: F401
from app.models.project import Project, ProjectTag, ProjectTagLink, ProjectStar  # noqa: F401
from app.models.resource import Sample, SampleFiles, BioDb, ReferenceGenomes, File  # noqa: F401
from app.models.workflow import Workflow, RunInstance, TaskInstance, TaskMonitor  # noqa: F401
from app.models.tools import (  # noqa: F401
    ToolDefine,
    ToolGroup,
    ToolTag,
    ToolTagLink,
    ToolImage,
    ToolHelpDoc,
)
from app.models.project_workflow import ProjectWorkflow  # noqa: F401
from app.models.chat import ChatSessions, ChatMessages  # noqa: F401
from app.models.document import ParentDocument  # noqa: F401
from app.models.setting import LLMProvider, LLMModel, LLMSetting, LLMStatistic  # noqa: F401
# ==================== 模型导入结束 ====================

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
        'app.tasks.run_tool',
    ],
    None,
)


if __name__ == '__main__':
    args = ['worker', f'--loglevel={get_settings().logger.CELERY_LEVEL}']
    celery_app.worker_main(argv=args)
