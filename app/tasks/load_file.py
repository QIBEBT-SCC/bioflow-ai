from pathlib import Path
from typing import Optional, Any

from celery import Task

from app.tasks.base.base_task import BaseBioTask, ToolOutput
from worker import celery_app


@celery_app.task(bind=True, pydantic=True, base=BaseBioTask)
def load_file_task(
    self: Task, result_dict: Optional[dict[str, Any]] = None, *, file_path: str
) -> ToolOutput:
    if Path(file_path).exists(follow_symlinks=True):
        return ToolOutput(result={'file_path': file_path})

    raise FileNotFoundError
