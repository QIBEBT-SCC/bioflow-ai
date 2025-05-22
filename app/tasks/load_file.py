from pathlib import Path
from typing import Optional, Any

from celery import Task
from pydantic import BaseModel

from app.tasks.base.base_task import BaseBioTask
from worker import celery_app


class LoadFileOutput(BaseModel):
    file_path: str


@celery_app.task(bind=True, pydantic=True, base=BaseBioTask)
def load_file_task(
    self: Task, result_dict: Optional[dict[str, Any]] = None, *, file_path: str
) -> LoadFileOutput:
    if Path(file_path).exists(follow_symlinks=True):
        return LoadFileOutput(file_path=file_path)

    raise FileNotFoundError
