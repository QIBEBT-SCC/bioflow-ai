from pathlib import Path
from typing import Any, Optional

from celery import Task

from app.models.workflow import ToolOutput
from app.tasks.base.base_task import BaseBioTask
from worker import celery_app


@celery_app.task(bind=True, pydantic=True, base=BaseBioTask)
def load_file_task(
    self: Task,  # noqa: ARG001
    result_dict: Optional[dict[str, Any]] = None,  # noqa: ARG001
    *,
    file_path: str,
) -> ToolOutput:
    """加载普通文件."""
    if not Path(file_path).exists():
        err_msg = f'File "{file_path}" not found.'
        raise FileNotFoundError(err_msg)

    return ToolOutput(result={'file_path': file_path})


@celery_app.task(bind=True, pydantic=True, base=BaseBioTask)
def load_sequence_task(
    self: Task,  # noqa: ARG001
    result_dict: Optional[dict[str, Any]] = None,  # noqa: ARG001
    *,
    r1: str,
    r2: str,
) -> ToolOutput:
    """加载双端序列文件."""
    if not Path(r1).exists():
        err_msg = f'File "{r1}" not found.'
        raise FileNotFoundError(err_msg)

    if not Path(r2).exists():
        err_msg = f'File "{r2}" not found.'
        raise FileNotFoundError(err_msg)

    return ToolOutput(result={'r1': r1, 'r2': r2})
