from pathlib import Path
from typing import Any, Optional

from celery import Task
from pydantic import BaseModel

from app.tasks.base.base_task import BaseBioTask, ToolOutput
from worker import celery_app


class SequenceInput(BaseModel):
    r1: Path
    r2: Path


@celery_app.task(bind=True, pydantic=True, base=BaseBioTask)
def load_file_task(self: Task, result_dict: Optional[dict[str, Any]] = None, *, file_path: str) -> ToolOutput:
    if Path(file_path).exists(follow_symlinks=True):
        return ToolOutput(result={'file_path': file_path})

    raise FileNotFoundError


@celery_app.task(bind=True, pydantic=True, base=BaseBioTask)
def load_sequence_task(self: Task, result_dict: Optional[dict[str, Any]] = None, *, args: dict[str, Any]) -> ToolOutput:
    sequence_input = SequenceInput.model_validate(args)
    if sequence_input.r1.exists(follow_symlinks=True) and sequence_input.r2.exists(follow_symlinks=True):
        return ToolOutput(result={'r1': str(sequence_input.r1), 'r2': str(sequence_input.r2)})

    raise FileNotFoundError
