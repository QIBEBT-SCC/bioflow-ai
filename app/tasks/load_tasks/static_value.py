from typing import Any, Literal, Optional

from celery import Task

from app.models.workflow import ToolOutput
from app.tasks.base.base_task import BaseBioTask
from worker import celery_app


@celery_app.task(bind=True, pydantic=True, base=BaseBioTask)
def set_value_task(
    self: Task, result_dict: Optional[dict[str, Any]] = None, *, value: str, value_type: Literal['string']
) -> ToolOutput:
    match value_type:
        case 'string':
            return ToolOutput(result={'value': value})
        case 'number':
            if '.' in value:
                return ToolOutput(result={'value': float(value)})
            else:
                return ToolOutput(result={'value': int(value)})
        case _:
            raise ValueError
