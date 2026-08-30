from typing import Literal

from celery import Task

from app.schemas.workflow_config import ToolOutput
from app.tasks.base.base_task import BaseBioTask
from script.worker import celery_app


@celery_app.task(bind=True, pydantic=True, base=BaseBioTask)
def set_value_task(
    self: Task,  # noqa: ARG001
    result_dict: ToolOutput | None = None,  # noqa: ARG001
    *,
    value: str,
    value_type: Literal['string'],
) -> ToolOutput:
    """返回变量值."""
    match value_type:
        case 'string':
            return ToolOutput(result={'value': value})
        # case 'number':
        #     if '.' in value:
        #         return ToolOutput(result={'value': float(value)})
        #     return ToolOutput(result={'value': int(value)})
        case _:
            raise ValueError
