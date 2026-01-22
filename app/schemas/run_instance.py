import uuid
from datetime import datetime
from typing import Optional, Self

from pydantic import BaseModel, ConfigDict, model_validator

from app.models.workflow import Status
from app.schemas.common import Statistics
from app.schemas.react_flow import WorkflowDefinition
from app.schemas.user import UserPublic


class TaskInfo4Run(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    uid: uuid.UUID
    name: str

    status: Status

    create_time: Optional[datetime]
    start_time: Optional[datetime]
    end_time: Optional[datetime]


class SimpleRunPublic(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    uid: uuid.UUID
    name: str
    owner: UserPublic

    project_id: Optional[int] = None
    workflow_uid: Optional[uuid.UUID] = None
    sample_uid: Optional[uuid.UUID] = None

    status: Status
    task_statistics: Optional[Statistics] = None

    create_time: Optional[datetime]
    start_time: Optional[datetime]
    end_time: Optional[datetime]


class RunPublic(SimpleRunPublic):
    tasks: list[TaskInfo4Run]

    @model_validator(mode='after')
    def get_task_stats(self) -> Self:
        if self.task_statistics is None:
            total = len(self.tasks)
            success = sum(1 for task in self.tasks if task.status == Status.SUCCESS)
            error = sum(1 for task in self.tasks if task.status == Status.ERROR)
            running = sum(1 for task in self.tasks if task.status == Status.RUNNING)
            waiting = sum(1 for task in self.tasks if task.status == Status.WAITING)

            if running > 0:
                self.status = Status.RUNNING
            elif success == total:
                self.status = Status.SUCCESS
            elif error > 0:
                self.status = Status.ERROR
            else:
                self.status = Status.WAITING

            self.task_statistics = Statistics(
                total=total, success=success, error=error, running=running, waiting=waiting
            )
        return self


class RunUpdate(BaseModel):
    workflow: Optional[WorkflowDefinition] = None
    status: Optional[Status] = None

    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
