from datetime import datetime
from typing import Optional, Self

from pydantic import BaseModel, UUID7, model_validator

from app.models.workflow import Status
from app.schemas.react_flow import WorkflowDefinition
from app.schemas.user import UserPublic


class RunUpdate(BaseModel):
    workflow: Optional[WorkflowDefinition] = None
    status: Optional[Status] = None

    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


class Stat(BaseModel):
    total: int
    waiting: int
    running: int
    success: int
    error: int

    def add_running(self):
        assert self.waiting > 0
        self.waiting -= 1
        self.running += 1

    def add_success(self):
        assert self.running > 0
        self.running -= 1
        self.success += 1

    def add_error(self):
        assert self.running > 0
        self.running -= 1
        self.error += 1


class TaskPublic4Run(BaseModel):
    uid: UUID7
    name: str

    status: Status

    create_time: Optional[datetime]
    start_time: Optional[datetime]
    end_time: Optional[datetime]


class RunPublic(BaseModel):
    uid: UUID7
    name: str
    owner: UserPublic

    status: Status
    task_stats: Optional[Stat] = None

    create_time: datetime
    start_time: datetime
    end_time: datetime


class AutoRunPublic(RunPublic):
    tasks: list[TaskPublic4Run]

    @model_validator(mode='after')
    def get_task_stats(self) -> Self:
        if self.task_stats is None:
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

            self.task_stats = Stat(total=total, success=success, error=error, running=running, waiting=waiting)
        return self
