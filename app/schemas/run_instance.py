import uuid
from datetime import datetime
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.common import Statistics
from app.schemas.react_flow import Edge, WorkflowDefinition
from app.schemas.user import UserPublic
from app.schemas.workflow_config import Status


class RunData(BaseModel):
    status: Status | None = None
    create_time: datetime | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None


class TaskInfo4Run(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    uid: uuid.UUID
    name: str
    track_progress: bool = Field(default=True, exclude=True)

    status: Status

    create_time: datetime | None
    start_time: datetime | None
    end_time: datetime | None


class SimpleRunPublic(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    uid: uuid.UUID
    name: str
    owner: UserPublic

    project_id: int | None = None
    workflow_uid: uuid.UUID | None = None
    sample_uid: uuid.UUID | None = None

    base_dir: str | None = None

    status: Status
    task_statistics: Statistics | None = None
    create_time: datetime | None
    start_time: datetime | None
    end_time: datetime | None


class RunPublic(SimpleRunPublic):
    tasks: list[TaskInfo4Run]

    @model_validator(mode='after')
    def _get_task_stats(self) -> Self:
        if self.task_statistics is None:
            progress_tasks = [task for task in self.tasks if task.track_progress]
            total = len(progress_tasks)
            success = sum(1 for task in progress_tasks if task.status == Status.SUCCESS)
            error = sum(1 for task in progress_tasks if task.status == Status.ERROR)
            running = sum(1 for task in progress_tasks if task.status == Status.RUNNING)
            waiting = sum(1 for task in progress_tasks if task.status == Status.WAITING)

            execution_error = any(task.status == Status.ERROR for task in self.tasks)
            execution_running = any(task.status == Status.RUNNING for task in self.tasks)
            execution_success = bool(self.tasks) and all(task.status == Status.SUCCESS for task in self.tasks)

            if execution_error:
                self.status = Status.ERROR
            elif execution_success:
                self.status = Status.SUCCESS
            elif execution_running:
                self.status = Status.RUNNING
            else:
                self.status = Status.WAITING

            self.task_statistics = Statistics(
                total=total, success=success, error=error, running=running, waiting=waiting
            )
        return self


class RunDetailPublic(SimpleRunPublic):
    tasks: list[TaskInfo4Run] = Field(exclude=True)
    workflow: WorkflowDefinition = Field(exclude=True)

    nodes: list[dict[str, Any]] = []
    edges: list[Edge] = []

    @model_validator(mode='after')
    def _inject_workflow_data(self) -> Self:
        # 构建 task 查找表: {str(task.uid): task}
        task_map = {str(task.uid): task for task in self.tasks}

        # 将 workflow nodes 转为 dict 并注入 run_data
        nodes = []
        for node in self.workflow.nodes:
            node_dict = node.model_dump()
            task = task_map.get(node.id)
            if task:
                if 'data' not in node_dict:
                    node_dict['data'] = {}
                node_dict['data']['run_data'] = RunData(
                    status=task.status,
                    create_time=task.create_time,
                    start_time=task.start_time,
                    end_time=task.end_time,
                ).model_dump()
            nodes.append(node_dict)

        self.nodes = nodes
        self.edges = self.workflow.edges

        # 从 tasks 中直接读取以计算统计信息
        if self.task_statistics is None:
            progress_tasks = [task for task in self.tasks if task.track_progress]
            total = len(progress_tasks)
            success = sum(1 for task in progress_tasks if task.status == Status.SUCCESS)
            error = sum(1 for task in progress_tasks if task.status == Status.ERROR)
            running = sum(1 for task in progress_tasks if task.status == Status.RUNNING)
            waiting = sum(1 for task in progress_tasks if task.status == Status.WAITING)

            execution_error = any(task.status == Status.ERROR for task in self.tasks)
            execution_running = any(task.status == Status.RUNNING for task in self.tasks)
            execution_success = bool(self.tasks) and all(task.status == Status.SUCCESS for task in self.tasks)

            if execution_error:
                self.status = Status.ERROR
            elif execution_success:
                self.status = Status.SUCCESS
            elif execution_running:
                self.status = Status.RUNNING

            self.task_statistics = Statistics(
                total=total, success=success, error=error, running=running, waiting=waiting
            )

        return self


class RunUpdate(BaseModel):
    workflow: WorkflowDefinition | None = None
    status: Status | None = None

    start_time: datetime | None = None
    end_time: datetime | None = None


class RunRecord(BaseModel):
    """Column-level projection used by the editor run-creation endpoint."""

    model_config = ConfigDict(from_attributes=True)

    uid: uuid.UUID
    name: str
    owner_id: int
    project_id: int | None
    workflow_uid: uuid.UUID | None
    sample_uid: uuid.UUID | None
    base_dir: str | None
    workflow: WorkflowDefinition
    status: Status
    create_time: datetime | None
    start_time: datetime | None
    end_time: datetime | None


class OutputFileItem(BaseModel):
    type: Literal['file'] = 'file'
    path: str
    name: str


class OutputFolderItem(BaseModel):
    type: Literal['folder'] = 'folder'
    path: str
    name: str
    children: list['OutputFileNode'] = Field(default_factory=list)


OutputFileNode = Annotated[OutputFileItem | OutputFolderItem, Field(discriminator='type')]
OutputFolderItem.model_rebuild()


class FileContentRequest(BaseModel):
    path: str
