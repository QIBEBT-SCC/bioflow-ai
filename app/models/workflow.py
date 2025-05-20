from datetime import datetime
from typing import Optional

from pydantic import UUID7
from sqlalchemy import Column, UUID
from sqlmodel import SQLModel, Field, Relationship

from app.models.decorator.special_field import PydanticJson
from app.models.react_flow import WorkflowDefinition
from app.models.user import User


# TABLE project_raw

# TABLE run_instance
# uid, owner, project_uid, templet(str), status, time..

# TABLE task
# uid, instance_uid, owner, task_type(name), web_node_id,other_args, status, local_path, result, time...


class Workflow(SQLModel, table=True):
    __tablename__ = 'workflows'

    uid: UUID7 = Field(sa_column=Column(UUID(as_uuid=True), primary_key=True))
    name: str
    owner_id: int = Field(foreign_key='users.id')
    workflow: WorkflowDefinition = Field(sa_column=Column(PydanticJson(WorkflowDefinition)))
    public: bool = False

    owner: Optional['User'] = Relationship()


class TaskInstance(SQLModel):
    __tablename__ = 'task_instances'

    uid: str = Field(primary_key=True)
    instance_uid: str
    owner: int
    name: str
    commands: str
    result: str
    status: str
    create_time: Optional[datetime] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
