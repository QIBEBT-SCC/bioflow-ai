from datetime import datetime
from typing import Optional

from sqlmodel import SQLModel, Field, Relationship


# TABLE project_raw

# TABLE run_instance
# uid, owner, project_uid, templet(str), status, time..

# TABLE task
# uid, instance_uid, owner, task_type(name), web_node_id,other_args, status, local_path, result, time...


class WorkflowInstance(SQLModel, table=True):
    __tablename__ = 'workflow_instances'

    uid: str = Field(primary_key=True)
    name: str
    owner: str = Relationship()
    project: str
    workflow: str
    status: str
    create_time: Optional[datetime] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


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
