from datetime import datetime
from typing import Optional

from pydantic import BaseModel, UUID7
from sqlmodel import SQLModel

from app.models.workflow import Status
from app.schemas.react_flow import WorkflowDefinition


class RunUpdate(SQLModel):
    workflow: Optional[WorkflowDefinition] = None
    status: Optional[Status] = None

    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


class RunInfo4Task(BaseModel):
    uid: UUID7
    name: str
