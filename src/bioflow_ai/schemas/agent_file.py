import uuid
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class AgentFileKind(StrEnum):
    PLAN = 'plan'
    SAMPLES = 'samples'
    DIAGNOSIS = 'diagnosis'
    UPDATE = 'update'


class AgentFilePublic(BaseModel):
    id: str
    kind: AgentFileKind
    name: str
    project_id: int
    run_uid: uuid.UUID | None = None
    agent_name: str | None = None
    run_status: str | None = None
    size: int
    updated_at: datetime
    content_type: str
    revision: str
    editable: bool
    read_only_reason: str | None = None


class AgentFileUpdate(BaseModel):
    content: str = Field(min_length=1, max_length=1_048_576)
    expected_revision: str = Field(min_length=64, max_length=64, pattern=r'^[0-9a-f]{64}$')
