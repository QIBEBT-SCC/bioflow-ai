import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.utils.system import utc_now
from llm.utils.ai_sdk.ui_message import UIMessage


class ChatSessionUpdate(BaseModel):
    uid: str

    description: str | None = None

    checkpoint_uid: uuid.UUID | None = None
    route: str | None = None
    interrupted: bool | None = None

    update_time: datetime = Field(default_factory=utc_now)


class ChatSessionPublic(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    uid: uuid.UUID
    description: str
    checkpoint_uid: uuid.UUID | None
    interrupted: bool
    create_time: datetime
    update_time: datetime


class ChatResponse(BaseModel):
    id: str
    messages: list[UIMessage]
    resume: bool = False
    approved: bool = False

    project_id: int | None = None


class ChatSessionPage(BaseModel):
    limit: int
    has_more: bool
    next_cursor: datetime | None = None
    data: list[ChatSessionPublic]
