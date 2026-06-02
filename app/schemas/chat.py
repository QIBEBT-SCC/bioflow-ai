import uuid
from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field

from llm.utils.ai_sdk.ui_message import UIMessage


class ChatSessionUpdate(BaseModel):
    uid: uuid.UUID
    description: str | None = None
    checkpoint_uid: uuid.UUID | None = None
    interrupted: bool | None = None
    update_time: datetime = Field(default_factory=lambda: datetime.now(UTC))


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
    rebuild: bool
    resume: bool = False
    approved: bool = False


class ChatSessionPage(BaseModel):
    limit: int = Field(description='每页数量')
    has_more: bool = Field(description='是否还有更多数据')
    next_cursor: datetime | None = Field(default=None, description='下一页游标, 为最后一条记录的 update_time; 无更多数据时为 null')
    data: list[ChatSessionPublic] = Field(description='聊天会话列表')
