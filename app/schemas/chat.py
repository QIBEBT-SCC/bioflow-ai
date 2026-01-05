import uuid
from datetime import UTC, datetime
from typing import Optional

from pydantic import BaseModel, Field

from llm.utils.ai_sdk.ui_message import UIMessage


class ChatSessionUpdate(BaseModel):
    uid: uuid.UUID
    description: Optional[str] = None
    checkpoint_uid: Optional[uuid.UUID] = None
    interrupted: Optional[bool] = None
    update_time: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ChatSessionPublic(BaseModel):
    uid: uuid.UUID
    description: str
    checkpoint_uid: Optional[uuid.UUID]
    interrupted: bool
    create_time: datetime
    update_time: datetime


class ChatResponse(BaseModel):
    id: str
    messages: list[UIMessage]
    rebuild: bool
    resume: bool = False
    approved: bool = False


class PaginatedChatSessions(BaseModel):
    """分页的聊天会话响应"""

    total: int = Field(description='总记录数')
    offset: int = Field(description='当前偏移量')
    limit: int = Field(description='每页数量')
    has_more: bool = Field(description='是否还有更多数据')
    data: list[ChatSessionPublic] = Field(description='聊天会话列表')
