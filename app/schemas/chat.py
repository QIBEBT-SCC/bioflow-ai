from typing import Optional

from pydantic import BaseModel, Field, UUID7
from datetime import datetime, UTC


class ChatSessionUpdate(BaseModel):
    uid: str
    description: Optional[str] = None
    update_time: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ChatSessionPublic(BaseModel):
    uid: UUID7
    description: str
    create_time: datetime
    update_time: datetime