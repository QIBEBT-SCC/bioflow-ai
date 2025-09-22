import uuid
from datetime import datetime, UTC
from typing import Optional

import uuid_extensions
from sqlmodel import Field, Relationship, SQLModel

from app.models.user import User


class ChatSessions(SQLModel, table=True):
    __tablename__ = 'chat_sessions'

    uid: uuid.UUID = Field(default_factory=uuid_extensions.uuid7, primary_key=True)
    owner_id: int = Field(foreign_key='users.id')
    description: str = Field(default='')
    checkpoint_uid: Optional[uuid.UUID] = None
    interrupted: bool = False
    create_time: datetime = Field(default_factory=lambda: datetime.now(UTC))
    update_time: datetime = Field(default_factory=lambda: datetime.now(UTC))

    owner: Optional['User'] = Relationship()
