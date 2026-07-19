import uuid
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import Column
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, Relationship, SQLModel

from app.models.user import User  # noqa: TC001
from app.utils.system import utc_now


class ChatSessions(SQLModel, table=True):
    __tablename__ = 'chat_sessions'

    uid: uuid.UUID = Field(primary_key=True)
    owner_id: int = Field(foreign_key='users.id')
    description: str = Field(default='')
    checkpoint_uid: Optional[uuid.UUID] = Field(default=None)
    route: Optional[str] = Field(default=None)
    interrupted: bool = False
    create_time: datetime = Field(default_factory=utc_now)
    update_time: datetime = Field(default_factory=utc_now)

    owner: Optional['User'] = Relationship()


class ChatMessages(SQLModel, table=True):
    __tablename__ = 'chat_messages'

    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: str
    message: dict[str, Any] = Field(sa_column=Column(JSONB))
