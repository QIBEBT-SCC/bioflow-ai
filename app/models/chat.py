from datetime import datetime, UTC
from typing import Optional

from pydantic import UUID7
from sqlalchemy import Column, UUID
from sqlmodel import Field, Relationship, SQLModel

from app.models.user import User


class ChatSessions(SQLModel, table=True):
    __tablename__ = 'chat_sessions'

    uid: UUID7 = Field(sa_column=Column(UUID(as_uuid=True), primary_key=True))
    owner_id: int = Field(foreign_key='users.id')
    description: str = Field(default='')
    create_time: datetime = Field(default_factory=lambda: datetime.now(UTC))
    update_time: datetime = Field(default_factory=lambda: datetime.now(UTC))

    owner: Optional['User'] = Relationship()
