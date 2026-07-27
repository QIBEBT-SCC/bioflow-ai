import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import ForeignKey
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base
from app.models.user import User
from app.utils.system import utc_now


class ChatSessions(Base):
    __tablename__ = 'chat_sessions'

    uid: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    owner_id: Mapped[int] = mapped_column(ForeignKey(User.id), nullable=False)
    description: Mapped[str] = mapped_column(default='')
    checkpoint_uid: Mapped[uuid.UUID | None] = mapped_column(default=None)
    route: Mapped[str | None] = mapped_column(default=None)
    interrupted: Mapped[bool] = mapped_column(default=False)
    create_time: Mapped[datetime] = mapped_column(default=utc_now)
    update_time: Mapped[datetime] = mapped_column(default=utc_now)

    owner: Mapped[User | None] = relationship(User)


class ChatMessages(Base):
    __tablename__ = 'chat_messages'

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    session_id: Mapped[str]
    message: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=True)
