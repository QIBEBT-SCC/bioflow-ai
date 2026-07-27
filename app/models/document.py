import uuid
from datetime import datetime

from sqlalchemy import DateTime, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base
from app.utils.system import random_uid, utc_now


class ParentDocument(Base):
    __tablename__ = 'parent_documents'

    uid: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=random_uid)
    doc_id: Mapped[str] = mapped_column(index=True)
    content: Mapped[str] = mapped_column(Text)
    title: Mapped[str | None] = mapped_column(default=None)
    section: Mapped[str | None] = mapped_column(default=None)
    create_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
