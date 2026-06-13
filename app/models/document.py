import uuid
from datetime import datetime
from typing import Optional

from sqlmodel import Field, SQLModel

from app.utils.system import random_uid, utc_now


class ParentDocument(SQLModel, table=True):
    __tablename__ = 'parent_documents'

    uid: uuid.UUID = Field(default_factory=random_uid, primary_key=True)
    doc_id: str = Field(index=True)
    content: str
    title: Optional[str] = Field(default=None)
    section: Optional[str] = Field(default=None)
    create_time: datetime = Field(default_factory=utc_now)
