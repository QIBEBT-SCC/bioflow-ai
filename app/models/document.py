import uuid
from datetime import UTC, datetime
from typing import Optional

import uuid_extensions
from sqlmodel import Field, SQLModel


class ParentDocument(SQLModel, table=True):
    """父文档表：存储完整的文档段落"""
    __tablename__ = 'parent_documents'
    
    uid: uuid.UUID = Field(default_factory=uuid_extensions.uuid7, primary_key=True)
    doc_id: str = Field(index=True)  # 原始文档 ID
    content: str  # 完整的父文档内容
    title: Optional[str] = Field(default=None)  # Markdown 标题
    section: Optional[str] = Field(default=None)  # Markdown 子标题
    create_time: datetime = Field(default_factory=lambda: datetime.now(UTC))