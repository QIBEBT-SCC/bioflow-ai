from datetime import datetime, UTC
from typing import Optional

from sqlalchemy import Column, BigInteger
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import SQLModel, Field, Relationship

from app.models.user import User


class SequencingData(SQLModel, table=True):
    __tablename__ = 'sequencing_datas'

    uid: str = Field(default=None, primary_key=True)
    owner_id: int = Field(foreign_key='users.id')
    project_id: int = Field(foreign_key='projects.id')
    sample_name: str
    storage_path: str
    file_size: int = Field(sa_column=Column(BigInteger()))
    md5_checksum: str
    meta_data: dict = Field(sa_column=Column(JSONB), default_factory=dict)
    create_time: datetime = Field(default_factory=lambda: datetime.now(UTC))

    owner: Optional['User'] = Relationship()
