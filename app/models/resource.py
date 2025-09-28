from datetime import datetime, UTC
from enum import IntEnum
from typing import Optional

from pydantic import UUID7
from sqlalchemy import Column, UUID, ForeignKey
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import SQLModel, Field, Relationship

from app.models.user import User


class SampleFileType(IntEnum):
    SEQUENCING_R1 = 0
    SEQUENCING_R2 = 1
    SPECTRUM = 2
    IMAGE = 3


class Sample(SQLModel, table=True):
    __tablename__ = 'samples'

    uid: UUID7 = Field(sa_column=Column(UUID(as_uuid=True), primary_key=True))
    owner_id: int = Field(foreign_key='users.id')
    project_id: int = Field(foreign_key='projects.id')
    sample_name: str
    meta_data: dict = Field(sa_column=Column(JSONB), default_factory=dict)
    create_time: datetime = Field(default_factory=lambda: datetime.now(UTC))

    owner: Optional['User'] = Relationship()


class SampleFiles(SQLModel, table=True):
    __tablename__ = 'sample_files'

    uid: UUID7 = Field(sa_column=Column(UUID(as_uuid=True), primary_key=True))
    sample_uid: UUID7 = Field(sa_column=Column(UUID(as_uuid=True), ForeignKey('samples.uid')))
    data_type: SampleFileType
    file_path: str
    file_format: str
    file_size: int
    md5_checksum: str
    uploaded_time: datetime = Field(default_factory=lambda: datetime.now(UTC))


class BioDb(SQLModel, table=True):
    __tablename__ = 'bio_dbs'

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(unique=True)
    description: Optional[str] = Field(default=None)
    path: str
    size: str
    last_update: Optional[str] = Field(default=None)


class ReferenceGenomes(SQLModel, table=True):
    __tablename__ = 'reference_genomes'

    id: Optional[int] = Field(default=None, primary_key=True)
    ncbi_tax_id: int = Field(unique=True)
    name: str = Field(unique=True)
    aliases: Optional[str] = Field(default=None, description="物种别名，多个别名用|分隔")
    genome_sequences: str
    annotation_features: str

    bowtie2_index: Optional[str]
    bwa_index: Optional[str]
    hisat2_index: Optional[str]
    star_index: Optional[str]
    minimap2_index: Optional[str]
