import uuid
from datetime import UTC, datetime
from enum import IntEnum
from typing import Optional

import uuid_extensions
from sqlalchemy import BigInteger, Column
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, Relationship, SQLModel

from app.models.user import User


class SampleFileType(IntEnum):
    SEQUENCING_R1 = 0
    SEQUENCING_R2 = 1
    SEQUENCING = 2
    SPECTRUM = 3
    IMAGE = 4


class Sample(SQLModel, table=True):
    __tablename__ = 'samples'

    uid: uuid.UUID = Field(default_factory=uuid_extensions.uuid7, primary_key=True)
    owner_id: int = Field(foreign_key='users.id')
    project_id: int = Field(foreign_key='projects.id')
    sample_name: str
    meta_data: dict = Field(sa_column=Column(JSONB), default_factory=dict)
    create_time: datetime = Field(default_factory=lambda: datetime.now(UTC))

    owner: Optional['User'] = Relationship()
    sample_files: list['SampleFiles'] = Relationship(back_populates='sample')


class SampleFiles(SQLModel, table=True):
    __tablename__ = 'sample_files'

    uid: uuid.UUID = Field(default_factory=uuid_extensions.uuid7, primary_key=True)
    sample_uid: uuid.UUID = Field(foreign_key='samples.uid')
    data_type: SampleFileType
    file_path: str
    file_format: str
    file_size: int = Field(sa_type=BigInteger)
    md5_checksum: str
    uploaded_time: datetime = Field(default_factory=lambda: datetime.now(UTC))

    sample: Optional['Sample'] = Relationship(back_populates='sample_files')


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
    accession: str
    aliases: Optional[str] = Field(default=None, description='物种别名，多个别名用|分隔')
    genome_sequences: str
    annotation_gff: str
    annotation_gtf: str

    bowtie2_index: Optional[str] = None
    bwa_index: Optional[str] = None
    hisat2_index: Optional[str] = None
    star_index: Optional[str] = None
    minimap2_index: Optional[str] = None


class File(SQLModel, table=True):
    __tablename__ = 'files'

    id: Optional[int] = Field(default=None, primary_key=True)
    sha256: str = Field(unique=True)
    file_name: str
    s3_path: str
    upload_time: datetime = Field(default_factory=lambda: datetime.now(UTC))
