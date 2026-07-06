import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import BigInteger, Column, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, Relationship, SQLModel

from app.models.user import User  # noqa: TC001
from app.utils.system import random_uid, utc_now


class Sample(SQLModel, table=True):
    __tablename__ = 'samples'
    __table_args__ = (UniqueConstraint('project_id', 'sample_name', name='uq_sample_project_name'),)

    uid: uuid.UUID = Field(default_factory=random_uid, primary_key=True)
    owner_id: int = Field(foreign_key='users.id')
    project_id: int = Field(foreign_key='projects.id', ondelete='CASCADE')
    sample_name: str
    meta_data: dict[str, str] = Field(sa_column=Column(JSONB), default_factory=dict)
    create_time: datetime = Field(default_factory=utc_now)

    owner: Optional['User'] = Relationship()
    sample_files: list['SampleFiles'] = Relationship(back_populates='sample', cascade_delete=True)


class SampleFiles(SQLModel, table=True):
    __tablename__ = 'sample_files'
    __table_args__ = (UniqueConstraint('sample_uid', 'tag', name='uq_sample_file_tag'),)

    uid: uuid.UUID = Field(default_factory=random_uid, primary_key=True)
    sample_uid: uuid.UUID = Field(foreign_key='samples.uid', ondelete='CASCADE')
    file_path: str
    file_format: Optional[str] = None
    file_size: int = Field(sa_type=BigInteger, default=0)
    md5_checksum: Optional[str] = None
    tag: str = Field(max_length=50, index=True)
    uploaded_time: datetime = Field(default_factory=utc_now)

    sample: Optional['Sample'] = Relationship(back_populates='sample_files')


class BioDb(SQLModel, table=True):
    __tablename__ = 'bio_dbs'

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(unique=True)
    description: Optional[str] = Field(default=None)
    path: Optional[str] = Field(default=None)
    size: Optional[str] = Field(default=None)
    last_update: Optional[str] = Field(default=None)
    download_command: Optional[str] = Field(default=None)
    download_image: Optional[str] = Field(default=None)


class ReferenceGenomes(SQLModel, table=True):
    __tablename__ = 'reference_genomes'

    id: Optional[int] = Field(default=None, primary_key=True)
    ncbi_tax_id: int = Field(unique=True)
    name: str = Field(unique=True)
    accession: str
    aliases: Optional[str] = Field(default=None, description='物种别名, 多个别名用|分隔')
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
    upload_time: datetime = Field(default_factory=utc_now)
