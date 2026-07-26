import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import BigInteger, ForeignKey, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlmodel import Field, SQLModel

from app.models.base import Base
from app.models.project import Project
from app.models.user import User
from app.utils.system import random_uid, utc_now


class Sample(Base):
    __tablename__ = 'samples'
    __table_args__ = (UniqueConstraint('project_id', 'sample_name', name='uq_sample_project_name'),)

    uid: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=random_uid)
    owner_id: Mapped[int] = mapped_column(ForeignKey(User.id))
    project_id: Mapped[int] = mapped_column(ForeignKey(Project.id, ondelete='CASCADE'))
    sample_name: Mapped[str]
    meta_data: Mapped[dict[str, str]] = mapped_column(JSONB, default=dict, nullable=True)
    create_time: Mapped[datetime] = mapped_column(default=utc_now)

    owner: Mapped[User | None] = relationship(User)
    sample_files: Mapped[list['SampleFiles']] = relationship(back_populates='sample', cascade='all, delete-orphan')


class SampleFiles(Base):
    __tablename__ = 'sample_files'
    __table_args__ = (UniqueConstraint('sample_uid', 'tag', name='uq_sample_file_tag'),)

    uid: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=random_uid)
    sample_uid: Mapped[uuid.UUID] = mapped_column(ForeignKey(Sample.uid, ondelete='CASCADE'))
    file_path: Mapped[str]
    file_format: Mapped[str | None] = mapped_column(default=None)
    file_size: Mapped[int] = mapped_column(BigInteger, default=0)
    md5_checksum: Mapped[str | None] = mapped_column(default=None)
    tag: Mapped[str] = mapped_column(String(50), index=True)
    description: Mapped[str] = mapped_column(default='')
    is_dynamic: Mapped[bool] = mapped_column(default=False, index=True)
    producer_workflow_uid: Mapped[uuid.UUID | None] = mapped_column(default=None, index=True)
    producer_run_uid: Mapped[uuid.UUID | None] = mapped_column(default=None)
    uploaded_time: Mapped[datetime] = mapped_column(default=utc_now)

    sample: Mapped[Sample | None] = relationship(back_populates='sample_files')


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
