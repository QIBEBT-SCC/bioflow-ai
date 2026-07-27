import uuid
from datetime import datetime

from sqlalchemy import BigInteger, ForeignKey, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

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


class BioDb(Base):
    __tablename__ = 'bio_dbs'

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(unique=True)
    description: Mapped[str | None] = mapped_column(default=None)
    path: Mapped[str | None] = mapped_column(default=None)
    size: Mapped[str | None] = mapped_column(default=None)
    last_update: Mapped[str | None] = mapped_column(default=None)
    download_command: Mapped[str | None] = mapped_column(default=None)
    download_image: Mapped[str | None] = mapped_column(default=None)


class ReferenceGenomes(Base):
    __tablename__ = 'reference_genomes'

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    ncbi_tax_id: Mapped[int] = mapped_column(unique=True)
    name: Mapped[str] = mapped_column(unique=True)
    accession: Mapped[str]
    aliases: Mapped[str | None] = mapped_column(default=None)
    genome_sequences: Mapped[str]
    annotation_gff: Mapped[str]
    annotation_gtf: Mapped[str]

    bowtie2_index: Mapped[str | None] = mapped_column(default=None)
    bwa_index: Mapped[str | None] = mapped_column(default=None)
    hisat2_index: Mapped[str | None] = mapped_column(default=None)
    star_index: Mapped[str | None] = mapped_column(default=None)
    minimap2_index: Mapped[str | None] = mapped_column(default=None)


class File(Base):
    __tablename__ = 'files'

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    sha256: Mapped[str] = mapped_column(unique=True)
    file_name: Mapped[str]
    s3_path: Mapped[str]
    upload_time: Mapped[datetime] = mapped_column(default=utc_now)
