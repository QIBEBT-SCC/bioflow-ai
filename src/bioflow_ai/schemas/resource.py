import uuid
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class IndexStatus(StrEnum):
    ready = 'ready'
    not_built = 'not_built'
    building = 'building'


class GenomeIndexStatus(BaseModel):
    bowtie2: IndexStatus = IndexStatus.not_built
    bwa: IndexStatus = IndexStatus.not_built
    hisat2: IndexStatus = IndexStatus.not_built
    star: IndexStatus = IndexStatus.not_built
    minimap2: IndexStatus = IndexStatus.not_built


class BioDbCreate(BaseModel):
    name: str
    description: str | None = None
    path: str | None = None
    last_update: str | None = None
    download_command: str | None = None
    download_image: str | None = None


class BioDbUpdate(BaseModel):
    """Partial update of a biological database entry; omitted fields stay unchanged."""

    name: str | None = Field(default=None, min_length=1)
    description: str | None = None
    path: str | None = None
    last_update: str | None = None
    download_command: str | None = None
    download_image: str | None = None


class BioDbSimple(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    path: str | None = None
    size: str | None = None


class BioDbPublic(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    description: str | None = None
    path: str | None = None
    size: str | None = None
    last_update: str | None = None
    download_command: str | None = None
    download_image: str | None = None


class BioDbDownloadResponse(BaseModel):
    task_id: str
    message: str


class BioDbDownloadStatus(StrEnum):
    not_downloaded = 'not_downloaded'
    downloading = 'downloading'
    ready = 'ready'


class BioDbDownloadStatusResponse(BaseModel):
    status: BioDbDownloadStatus
    message: str | None = None


class GenomeUpdate(BaseModel):
    id: int
    aliases: str | None = None

    bowtie2_index: str | None = None
    bwa_index: str | None = None
    hisat2_index: str | None = None
    star_index: str | None = None
    minimap2_index: str | None = None


class ReferenceGenomeListItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    ncbi_tax_id: int
    name: str
    accession: str
    aliases: str | None = None
    index_status: GenomeIndexStatus = Field(default_factory=GenomeIndexStatus)


class ReferenceGenomePublic(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    ncbi_tax_id: int
    name: str
    accession: str
    aliases: str | None = None
    genome_sequences: str
    annotation_gff: str
    annotation_gtf: str
    bowtie2_index: str | None = None
    bwa_index: str | None = None
    hisat2_index: str | None = None
    star_index: str | None = None
    minimap2_index: str | None = None
    index_status: GenomeIndexStatus = Field(default_factory=GenomeIndexStatus)


class ReferenceGenomeDownloadRequest(BaseModel):
    ncbi_tax_id: int | None = None
    species_name: str | None = None
    required_index: list[str] = Field(
        default_factory=list, description='需要构建的索引类型, 如 ["bowtie2_index", "hisat2_index"]'
    )


class ReferenceGenomeBuildIndexRequest(BaseModel):
    required_index: list[str] = Field(description='需要构建的索引类型, 如 ["star_index"]')


class CeleryTaskResponse(BaseModel):
    task_id: str
    message: str


class GenomeDownloadResponse(BaseModel):
    task_id: str
    message: str
    # NCBI metadata resolved before the download task was submitted
    ncbi_accession: str
    ncbi_tax_id: int
    species_name: str


# Sample Schemas
class SampleFileCreate(BaseModel):
    file_path: str
    tag: str | None = None


class SampleFilePublic(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    uid: uuid.UUID
    sample_uid: uuid.UUID
    file_path: str
    file_format: str | None
    file_size: int
    md5_checksum: str | None
    tag: str | None = None
    description: str
    is_dynamic: bool
    producer_workflow_uid: uuid.UUID | None
    producer_run_uid: uuid.UUID | None
    uploaded_time: datetime


class SampleCreate(BaseModel):
    project_id: int
    sample_name: str
    meta_data: dict | None = None


class SampleUpdate(BaseModel):
    sample_name: str | None = None
    meta_data: dict | None = None


class SamplePublic(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    uid: uuid.UUID
    owner_id: int
    project_id: int
    sample_name: str
    meta_data: dict
    create_time: datetime
    # 使用别名, 让 API 响应中显示为 "files", 但从模型中读取 "sample_files"
    sample_files: list[SampleFilePublic] = Field(default_factory=list, alias='files', serialization_alias='files')


class SampleListItem(BaseModel):
    uid: uuid.UUID
    project_id: int
    sample_name: str
    meta_data: dict
    create_time: datetime
    file_count: int = 0


class SampleGroupSummary(BaseModel):
    count: int
    meta_data: dict
    tags: list[str]
    sample_name_examples: list[str]


class SampleBatchImportResult(BaseModel):
    success: int  # 成功导入的样本数
    failed: int  # 失败的样本数
    errors: list[dict] = Field(default_factory=list)  # 错误详情: [{"row": 1, "sample_name": "xxx", "error": "xxx"}]
    samples: list[uuid.UUID] = Field(default_factory=list)  # 成功创建的样本 UUID 列表
