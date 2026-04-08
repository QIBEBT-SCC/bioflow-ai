import uuid
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from app.models.resource import SampleFileType


class IndexStatus(str, Enum):
    """Status of a single alignment index."""

    ready = 'ready'
    not_built = 'not_built'
    building = 'building'


class GenomeIndexStatus(BaseModel):
    """Per-index build status for a reference genome."""

    bowtie2: IndexStatus = IndexStatus.not_built
    bwa: IndexStatus = IndexStatus.not_built
    hisat2: IndexStatus = IndexStatus.not_built
    star: IndexStatus = IndexStatus.not_built
    minimap2: IndexStatus = IndexStatus.not_built


class BioDbCreate(BaseModel):
    name: str
    description: Optional[str] = None
    path: str
    last_update: Optional[str] = None


class BioDbSimple(BaseModel):
    id: int
    name: str


class GenomeUpdate(BaseModel):
    id: int
    aliases: Optional[str] = None

    bowtie2_index: Optional[str] = None
    bwa_index: Optional[str] = None
    hisat2_index: Optional[str] = None
    star_index: Optional[str] = None
    minimap2_index: Optional[str] = None


class ReferenceGenomeListItem(BaseModel):
    """Reference genome list item (lightweight)."""

    model_config = {'from_attributes': True}

    id: int
    ncbi_tax_id: int
    name: str
    accession: str
    aliases: Optional[str] = None
    index_status: GenomeIndexStatus = Field(default_factory=GenomeIndexStatus)


class ReferenceGenomePublic(BaseModel):
    """Full details of a reference genome, including per-index build status."""

    model_config = {'from_attributes': True}

    id: int
    ncbi_tax_id: int
    name: str
    accession: str
    aliases: Optional[str] = None
    genome_sequences: str
    annotation_gff: str
    annotation_gtf: str
    bowtie2_index: Optional[str] = None
    bwa_index: Optional[str] = None
    hisat2_index: Optional[str] = None
    star_index: Optional[str] = None
    minimap2_index: Optional[str] = None
    index_status: GenomeIndexStatus = Field(default_factory=GenomeIndexStatus)


class ReferenceGenomeDownloadRequest(BaseModel):
    """触发后台下载新参考基因组的请求体, ncbi_tax_id 和 species_name 二选一."""

    ncbi_tax_id: Optional[int] = None
    species_name: Optional[str] = None
    required_index: list[str] = Field(
        default_factory=list, description='需要构建的索引类型, 如 ["bowtie2_index", "hisat2_index"]'
    )


class ReferenceGenomeBuildIndexRequest(BaseModel):
    """对已有参考基因组补充构建索引的请求体."""

    required_index: list[str] = Field(description='需要构建的索引类型, 如 ["star_index"]')


class CeleryTaskResponse(BaseModel):
    """Response after submitting a Celery background task."""

    task_id: str
    message: str


class GenomeDownloadResponse(BaseModel):
    """Response for a genome download request, including NCBI metadata found before dispatching."""

    task_id: str
    message: str
    # NCBI metadata resolved before the download task was submitted
    ncbi_accession: str
    ncbi_tax_id: int
    species_name: str


# Sample Schemas
class SampleFileCreate(BaseModel):
    """创建样本文件的请求模型.

    前端只需传递 data_type, tag 和 file_path
    file_format、file_size 和 md5_checksum 由后端自动计算
    """

    data_type: SampleFileType
    file_path: str
    tag: Optional[str] = None


class SampleFilePublic(BaseModel):
    """样本文件的公开信息."""

    model_config = {'from_attributes': True}

    uid: uuid.UUID
    sample_uid: uuid.UUID
    data_type: SampleFileType
    file_path: str
    file_format: str
    file_size: int
    md5_checksum: str
    tag: Optional[str] = None
    uploaded_time: datetime


class SampleCreate(BaseModel):
    """创建样本的请求模型."""

    project_id: int
    sample_name: str
    meta_data: Optional[dict] = None
    files: Optional[list[SampleFileCreate]] = None


class SampleUpdate(BaseModel):
    """更新样本的请求模型."""

    sample_name: Optional[str] = None
    meta_data: Optional[dict] = None


class SamplePublic(BaseModel):
    """样本的公开信息."""

    model_config = {'from_attributes': True, 'populate_by_name': True}

    uid: uuid.UUID
    owner_id: int
    project_id: int
    sample_name: str
    meta_data: dict
    create_time: datetime
    # 使用别名, 让 API 响应中显示为 "files", 但从模型中读取 "sample_files"
    sample_files: list[SampleFilePublic] = Field(default=[], alias='files', serialization_alias='files')


class SampleListItem(BaseModel):
    """样本列表项."""

    uid: uuid.UUID
    project_id: int
    sample_name: str
    meta_data: dict
    create_time: datetime
    file_count: int = 0


class SampleBatchImportResult(BaseModel):
    """批量导入结果."""

    success: int  # 成功导入的样本数
    failed: int  # 失败的样本数
    errors: list[dict] = []  # 错误详情: [{"row": 1, "sample_name": "xxx", "error": "xxx"}]
    samples: list[uuid.UUID] = []  # 成功创建的样本 UUID 列表
