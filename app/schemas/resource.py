import uuid
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from app.models.resource import SampleFileType


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


# Sample Schemas
class SampleFileCreate(BaseModel):
    """创建样本文件的请求模型
    
    前端只需传递 data_type 和 file_path，
    file_format、file_size 和 md5_checksum 由后端自动计算
    """

    data_type: SampleFileType
    file_path: str


class SampleFilePublic(BaseModel):
    """样本文件的公开信息"""

    model_config = {'from_attributes': True}

    uid: uuid.UUID
    sample_uid: uuid.UUID
    data_type: SampleFileType
    file_path: str
    file_format: str
    file_size: int
    md5_checksum: str
    uploaded_time: datetime


class SampleCreate(BaseModel):
    """创建样本的请求模型"""

    project_id: int
    sample_name: str
    meta_data: Optional[dict] = None
    files: Optional[list[SampleFileCreate]] = None


class SampleUpdate(BaseModel):
    """更新样本的请求模型"""

    sample_name: Optional[str] = None
    meta_data: Optional[dict] = None


class SamplePublic(BaseModel):
    """样本的公开信息"""

    model_config = {'from_attributes': True, 'populate_by_name': True}

    uid: uuid.UUID
    owner_id: int
    project_id: int
    sample_name: str
    meta_data: dict
    create_time: datetime
    # 使用别名，让 API 响应中显示为 "files"，但从模型中读取 "sample_files"
    sample_files: list[SampleFilePublic] = Field(default=[], alias='files', serialization_alias='files')


class SampleListItem(BaseModel):
    """样本列表项"""

    uid: uuid.UUID
    project_id: int
    sample_name: str
    meta_data: dict
    create_time: datetime
    file_count: int = 0


class SampleBatchImportResult(BaseModel):
    """批量导入结果"""

    success: int  # 成功导入的样本数
    failed: int  # 失败的样本数
    errors: list[dict] = []  # 错误详情：[{"row": 1, "sample_name": "xxx", "error": "xxx"}]
    samples: list[uuid.UUID] = []  # 成功创建的样本 UUID 列表
