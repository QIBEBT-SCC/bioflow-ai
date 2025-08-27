from datetime import datetime, UTC

from pydantic import BaseModel, Field, computed_field

from app.schemas.chat import ContentType


class ValidatedFileInfo(BaseModel):
    """文件验证后的信息模型"""

    filename: str = Field(description='原始文件名')
    file_extension: str = Field(description='文件扩展名（小写，包含.）')
    mime_type: str = Field(description='MIME类型')
    file_size: int = Field(description='文件大小（字节）')
    content_type: ContentType = Field(description='内容类型分类')

    @computed_field
    @property
    def file_size_mb(self) -> float:
        """文件大小（MB）"""
        return round(self.file_size / (1024 * 1024), 2)

    @computed_field
    @property
    def safe_filename(self) -> str:
        """安全的文件名（移除特殊字符）"""
        # 移除路径分隔符和特殊字符
        safe_name = self.filename.replace('/', '_').replace('\\', '_').replace('..', '_')
        return safe_name


class StoredFileInfo(BaseModel):
    """已存储文件的信息模型"""

    file_id: str = Field(description='文件唯一标识')
    file_path: str = Field(description='文件存储路径')
    original_info: ValidatedFileInfo = Field(description='原始文件信息')
    storage_time: datetime = Field(default_factory=lambda: datetime.now(UTC), description='存储时间')

    @computed_field
    @property
    def filename(self) -> str:
        """原始文件名"""
        return self.original_info.filename

    @computed_field
    @property
    def file_size(self) -> int:
        """文件大小"""
        return self.original_info.file_size

    @computed_field
    @property
    def mime_type(self) -> str:
        """MIME类型"""
        return self.original_info.mime_type

    @computed_field
    @property
    def content_type(self) -> ContentType:
        """内容类型"""
        return self.original_info.content_type

    def to_upload_response(self):
        """转换为文件上传响应格式"""
        # 延迟导入避免循环导入
        from app.schemas.chat import FileUploadResponse

        return FileUploadResponse(
            file_id=self.file_id,
            filename=self.filename,
            file_path=self.file_path,
            file_size=self.file_size,
            mime_type=self.mime_type,
            content_type=self.content_type,
            upload_time=self.storage_time,
        )
