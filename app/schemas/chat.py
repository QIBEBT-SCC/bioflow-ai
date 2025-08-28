from typing import Optional
from enum import Enum

from pydantic import BaseModel, Field, UUID7
from datetime import datetime, UTC


class ContentType(str, Enum):
    """消息内容类型枚举"""

    TEXT = 'text'
    IMAGE = 'image'
    FILE = 'file'
    DOCUMENT = 'document'


class FileUploadResponse(BaseModel):
    """文件上传响应模型"""

    file_id: str = Field(description='文件唯一标识')
    filename: str = Field(description='原始文件名')
    file_path: str = Field(description='文件存储路径')
    file_size: int = Field(description='文件大小（字节）')
    mime_type: str = Field(description='文件MIME类型')
    upload_time: datetime = Field(default_factory=lambda: datetime.now(UTC))
    content_type: ContentType = Field(description='内容类型')


class ChatSessionUpdate(BaseModel):
    uid: str
    description: Optional[str] = None
    update_time: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ChatSessionPublic(BaseModel):
    uid: UUID7
    description: str
    create_time: datetime
    update_time: datetime
