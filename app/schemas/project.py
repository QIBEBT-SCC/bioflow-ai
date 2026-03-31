from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field
from sqlmodel import SQLModel

from app.models.project import ProjectTag
from app.schemas.user import UserPublic


class ProjectPublic(BaseModel):
    id: int
    name: str
    description: str
    starred: bool
    public: bool
    create_time: datetime
    update_time: datetime

    owner: UserPublic
    tags: list[ProjectTag]
    file_mappings: list['ProjectFileMappingPublic'] = Field(default_factory=list)


class TagWithCount(BaseModel):
    id: int
    name: str
    color: str
    project_count: int


class ProjectCreateProp(BaseModel):
    name: str
    description: str = ''
    public: bool = True
    tag_ids: list[int] = Field(default_factory=list)


class TagProp(BaseModel):
    name: str
    color: str


class ProjectUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    starred: Optional[bool] = None
    update_time: Optional[datetime] = None
    tags: Optional[list[ProjectTag]] = None


class ProjectFileMappingCreate(SQLModel):
    """创建项目文件映射的请求模型"""

    keyword: str = Field(max_length=50)
    file_path: str = Field(max_length=500)
    description: str = Field(default='')


class ProjectFileMappingUpdate(SQLModel):
    """更新项目文件映射的请求模型"""

    file_path: Optional[str] = None
    description: Optional[str] = None


class ProjectFileMappingPublic(BaseModel):
    """项目文件映射的公开信息"""

    model_config = {'from_attributes': True}

    id: int
    project_id: int
    keyword: str
    file_path: str
    description: str
    create_time: datetime
    update_time: datetime
