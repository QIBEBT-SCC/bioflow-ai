from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field
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
    name: str | None = None
    description: str | None = None
    starred: bool | None = None
    update_time: datetime | None = None
    tags: list[ProjectTag] | None = None


class ProjectFileMappingCreate(SQLModel):
    keyword: str = Field(max_length=50)
    file_path: str = Field(max_length=500)
    description: str = Field(default='')


class ProjectFileMappingUpdate(SQLModel):
    file_path: str | None = None
    description: str | None = None


class ProjectFileMappingPublic(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    project_id: int
    keyword: str
    file_path: str
    description: str
    create_time: datetime
    update_time: datetime
