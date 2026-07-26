import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.user import UserPublic

ProjectSort = Literal['recent', 'nameAsc', 'nameDesc']


class ProjectTagPublic(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int | None = None
    name: str
    color: str


class ProjectPublic(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    description: str
    starred: bool
    public: bool
    create_time: datetime
    update_time: datetime

    owner: UserPublic
    tags: list[ProjectTagPublic]
    file_mappings: list['ProjectFileMappingPublic'] = Field(default_factory=list)


class ProjectCreateResult(BaseModel):
    """Project column projection returned by the create endpoint."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    owner_id: int
    name: str
    description: str = ''
    public: bool = True
    create_time: datetime
    update_time: datetime


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
    """Editable project metadata."""

    name: str | None = None
    description: str | None = None
    tags: list[ProjectTagPublic] | None = None


class ProjectFileMappingCreate(BaseModel):
    keyword: str = Field(max_length=50)
    file_path: str = Field(max_length=500)
    description: str = Field(default='')


class ProjectFileMappingUpdate(BaseModel):
    file_path: str | None = None
    description: str | None = None


class ProjectFileMappingPublic(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    project_id: int
    keyword: str
    file_path: str
    description: str
    is_dynamic: bool
    producer_workflow_uid: uuid.UUID | None
    producer_run_uid: uuid.UUID | None
    create_time: datetime
    update_time: datetime
