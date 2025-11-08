from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

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
