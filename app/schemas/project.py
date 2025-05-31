from datetime import datetime
from typing import Optional, Dict, Any

from pydantic import field_serializer, BaseModel, Field

from app.models.project import ProjectTag, Project


class ProjectPublic(BaseModel):
    id: int
    owner_name: str
    name: str
    description: str
    starred: bool
    create_time: datetime
    update_time: datetime
    tags: list[ProjectTag]

    @classmethod
    def from_orm(cls, project: Project, update: Optional[Dict[str, Any]] = None):
        kwargs = project.model_dump()
        kwargs['owner_name'] = project.owner.username if project.owner else ''
        kwargs['tags'] = project.tags
        if update:
            kwargs.update(update)
        return cls.model_validate(kwargs)

    @field_serializer('create_time', 'update_time')
    def serialize_datetime(self, value: datetime, _info):
        return value.strftime('%Y-%m-%d %H:%M') if value else ''


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
