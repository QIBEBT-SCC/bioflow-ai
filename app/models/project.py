from datetime import datetime, UTC
from typing import Optional, Any, Dict

from pydantic import BaseModel, field_serializer
from sqlalchemy import func
from sqlalchemy.ext.hybrid import hybrid_property
from sqlmodel import SQLModel, Field, Relationship, select

from app.models.user import User


class ProjectTagLink(SQLModel, table=True):
    __tablename__ = 'project_tag_link'

    project_id: int = Field(foreign_key='projects.id', primary_key=True)
    tag_id: int = Field(foreign_key='project_tags.id', primary_key=True)


class ProjectStar(SQLModel, table=True):
    __tablename__ = 'project_star'

    project_id: int = Field(foreign_key='projects.id', primary_key=True)
    user_id: int = Field(foreign_key='users.id')


class ProjectTag(SQLModel, table=True):
    __tablename__ = 'project_tags'
    model_config = {'ignored_types': (hybrid_property,)}

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    color: str

    projects: list['Project'] = Relationship(back_populates='tags', link_model=ProjectTagLink)

    @hybrid_property
    def project_count(self):
        return len(self.projects)

    @project_count.expression
    def project_count(self):
        return (
            select(func.count(ProjectTagLink.project_id))
            .where(ProjectTagLink.tag_id == self.id)
            .correlate_except(ProjectTagLink)
            .scalar_subquery()
        )


class TagWithCount(BaseModel):
    id: int
    name: str
    color: str
    project_count: int


class Project(SQLModel, table=True):
    __tablename__ = 'projects'

    id: int = Field(default=None, primary_key=True)
    owner_id: int = Field(foreign_key='users.id')
    name: str
    description: str = Field(default='')
    public: bool = Field(default=True)
    create_time: datetime = Field(default_factory=lambda: datetime.now(UTC))
    update_time: datetime = Field(default_factory=lambda: datetime.now(UTC))

    tags: list['ProjectTag'] = Relationship(back_populates='projects', link_model=ProjectTagLink)
    owner: Optional['User'] = Relationship()


class ProjectPublic(SQLModel):
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

class ProjectCreateProp(BaseModel):
    name: str
    description: str = ''
    starred: bool = False
    public: bool = True
    tag_ids: list[int] = []

class TagProp(BaseModel):
    name: str
    color: str

class ProjectUpdate(SQLModel):
    name: Optional[str] = None
    description: Optional[str] = None
    starred: Optional[bool] = None
    update_time: Optional[datetime] = None
    tags: Optional[list[ProjectTag]] = None
