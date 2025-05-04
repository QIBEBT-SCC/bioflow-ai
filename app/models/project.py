from datetime import datetime, UTC
from typing import Optional, Any, Dict

from sqlmodel import SQLModel, Field, Relationship

from app.models.user import User


class ProjectTag(SQLModel, table=True):
    __tablename__ = 'project_tag'

    project_id: int = Field(foreign_key='projects.id', primary_key=True)
    tag_id: int = Field(foreign_key='tags.id', primary_key=True)


class Tag(SQLModel, table=True):
    __tablename__ = 'tags'

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    color: str

    projects: list['Project'] = Relationship(back_populates='tags', link_model=ProjectTag)


class Project(SQLModel, table=True):
    __tablename__ = 'projects'

    id: int = Field(default=None, primary_key=True)
    owner_id: int = Field(foreign_key='users.id')
    name: str
    description: str = Field(default='')
    starred: bool = Field(default=False)
    public: bool = Field(default=True)
    create_time: datetime = Field(default_factory=lambda: datetime.now(UTC))
    update_time: datetime = Field(default_factory=lambda: datetime.now(UTC))

    tags: list['Tag'] = Relationship(back_populates='projects', link_model=ProjectTag)
    owner: Optional['User'] = Relationship()


class ProjectPublic(SQLModel):
    id: int
    owner_name: str
    name: str
    description: str
    starred: bool
    create_time: datetime
    update_time: datetime
    tags: list[Tag]

    @classmethod
    def from_orm(cls, project: Project, update: Optional[Dict[str, Any]] = None):
        kwargs = project.model_dump()
        kwargs['owner_name'] = project.owner.username if project.owner else ''
        kwargs['tags'] = project.tags
        return cls.model_validate(kwargs)
