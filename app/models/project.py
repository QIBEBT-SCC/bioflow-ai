from datetime import datetime, UTC
from typing import Optional

from sqlmodel import SQLModel, Field, Relationship


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
    owner: str
    name: str
    description: str = Field(default='')
    starred: bool = Field(default=False)
    public: bool = Field(default=True)
    create_time: datetime = Field(default_factory=lambda: datetime.now(UTC))
    update_time: datetime = Field(default_factory=lambda: datetime.now(UTC))

    tags: list['Tag'] = Relationship(back_populates='projects', link_model=ProjectTag)


class ProjectPublic(SQLModel):
    id: int
    owner: str
    name: str
    description: str
    starred: bool
    create_time: datetime
    update_time: datetime
    tags: list[Tag]
