import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import UniqueConstraint, func
from sqlalchemy.ext.hybrid import hybrid_property
from sqlmodel import Field, Relationship, SQLModel, select

if TYPE_CHECKING:
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
    file_mappings: list['ProjectFileMapping'] = Relationship(back_populates='project', cascade_delete=True)
    workflows: list['ProjectWorkflow'] = Relationship(back_populates='project', cascade_delete=True)


class ProjectFileMapping(SQLModel, table=True):
    __tablename__ = 'project_file_mappings'
    __table_args__ = (UniqueConstraint('project_id', 'keyword', name='uq_project_file_mapping'),)

    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: int = Field(foreign_key='projects.id', ondelete='CASCADE')
    keyword: str = Field(max_length=50, index=True)
    file_path: str = Field(max_length=500)
    description: str = Field(default='')
    create_time: datetime = Field(default_factory=lambda: datetime.now(UTC))
    update_time: datetime = Field(default_factory=lambda: datetime.now(UTC))

    project: Optional['Project'] = Relationship(back_populates='file_mappings')


class ProjectWorkflow(SQLModel, table=True):
    """项目和工作流模板的关联表."""

    __tablename__ = 'project_workflows'

    project_id: int = Field(foreign_key='projects.id', primary_key=True, ondelete='CASCADE')
    workflow_uid: Optional[uuid.UUID] = Field(foreign_key='workflows.uid', ondelete='SET NULL')
    import_time: datetime = Field(default_factory=lambda: datetime.now(UTC))
    enabled: bool = Field(default=True, description='是否启用该工作流')

    project: Optional['Project'] = Relationship(back_populates='workflows')
