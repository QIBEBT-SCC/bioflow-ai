import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, String, UniqueConstraint, func, select
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql.selectable import ScalarSelect

from bioflow_ai.models.base import Base
from bioflow_ai.models.user import User
from bioflow_ai.utils.system import utc_now

if TYPE_CHECKING:
    from bioflow_ai.models.resource import Sample


class ProjectTagLink(Base):
    __tablename__ = 'project_tag_link'

    project_id: Mapped[int] = mapped_column(ForeignKey('projects.id', ondelete='CASCADE'), primary_key=True)
    tag_id: Mapped[int] = mapped_column(ForeignKey('project_tags.id'), primary_key=True)


class ProjectStar(Base):
    __tablename__ = 'project_star'

    project_id: Mapped[int] = mapped_column(ForeignKey('projects.id', ondelete='CASCADE'), primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey(User.id))


class ProjectTag(Base):
    __tablename__ = 'project_tags'

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str]
    color: Mapped[str]

    projects: Mapped[list['Project']] = relationship(
        secondary=ProjectTagLink.__table__,
        back_populates='tags',
    )

    @hybrid_property
    def project_count(self) -> int:
        """Return the number of projects using this tag."""
        return len(self.projects)

    @project_count.inplace.expression
    @classmethod
    def _project_count_expression(cls) -> ScalarSelect[int]:
        """Return the SQL expression for counting projects using this tag."""
        return (
            select(func.count(ProjectTagLink.project_id))
            .where(ProjectTagLink.tag_id == cls.id)
            .correlate_except(ProjectTagLink)
            .scalar_subquery()
        )


class Project(Base):
    __tablename__ = 'projects'

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    owner_id: Mapped[int] = mapped_column(ForeignKey(User.id))
    name: Mapped[str]
    description: Mapped[str] = mapped_column(default='')
    public: Mapped[bool] = mapped_column(default=True)
    create_time: Mapped[datetime] = mapped_column(default=utc_now)
    update_time: Mapped[datetime] = mapped_column(default=utc_now)

    tags: Mapped[list[ProjectTag]] = relationship(
        secondary=ProjectTagLink.__table__,
        back_populates='projects',
    )
    owner: Mapped[User | None] = relationship(User)
    samples: Mapped[list['Sample']] = relationship(
        'Sample',
        cascade='all, delete-orphan',
        passive_deletes=True,
    )
    file_mappings: Mapped[list['ProjectFileMapping']] = relationship(
        back_populates='project',
        cascade='all, delete-orphan',
        passive_deletes=True,
    )


class ProjectFileMapping(Base):
    __tablename__ = 'project_file_mappings'
    __table_args__ = (UniqueConstraint('project_id', 'keyword', name='uq_project_file_mapping'),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    project_id: Mapped[int] = mapped_column(ForeignKey(Project.id, ondelete='CASCADE'))
    keyword: Mapped[str] = mapped_column(String(50), index=True)
    file_path: Mapped[str] = mapped_column(String(500))
    description: Mapped[str] = mapped_column(default='')
    is_dynamic: Mapped[bool] = mapped_column(default=False, index=True)
    producer_workflow_uid: Mapped[uuid.UUID | None] = mapped_column(default=None, index=True)
    producer_run_uid: Mapped[uuid.UUID | None] = mapped_column(default=None)
    create_time: Mapped[datetime] = mapped_column(default=utc_now)
    update_time: Mapped[datetime] = mapped_column(default=utc_now)

    project: Mapped[Project | None] = relationship(back_populates='file_mappings')
