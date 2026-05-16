from collections.abc import Sequence
from typing import TYPE_CHECKING

from sqlmodel import Session, col, select

from app.models.project import ProjectTag
from app.schemas.project import TagWithCount

if TYPE_CHECKING:
    from sqlalchemy import Select


def insert(session: Session, tag: ProjectTag) -> ProjectTag:
    """Insert a new project tag into the database."""
    session.add(tag)
    session.commit()
    session.refresh(tag)

    return tag


def get_all(session: Session) -> Sequence[TagWithCount]:
    """Return all project tags with their project counts."""
    statement = select(ProjectTag, ProjectTag.project_count)
    tags = session.exec(statement).all()
    return [TagWithCount(id=t.id, name=t.name, color=t.color, project_count=c) for t, c in tags]


def get_all_by_id(session: Session, ids: list[int]) -> Sequence[ProjectTag]:
    """Return all project tags matching the given list of ids."""
    statement: Select = select(ProjectTag).where(col(ProjectTag.id).in_(ids))
    return session.exec(statement).all()
