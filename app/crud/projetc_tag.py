from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.project import ProjectTag
from app.schemas.project import TagWithCount


def insert(session: Session, tag: ProjectTag) -> ProjectTag:
    """Stage a new project tag."""
    session.add(tag)
    session.flush()
    session.refresh(tag)
    return tag


def get_all(session: Session) -> list[TagWithCount]:
    """Return all project tags with their project counts."""
    statement = select(ProjectTag, ProjectTag.project_count)
    return [
        TagWithCount(id=tag.id, name=tag.name, color=tag.color, project_count=count)
        for tag, count in session.execute(statement)
    ]


def get_all_by_id(session: Session, ids: list[int]) -> list[ProjectTag]:
    """Return all project tags matching the given list of IDs."""
    if not ids:
        return []
    return list(session.scalars(select(ProjectTag).where(ProjectTag.id.in_(ids))))
