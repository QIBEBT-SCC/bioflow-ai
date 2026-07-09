from datetime import UTC, datetime

from fastapi import HTTPException
from sqlmodel import Session, select
from starlette.status import HTTP_404_NOT_FOUND

from app.models.project import ProjectFileMapping


def create(
    session: Session, project_id: int, keyword: str, file_path: str, description: str = ''
) -> ProjectFileMapping:
    """Create a project file mapping."""
    mapping = ProjectFileMapping(project_id=project_id, keyword=keyword, file_path=file_path, description=description)
    session.add(mapping)
    session.commit()
    session.refresh(mapping)
    return mapping


def get_by_keyword(session: Session, project_id: int, keyword: str) -> ProjectFileMapping | None:
    """Return the file mapping for a given project and keyword."""
    statement = select(ProjectFileMapping).where(
        ProjectFileMapping.project_id == project_id, ProjectFileMapping.keyword == keyword
    )
    return session.exec(statement).one_or_none()


def get_all_by_project(session: Session, project_id: int) -> list[ProjectFileMapping]:
    """Return all file mappings for a project."""
    statement = select(ProjectFileMapping).where(ProjectFileMapping.project_id == project_id)
    return list(session.exec(statement).all())


def update(
    session: Session, mapping_id: int, file_path: str | None = None, description: str | None = None
) -> ProjectFileMapping:
    """Update a file mapping's path or description."""
    statement = select(ProjectFileMapping).where(ProjectFileMapping.id == mapping_id)
    mapping = session.exec(statement).one_or_none()

    if not mapping:
        raise HTTPException(HTTP_404_NOT_FOUND, 'File mapping not found')

    if file_path is not None:
        mapping.file_path = file_path
    if description is not None:
        mapping.description = description

    mapping.update_time = datetime.now(UTC)
    session.add(mapping)
    session.commit()
    session.refresh(mapping)
    return mapping


def delete(session: Session, mapping_id: int) -> None:
    """Delete a file mapping by ID."""
    statement = select(ProjectFileMapping).where(ProjectFileMapping.id == mapping_id)
    mapping = session.exec(statement).one_or_none()

    if not mapping:
        raise HTTPException(HTTP_404_NOT_FOUND, 'File mapping not found')

    session.delete(mapping)
    session.commit()
