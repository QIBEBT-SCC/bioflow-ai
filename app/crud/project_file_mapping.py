import uuid
from datetime import UTC, datetime

from fastapi import HTTPException
from sqlalchemy import delete as sa_delete
from sqlmodel import Session, col, select
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
    statement = (
        select(ProjectFileMapping)
        .where(ProjectFileMapping.project_id == project_id)
        .order_by(col(ProjectFileMapping.is_dynamic), col(ProjectFileMapping.keyword))
    )
    return list(session.exec(statement).all())


def upsert_dynamic(
    session: Session,
    *,
    project_id: int,
    workflow_uid: uuid.UUID,
    run_uid: uuid.UUID,
    keyword: str,
    file_path: str,
    description: str,
) -> ProjectFileMapping:
    """Create or replace one dynamic project mapping after its mark node succeeds."""
    mapping = get_by_keyword(session, project_id, keyword)
    if mapping is not None and not mapping.is_dynamic:
        msg = f'Dynamic mark "{keyword}" conflicts with initial project resource "proj:{keyword}".'
        raise ValueError(msg)
    if mapping is not None and mapping.producer_workflow_uid != workflow_uid:
        msg = (
            f'Dynamic project mark "{keyword}" is already owned by workflow '
            f'{mapping.producer_workflow_uid}.'
        )
        raise ValueError(msg)

    now = datetime.now(UTC)
    if mapping is None:
        mapping = ProjectFileMapping(
            project_id=project_id,
            keyword=keyword,
            file_path=file_path,
            description=description,
            is_dynamic=True,
            producer_workflow_uid=workflow_uid,
            producer_run_uid=run_uid,
        )
    else:
        mapping.file_path = file_path
        mapping.description = description
        mapping.is_dynamic = True
        mapping.producer_workflow_uid = workflow_uid
        mapping.producer_run_uid = run_uid
        mapping.update_time = now

    session.add(mapping)
    session.commit()
    session.refresh(mapping)
    return mapping


def delete_dynamic_for_workflow(
    session: Session,
    *,
    project_id: int,
    workflow_uid: uuid.UUID,
) -> None:
    """Remove dynamic project mappings owned by a workflow before rerunning it."""
    session.exec(
        sa_delete(ProjectFileMapping).where(
            col(ProjectFileMapping.project_id) == project_id,
            col(ProjectFileMapping.is_dynamic).is_(True),
            col(ProjectFileMapping.producer_workflow_uid) == workflow_uid,
        )
    )


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
