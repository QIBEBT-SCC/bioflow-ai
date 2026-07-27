import uuid
from datetime import UTC, datetime

from sqlalchemy.orm import Session

from app.crud import project_file_mapping as mapping_crud
from app.models.project import ProjectFileMapping
from app.services._transaction import committing


class ProjectFileMappingError(ValueError):
    """Base error for project file-mapping operations."""


class ProjectFileMappingNotFoundError(ProjectFileMappingError):
    """Raised when a requested mapping does not exist."""


class DynamicProjectMappingConflictError(ProjectFileMappingError):
    """Raised when a dynamic mark conflicts with an existing mapping."""


def create_mapping(
    session: Session,
    project_id: int,
    keyword: str,
    file_path: str,
    description: str = '',
) -> ProjectFileMapping:
    """Create and commit a project file mapping."""
    mapping = ProjectFileMapping(
        project_id=project_id,
        keyword=keyword,
        file_path=file_path,
        description=description,
    )
    with committing(session):
        mapping_crud.insert(session, mapping)
    return mapping


def update_mapping(
    session: Session,
    mapping_id: int,
    *,
    file_path: str | None = None,
    description: str | None = None,
) -> ProjectFileMapping:
    """Update and commit a project file mapping."""
    mapping = mapping_crud.get_by_id(session, mapping_id)
    if mapping is None:
        raise ProjectFileMappingNotFoundError
    mapping.update_time = datetime.now(UTC)
    with committing(session):
        updated = mapping_crud.update(
            session,
            mapping_id,
            file_path=file_path,
            description=description,
        )
    if updated is None:
        raise ProjectFileMappingNotFoundError
    return updated


def delete_mapping(session: Session, mapping_id: int) -> None:
    """Delete and commit a project file mapping."""
    with committing(session):
        deleted = mapping_crud.delete(session, mapping_id)
    if not deleted:
        raise ProjectFileMappingNotFoundError


def upsert_dynamic_mapping(
    session: Session,
    *,
    project_id: int,
    workflow_uid: uuid.UUID,
    run_uid: uuid.UUID,
    keyword: str,
    file_path: str,
    description: str,
) -> ProjectFileMapping:
    """Create or replace one workflow-owned dynamic project mapping."""
    mapping = mapping_crud.get_by_keyword(session, project_id, keyword)
    if mapping is not None and not mapping.is_dynamic:
        msg = f'Dynamic mark "{keyword}" conflicts with initial project resource "proj:{keyword}".'
        raise DynamicProjectMappingConflictError(msg)
    if mapping is not None and mapping.producer_workflow_uid != workflow_uid:
        msg = f'Dynamic project mark "{keyword}" is already owned by workflow {mapping.producer_workflow_uid}.'
        raise DynamicProjectMappingConflictError(msg)

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
        mapping.update_time = datetime.now(UTC)

    with committing(session):
        mapping_crud.insert(session, mapping)
    return mapping
