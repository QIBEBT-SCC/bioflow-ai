import uuid

from sqlalchemy import delete as sa_delete
from sqlalchemy import select
from sqlalchemy.orm import Session

from bioflow_ai.models.project import ProjectFileMapping


def insert(session: Session, mapping: ProjectFileMapping) -> ProjectFileMapping:
    """Stage a new project file mapping."""
    session.add(mapping)
    session.flush()
    session.refresh(mapping)
    return mapping


def get_by_id(session: Session, mapping_id: int) -> ProjectFileMapping | None:
    """Return a project file mapping by primary key."""
    return session.get(ProjectFileMapping, mapping_id)


def get_by_keyword(session: Session, project_id: int, keyword: str) -> ProjectFileMapping | None:
    """Return the file mapping for a given project and keyword."""
    statement = select(ProjectFileMapping).where(
        ProjectFileMapping.project_id == project_id,
        ProjectFileMapping.keyword == keyword,
    )
    return session.scalar(statement)


def get_all_by_project(session: Session, project_id: int) -> list[ProjectFileMapping]:
    """Return all file mappings for a project."""
    statement = (
        select(ProjectFileMapping)
        .where(ProjectFileMapping.project_id == project_id)
        .order_by(ProjectFileMapping.is_dynamic, ProjectFileMapping.keyword)
    )
    return list(session.scalars(statement))


def delete_dynamic_for_workflow(
    session: Session,
    *,
    project_id: int,
    workflow_uid: uuid.UUID,
) -> None:
    """Stage deletion of dynamic project mappings owned by a workflow."""
    session.execute(
        sa_delete(ProjectFileMapping).where(
            ProjectFileMapping.project_id == project_id,
            ProjectFileMapping.is_dynamic.is_(True),
            ProjectFileMapping.producer_workflow_uid == workflow_uid,
        )
    )


def update(
    session: Session,
    mapping_id: int,
    *,
    file_path: str | None = None,
    description: str | None = None,
) -> ProjectFileMapping | None:
    """Stage changes to a file mapping."""
    mapping = get_by_id(session, mapping_id)
    if mapping is None:
        return None
    if file_path is not None:
        mapping.file_path = file_path
    if description is not None:
        mapping.description = description
    session.flush()
    session.refresh(mapping)
    return mapping


def delete(session: Session, mapping_id: int) -> bool:
    """Stage deletion of a file mapping."""
    mapping = get_by_id(session, mapping_id)
    if mapping is None:
        return False
    session.delete(mapping)
    session.flush()
    return True
