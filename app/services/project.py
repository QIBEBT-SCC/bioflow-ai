import shutil
from pathlib import Path

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.crud import project as project_crud
from app.crud import projetc_tag as tag_crud
from app.models.project import Project, ProjectTag
from app.schemas.project import ProjectCreateProp, ProjectUpdate, TagProp
from app.services._transaction import committing

STORAGE_PATH = get_settings().persistent.storage.STORAGE_PATH


class ProjectServiceError(ValueError):
    """Base error for project application operations."""


class ProjectNotFoundError(ProjectServiceError):
    """Raised when a requested project does not exist."""


class ProjectPermissionError(ProjectServiceError):
    """Raised when a user cannot mutate a project."""


def create_tag(session: Session, data: TagProp) -> ProjectTag:
    """Create and commit a project tag."""
    tag = ProjectTag(name=data.name, color=data.color)
    with committing(session):
        tag_crud.insert(session, tag)
    return tag


def create_project(session: Session, data: ProjectCreateProp, owner_id: int) -> Project:
    """Create a project, commit it, then create its storage directory."""
    tags = list(tag_crud.get_all_by_id(session, data.tag_ids))
    project = Project(
        owner_id=owner_id,
        name=data.name,
        description=data.description,
        public=data.public,
        tags=tags,
    )
    with committing(session):
        project_crud.insert(session, project)

    project_path = Path(STORAGE_PATH) / 'proj' / str(project.id)
    project_path.mkdir(parents=True, exist_ok=True)
    return project


def update_project(
    session: Session,
    project_id: int,
    data: ProjectUpdate,
    *,
    current_user_id: int,
) -> Project:
    """Authorize, update, and commit project metadata."""
    project = session.get(Project, project_id)
    if project is None:
        raise ProjectNotFoundError
    if project.owner_id != current_user_id:
        raise ProjectPermissionError
    with committing(session):
        updated = project_crud.update(session, project_id, data)
    if updated is None:
        raise ProjectNotFoundError
    return updated


def delete_project(session: Session, project_id: int) -> None:
    """Delete a project, commit it, then remove its storage directory."""
    with committing(session):
        project_crud.delete(session, project_id)
    project_path = Path(STORAGE_PATH) / 'proj' / str(project_id)
    shutil.rmtree(project_path)


def star_project(session: Session, user_id: int, project_id: int) -> None:
    """Commit an idempotent project star operation."""
    with committing(session):
        project_crud.star_project(session, user_id, project_id)


def unstar_project(session: Session, user_id: int, project_id: int) -> None:
    """Commit an idempotent project unstar operation."""
    with committing(session):
        project_crud.unstar_project(session, user_id, project_id)
