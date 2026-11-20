import uuid
from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from bioflow_ai.models.workflow import ProjectWorkflow, Workflow
from bioflow_ai.utils.identifiers import UUIDLike, as_uuid


def get(session: Session, project_id: int, workflow_uid: UUIDLike, *, lock: bool = False) -> ProjectWorkflow | None:
    """Return a project-workflow association, optionally locking it for the transaction."""
    statement = select(ProjectWorkflow).where(
        ProjectWorkflow.project_id == project_id,
        ProjectWorkflow.workflow_uid == as_uuid(workflow_uid),
    )
    return session.scalar(statement.with_for_update() if lock else statement)


def insert(session: Session, project_workflow: ProjectWorkflow) -> ProjectWorkflow:
    """Insert a project-workflow association."""
    session.add(project_workflow)
    session.flush()
    session.refresh(project_workflow)
    return project_workflow


def delete(session: Session, project_id: int, workflow_uid: UUIDLike) -> bool:
    """Delete a project-workflow association if present."""
    project_workflow = get(session, project_id, workflow_uid)
    if project_workflow is None:
        return False
    session.delete(project_workflow)
    session.flush()
    return True


def list_with_workflows(
    session: Session,
    project_id: int,
    limit: int = 20,
    offset: int = 0,
) -> Sequence[tuple[ProjectWorkflow, Workflow]]:
    """Return project-workflow associations with their workflow templates."""
    statement = (
        select(ProjectWorkflow, Workflow)
        .join(Workflow, ProjectWorkflow.workflow_uid == Workflow.uid)
        .where(ProjectWorkflow.project_id == project_id)
        .offset(offset)
        .limit(limit)
    )
    return [(association, workflow) for association, workflow in session.execute(statement)]


def exists(session: Session, project_id: int, workflow_uid: UUIDLike) -> bool:
    """Return whether a workflow template belongs to a project."""
    return get(session, project_id, workflow_uid) is not None


def set_enabled(
    session: Session,
    project_id: int,
    workflow_uid: UUIDLike,
    *,
    enabled: bool,
) -> ProjectWorkflow | None:
    """Set the enabled state of an existing project-workflow association."""
    project_workflow = get(session, project_id, workflow_uid)
    if project_workflow is None:
        return None
    project_workflow.enabled = enabled
    session.add(project_workflow)
    session.flush()
    session.refresh(project_workflow)
    return project_workflow


def workflow_uids_for_project(session: Session, project_id: int) -> Sequence[uuid.UUID]:
    """Return workflow UIDs imported into a project."""
    statement = select(ProjectWorkflow.workflow_uid).where(ProjectWorkflow.project_id == project_id)
    return session.scalars(statement).all()
