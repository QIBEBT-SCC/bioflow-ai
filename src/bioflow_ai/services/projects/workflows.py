"""Project workflow orchestration services."""

import uuid

from sqlalchemy import delete
from sqlalchemy.orm import Session

from bioflow_ai.crud import project_workflow as project_workflow_crud
from bioflow_ai.crud import workflow as wf_crud
from bioflow_ai.models.workflow import ProjectWorkflow, WorkflowRun
from bioflow_ai.schemas.project_workflow import ProjectWorkflowPublic
from bioflow_ai.services._transaction import committing

_WORKFLOW_NOT_FOUND = 'Workflow not found'
_WORKFLOW_CONFLICT = 'Workflow already exists in this project'
_PROJECT_WORKFLOW_NOT_FOUND = 'Workflow not found in this project'


class ProjectWorkflowError(Exception):
    """Base exception for project-workflow application services."""


class ProjectWorkflowNotFoundError(ProjectWorkflowError):
    """A requested workflow or project association does not exist."""


class ProjectWorkflowConflictError(ProjectWorkflowError):
    """A workflow is already associated with the project."""


def add_workflow_to_project(
    session: Session,
    project_id: int,
    workflow_uid: uuid.UUID,
) -> ProjectWorkflowPublic:
    """Associate an existing workflow template with a project."""
    workflow = wf_crud.get(session, workflow_uid)
    if workflow is None:
        raise ProjectWorkflowNotFoundError(_WORKFLOW_NOT_FOUND)
    if project_workflow_crud.exists(session, project_id, workflow_uid):
        raise ProjectWorkflowConflictError(_WORKFLOW_CONFLICT)

    with committing(session):
        association = project_workflow_crud.insert(
            session,
            ProjectWorkflow(project_id=project_id, workflow_uid=workflow_uid),
        )

    return ProjectWorkflowPublic(
        workflow_uid=workflow.uid,
        workflow_name=workflow.name,
        execution_scope=workflow.execution_scope,
        import_time=association.import_time,
        enabled=association.enabled,
    )


def list_project_workflows(
    session: Session,
    project_id: int,
    limit: int = 20,
    offset: int = 0,
) -> list[ProjectWorkflowPublic]:
    """Return workflow templates imported into a project."""
    return [
        ProjectWorkflowPublic(
            workflow_uid=association.workflow_uid,
            workflow_name=workflow.name,
            execution_scope=workflow.execution_scope,
            import_time=association.import_time,
            enabled=association.enabled,
        )
        for association, workflow in project_workflow_crud.list_with_workflows(
            session,
            project_id,
            limit,
            offset,
        )
    ]


def remove_workflow_from_project(session: Session, project_id: int, workflow_uid: uuid.UUID) -> None:
    """Remove a workflow from a project and clean up its run data."""
    if not project_workflow_crud.exists(session, project_id, workflow_uid):
        raise ProjectWorkflowNotFoundError(_PROJECT_WORKFLOW_NOT_FOUND)

    with committing(session):
        _delete_workflow_runs(session, project_id, workflow_uid)
        project_workflow_crud.delete(session, project_id, workflow_uid)


def _delete_workflow_runs(session: Session, project_id: int, workflow_uid: uuid.UUID) -> None:
    """Delete workflow runs; database cascades remove nodes, dependencies, and outbox rows."""
    session.execute(
        delete(WorkflowRun).where(
            WorkflowRun.project_id == project_id,
            WorkflowRun.workflow_uid == workflow_uid,
        )
    )
