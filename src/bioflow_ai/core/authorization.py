from fastapi import HTTPException, Request
from starlette import status

from bioflow_ai.core.auth import UserDep
from bioflow_ai.core.session import SessionDep
from bioflow_ai.crud import workflow as workflow_crud
from bioflow_ai.models.project import Project
from bioflow_ai.schemas.user import CurrentUser
from bioflow_ai.schemas.user_config import UserRole

PROJECT_RESOURCE = 'Project'
WORKFLOW_RESOURCE = 'Workflow'


def _not_found(resource: str) -> HTTPException:
    return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f'{resource} not found')


def _is_admin(user: CurrentUser) -> bool:
    return user.role == UserRole.ADMIN


def _require_member(user: CurrentUser) -> None:
    if user.role < UserRole.MEMBER:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail='Member permission required')


def _require_owner(owner_id: int, user: CurrentUser, resource: str) -> None:
    if owner_id != user.id and not _is_admin(user):
        raise _not_found(resource)


async def authorize_project_request(
    request: Request,
    session: SessionDep,
    current_user: UserDep,
    project_id: int | None = None,
) -> None:
    """Enforce project visibility and ownership for every nested project route."""
    if project_id is None:
        return
    project = session.get(Project, project_id)
    if project is None:
        raise _not_found(PROJECT_RESOURCE)

    path = request.url.path
    is_safe = request.method in {'GET', 'HEAD', 'OPTIONS'}
    sample_suffix = path.partition('/samples')[2]
    private_sample_content = '/samples' in path and sample_suffix.rstrip('/') not in {'', '/count'}
    private_content = private_sample_content or any(
        segment in path for segment in ('/runs', '/agent-files', '/file-mappings', '/package')
    )
    personal_action = path.endswith(('/star', '/unstar'))

    if (is_safe and not private_content) or personal_action:
        owns_visible_resource = current_user.role >= UserRole.MEMBER and project.owner_id == current_user.id
        if not (project.public or owns_visible_resource or _is_admin(current_user)):
            raise _not_found(PROJECT_RESOURCE)
        return

    _require_member(current_user)
    _require_owner(project.owner_id, current_user, PROJECT_RESOURCE)


async def authorize_workflow_request(
    request: Request,
    session: SessionDep,
    current_user: UserDep,
    uid: str | None = None,
) -> None:
    """Allow public workflow reads and owner/admin writes."""
    if uid is None:
        return
    workflow = workflow_crud.get(session, uid)
    if workflow is None:
        raise _not_found(WORKFLOW_RESOURCE)
    if request.method in {'GET', 'HEAD', 'OPTIONS'}:
        owns_visible_resource = current_user.role >= UserRole.MEMBER and workflow.owner_id == current_user.id
        if not (workflow.public or owns_visible_resource or _is_admin(current_user)):
            raise _not_found(WORKFLOW_RESOURCE)
        return
    _require_member(current_user)
    _require_owner(workflow.owner_id, current_user, WORKFLOW_RESOURCE)
