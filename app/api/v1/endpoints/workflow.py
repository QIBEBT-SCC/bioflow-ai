from typing import Annotated

from fastapi import APIRouter, HTTPException, Query
from starlette import status
from uuid_extensions import uuid7

import app.crud.workflow as wf_crud
from app.core.auth import MemberDep, UserDep
from app.core.session import SessionDep
from app.models.workflow import Workflow
from app.schemas.common import Paginated
from app.schemas.react_flow import WorkflowDefinition
from app.schemas.user_config import UserRole
from app.schemas.workflow import WorkflowCreateProp, WorkflowPublic, WorkflowSimple, WorkflowUpdate
from app.schemas.workflow_config import WorkflowType
from app.services import workflow_catalog
from app.services.workflow.compiler import validate_workflow
from app.services.workflow.subgraphs import validate_interface_handles

router = APIRouter()


@router.get('')
async def get_workflows(
    session: SessionDep,
    current_user: UserDep,
    wf_type: WorkflowType | None = None,
    offset: int = 0,
    limit: Annotated[int, Query(le=20)] = 20,
) -> Paginated[WorkflowSimple]:
    """Get a paginated list of workflows for the current user."""
    workflows, total = wf_crud.get_page(
        session,
        limit,
        offset,
        current_user.id,
        wf_type=wf_type,
        include_private=current_user.role == UserRole.ADMIN,
        include_owned=current_user.role >= UserRole.MEMBER,
    )
    return Paginated(
        total=total,
        offset=offset,
        limit=limit,
        has_more=offset + limit < total,
        data=[WorkflowSimple.model_validate(wf) for wf in workflows],
    )


@router.get('/{uid}')
async def get_workflow(uid: str, session: SessionDep, _current_user: UserDep) -> WorkflowPublic:
    """Get a workflow by its UID."""
    workflow = wf_crud.get(session, uid)
    if workflow is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, 'Workflow not found')
    return WorkflowPublic.model_validate(workflow)


@router.post('')
async def save_workflow(
    workflow_prop: WorkflowCreateProp,
    session: SessionDep,
    current_user: MemberDep,
) -> str:
    """Save a new workflow template."""
    if wf_crud.workflow_name_exists(session, workflow_prop.name):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail='name already exists!')

    _validate_definition(workflow_prop.workflow, workflow_prop.wf_type, workflow_prop.description, session)

    uid = uuid7()
    workflow = Workflow(
        uid=uid,
        name=workflow_prop.name,
        description=workflow_prop.description,
        owner_id=current_user.id,
        wf_type=workflow_prop.wf_type,
        execution_scope=workflow_prop.execution_scope,
        workflow=workflow_prop.workflow,
        public=workflow_prop.public,
    )

    await workflow_catalog.create_workflow(session, workflow)
    return str(uid)


@router.delete('/{uid}', status_code=status.HTTP_204_NO_CONTENT)
async def delete_workflow(
    uid: str,
    session: SessionDep,
    _current_user: UserDep,
) -> None:
    """Delete a workflow by its UID."""
    try:
        await workflow_catalog.delete_workflow(session, uid)
    except workflow_catalog.WorkflowCatalogNotFoundError as error:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(error)) from error


@router.patch('/{uid}')
async def update_workflow(
    uid: str,
    data: WorkflowUpdate,
    session: SessionDep,
    _current_user: UserDep,
) -> str:
    """Update an existing workflow's name and/or definition."""
    existing = wf_crud.get(session, uid)
    if existing is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, 'Workflow not found')
    _validate_definition(
        data.workflow or existing.workflow,
        existing.wf_type,
        data.description if data.description is not None else existing.description,
        session,
    )
    try:
        await workflow_catalog.update_workflow(session, uid, data)
    except workflow_catalog.WorkflowCatalogNotFoundError as error:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(error)) from error
    return uid


def _validate_definition(
    workflow: WorkflowDefinition, wf_type: WorkflowType, description: str, session: SessionDep
) -> None:
    if wf_type == WorkflowType.SUBMODULE and (not description.strip() or workflow.interface is None):
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY, 'Subgraphs require a description and interface definition'
        )
    if workflow.interface is not None or any(node.type == 'subgraph' for node in workflow.nodes):
        try:
            validate_workflow(workflow)
            validate_interface_handles(workflow, session)
        except ValueError as error:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(error)) from error
