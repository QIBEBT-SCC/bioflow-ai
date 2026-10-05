import logging
from typing import Annotated, cast

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy.orm import Session
from starlette import status
from uuid_extensions import uuid7

import bioflow_ai.crud.workflow as wf_crud
from bioflow_ai.core.auth import MemberDep, UserDep
from bioflow_ai.infrastructure.database import SessionDep
from bioflow_ai.infrastructure.redis.client import RedisDep
from bioflow_ai.models.project import Project
from bioflow_ai.models.workflow import Workflow
from bioflow_ai.schemas.common import Paginated
from bioflow_ai.schemas.user import CurrentUser
from bioflow_ai.schemas.user_config import UserRole
from bioflow_ai.schemas.workflow import (
    WorkflowCreateProp,
    WorkflowPublic,
    WorkflowSearchResult,
    WorkflowSimple,
    WorkflowUpdate,
)
from bioflow_ai.schemas.workflow_config import WorkflowType
from bioflow_ai.services.tools.cache import get_tool_definitions
from bioflow_ai.services.workflow import catalog as workflow_catalog
from bioflow_ai.services.workflow.validation import ValidationResult, WorkflowDocument, validate_workflow
from bioflow_ai.services.workflow.validation.resolution import (
    collect_tool_uids,
    mark_storage_conflicts,
    resolve_workflow,
)
from bioflow_ai.services.workflow.validation.types import (
    ExecutionScopeName,
    ValidationContext,
    WorkflowValidationRequest,
)

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post('/validate')
async def validate_graph(
    request: WorkflowValidationRequest,
    session: SessionDep,
    redis: RedisDep,
    current_user: UserDep,
) -> ValidationResult:
    """Diagnose a submitted graph without saving it or starting a run."""
    conflicts = {}
    if request.project_id is not None:
        project = session.get(Project, request.project_id)
        if project is None or not (
            project.public or project.owner_id == current_user.id or current_user.role == UserRole.ADMIN
        ):
            raise HTTPException(status.HTTP_404_NOT_FOUND, 'Project not found')
        conflicts = mark_storage_conflicts(session, request.project_id, request.graph)
    definitions = await get_tool_definitions(session, redis, collect_tool_uids(request.graph))
    return validate_workflow(
        WorkflowDocument(
            graph=request.graph,
            context=ValidationContext(execution_scope=request.execution_scope, mark_conflicts=conflicts),
            catalog=definitions,
        )
    )


def _get_workflow_for_user(session: Session, uid: str, user: CurrentUser, *, edit: bool = False) -> Workflow:
    """Apply the list visibility rule and restrict writes to owners or admins."""
    workflow = wf_crud.get(session, uid)
    is_admin = user.role == UserRole.ADMIN
    is_owner = user.role >= UserRole.MEMBER and workflow is not None and workflow.owner_id == user.id
    if workflow is None or not (is_admin or is_owner or (not edit and workflow.public)):
        raise HTTPException(status.HTTP_404_NOT_FOUND, 'Workflow not found')
    return workflow


@router.get('')
async def get_workflows(
    session: SessionDep,
    current_user: UserDep,
    wf_type: WorkflowType | None = None,
    offset: int = 0,
    limit: Annotated[int, Query(le=20)] = 20,
    q: Annotated[str, Query(max_length=512)] = '',
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
        q=q,
    )
    return Paginated(
        total=total,
        offset=offset,
        limit=limit,
        has_more=offset + limit < total,
        data=[WorkflowSimple.model_validate(wf) for wf in workflows],
    )


@router.get('/search')
async def search_subgraphs(
    session: SessionDep,
    current_user: UserDep,
    q: Annotated[str, Query(min_length=1, max_length=512)],
    limit: Annotated[int, Query(ge=1, le=20)] = 20,
    wf_type: Annotated[str | None, Query(pattern='^(all|0|1)$')] = None,
) -> list[WorkflowSearchResult]:
    """Search visible workflow descriptions; omitted type retains the subgraph default."""
    if wf_type is None:
        selected_type = WorkflowType.SUBMODULE
    elif wf_type == 'all':
        selected_type = None
    else:
        selected_type = WorkflowType(int(wf_type))
    try:
        matches = await workflow_catalog.search_workflows(
            session,
            q,
            current_user.id,
            include_private=current_user.role == UserRole.ADMIN,
            include_owned=current_user.role >= UserRole.MEMBER,
            wf_type=selected_type,
            limit=limit,
        )
    except Exception as error:
        logger.exception('Workflow semantic search failed')
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, 'Semantic search is temporarily unavailable'
        ) from error
    return [
        WorkflowSearchResult.model_validate({**wf_crud.to_simple(workflow), 'relevance_score': score})
        for workflow, score in matches
    ]


@router.get('/{uid}')
async def get_workflow(uid: str, session: SessionDep, current_user: UserDep) -> WorkflowPublic:
    """Get a workflow by its UID."""
    workflow = _get_workflow_for_user(session, uid, current_user)
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

    if workflow_prop.wf_type == WorkflowType.SUBMODULE and not workflow_prop.description.strip():
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, 'Subgraphs require a description')
    result = validate_workflow(
        resolve_workflow(
            {**workflow_prop.workflow.model_dump(mode='json'), 'wf_type': int(workflow_prop.wf_type)},
            session,
            fresh=True,
            execution_scope=cast('ExecutionScopeName', workflow_prop.execution_scope.name.lower()),
        )
    )
    if not result.valid:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, result.message)

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
    current_user: MemberDep,
) -> None:
    """Delete a workflow by its UID."""
    _get_workflow_for_user(session, uid, current_user, edit=True)
    try:
        await workflow_catalog.delete_workflow(session, uid)
    except workflow_catalog.WorkflowCatalogNotFoundError as error:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(error)) from error


@router.patch('/{uid}')
async def update_workflow(
    uid: str,
    data: WorkflowUpdate,
    session: SessionDep,
    current_user: MemberDep,
) -> str:
    """Update an existing workflow's name and/or definition."""
    existing = _get_workflow_for_user(session, uid, current_user, edit=True)
    description = data.description if data.description is not None else existing.description
    if existing.wf_type == WorkflowType.SUBMODULE and not description.strip():
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, 'Subgraphs require a description')
    result = validate_workflow(
        resolve_workflow(
            {**(data.workflow or existing.workflow).model_dump(mode='json'), 'wf_type': int(existing.wf_type)},
            session,
            fresh=True,
            execution_scope=cast('ExecutionScopeName', (data.execution_scope or existing.execution_scope).name.lower()),
        )
    )
    if not result.valid:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, result.message)
    try:
        await workflow_catalog.update_workflow(session, uid, data)
    except workflow_catalog.WorkflowCatalogNotFoundError as error:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(error)) from error
    return uid
