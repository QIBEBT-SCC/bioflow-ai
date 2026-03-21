from typing import Annotated, Optional

from fastapi import APIRouter, HTTPException, Query
from starlette import status
from uuid_extensions import uuid7

import app.crud.run_instance as run_crud
import app.crud.workflow as wf_crud
from app.core.session import SessionDep
from app.models.workflow import RunInstance, Status, Workflow
from app.schemas.react_flow import WorkflowDefinition
from app.schemas.workflow import WorkflowCreateProp, WorkflowUpdate
from app.utils.input_format import get_sample_name
from app.utils.security import UserDep
from app.utils.workflow import build_task_list, send_task, update_node_id

router = APIRouter()


@router.get('')
async def get_workflows(
    session: SessionDep,
    current_user: UserDep,
    offset: int = 0,
    limit: Annotated[int, Query(le=20)] = 20,
) -> list[Workflow]:
    """Get a paginated list of workflows for the current user."""
    return wf_crud.get_all(session, limit, offset, current_user.id)


@router.get('/count')
async def get_workflows_count(session: SessionDep, current_user: UserDep) -> int:
    """Get the total count of workflows for the current user."""
    return wf_crud.get_count(session, current_user.id)


@router.post('/run', description='从editor新建工作流运行实例')
async def run_workflow(
    workflow: WorkflowDefinition,
    session: SessionDep,
    current_user: UserDep,
    template_name: Optional[str] = None,
) -> RunInstance:
    """Create a new workflow run instance from the editor."""
    if template_name is None:
        template_name = 'unname'

    # TODO: 合法性检查

    # 更新节点uid
    workflow = update_node_id(workflow)

    # 创建运行实例
    run = RunInstance(project_id=None, owner_id=current_user.id, workflow=workflow, status=Status.WAITING)

    task_list = build_task_list(workflow)
    sample_name = get_sample_name(task_list[0])

    run.name = f'temporarily-{template_name}-{sample_name}'
    run = run_crud.insert(session, run)

    send_task(session, task_list, sample_name, f'temporarily/{template_name}-{sample_name}', run.uid, current_user)

    return run


@router.get('/{uid}')
async def get_workflow(uid: str, session: SessionDep, _current_user: UserDep) -> Workflow:
    """Get a workflow by its UID."""
    return wf_crud.get(session, uid)


@router.post('')
async def save_workflow(
    workflow_prop: WorkflowCreateProp,
    session: SessionDep,
    current_user: UserDep,
) -> str:
    """Save a new workflow template."""
    if wf_crud.workflow_name_exists(session, workflow_prop.name):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail='name already exists!')

    # TODO: 合法性检查

    uid = uuid7(as_type='str')
    workflow = Workflow(
        uid=uid,
        name=workflow_prop.name,
        owner_id=current_user.id,
        wf_type=workflow_prop.wf_type,
        workflow=workflow_prop.workflow,
        public=workflow_prop.public,
    )

    wf_crud.insert(session, workflow)
    return uid


@router.delete('/{uid}', status_code=status.HTTP_204_NO_CONTENT)
async def delete_workflow(
    uid: str,
    session: SessionDep,
    _current_user: UserDep,
) -> None:
    """Delete a workflow by its UID."""
    wf_crud.delete(session, uid)


@router.patch('/{uid}')
async def update_workflow(
    uid: str,
    data: WorkflowUpdate,
    session: SessionDep,
    _current_user: UserDep,
) -> str:
    """Update an existing workflow's name and/or definition."""
    wf_crud.update(session, uid, data)
    return uid
