from typing import TYPE_CHECKING, Annotated, cast

from anyio import Path as AsyncPath
from fastapi import APIRouter, HTTPException, Query
from starlette import status
from uuid_extensions import uuid7

import app.crud.run_instance as run_crud
import app.crud.workflow as wf_crud
from app.core.config import get_settings
from app.core.session import SessionDep
from app.models.workflow import RunInstance, Status, Workflow
from app.schemas.common import Paginated
from app.schemas.react_flow import WorkflowDefinition
from app.schemas.workflow import WorkflowCreateProp, WorkflowSimple, WorkflowUpdate
from app.utils.input_format import to_safe_path_segment
from app.utils.security import UserDep
from app.utils.workflow import build_task_list, send_task, update_node_id

if TYPE_CHECKING:
    from app.models.user import User

STORAGE_PATH = get_settings().persistent.storage.STORAGE_PATH

router = APIRouter()


@router.get('')
async def get_workflows(
    session: SessionDep,
    current_user: UserDep,
    offset: int = 0,
    limit: Annotated[int, Query(le=20)] = 20,
) -> Paginated[WorkflowSimple]:
    """Get a paginated list of workflows for the current user."""
    workflows, total = wf_crud.get_page(session, limit, offset, current_user.id)
    return Paginated(
        total=total,
        offset=offset,
        limit=limit,
        has_more=offset + limit < total,
        data=[WorkflowSimple.model_validate(wf) for wf in workflows],
    )


@router.post('/run', description='从editor新建工作流运行实例')
async def run_workflow(
    workflow: WorkflowDefinition,
    session: SessionDep,
    current_user: UserDep,
    template_name: str | None = None,
) -> RunInstance:
    """Create a new workflow run instance from the editor."""
    if template_name is None:
        template_name = 'unname'

    base_dir = (
        await AsyncPath(STORAGE_PATH).absolute()
        / f'{current_user.username}({current_user.id})'
        / 'temporarily'
        / to_safe_path_segment(template_name)
    )

    # 更新节点uid
    workflow = update_node_id(workflow)

    # 创建运行实例
    run = RunInstance(
        name=f'temporarily-{template_name}',
        project_id=None,
        # pyrefly: ignore [bad-argument-type]
        owner_id=current_user.id,
        workflow=workflow,
        base_dir=str(base_dir),
        status=Status.WAITING,
    )
    run = run_crud.insert(session, run)

    task_list = build_task_list(workflow)
    send_task(
        task_list,
        session,
        None,
        None,
        None,
        cast('User',current_user),
        f'{base_dir}',
        run.uid,
    )

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
        # pyrefly: ignore [bad-argument-type]
        uid=uid,
        name=workflow_prop.name,
        description='',
        owner_id=current_user.id,
        wf_type=workflow_prop.wf_type,
        execution_scope=workflow_prop.execution_scope,
        workflow=workflow_prop.workflow,
        public=workflow_prop.public,
        auto_summary=workflow_prop.auto_summary,
        summary_prompt=workflow_prop.summary_prompt,
    )

    wf_crud.insert(session, workflow)
    return uid  # type: ignore


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
