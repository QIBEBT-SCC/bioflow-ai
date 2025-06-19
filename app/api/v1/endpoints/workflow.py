from typing import Annotated, Optional

from fastapi import APIRouter, Depends, Query, HTTPException, Form
from starlette import status
from starlette.responses import StreamingResponse
from uuid_extensions import uuid7

import app.crud.workflow as wf_crud
import app.crud.run_instance as run_crud
from app.api.sse_base import SSEMessage, EventType
from app.core.session import SessionDep
from app.schemas.react_flow import WorkflowDefinition
from app.schemas.workflow import WorkflowCreateProp
from app.models.user import User
from app.models.workflow import Workflow, RunInstance, Status
from app.utils.input_format import get_sample_name
from app.utils.security import get_current_active_user
from app.utils.workflow import update_node_id, build_task_list, send_task

router = APIRouter()


@router.get('')
async def get_workflows(
    session: SessionDep,
    current_user: Annotated[User, Depends(get_current_active_user)],
    offset: int = 0,
    limit: Annotated[int, Query(le=20)] = 20,
):
    return wf_crud.get_all(session, limit, offset, current_user.id)


@router.get('/count')
async def get_workflows_count(session: SessionDep, current_user: Annotated[User, Depends(get_current_active_user)]):
    return wf_crud.get_count(session, current_user.id)


@router.post('/run', description='从editor新建工作流运行实例')
async def run_workflow(
    workflow: WorkflowDefinition,
    session: SessionDep,
    current_user: Annotated[User, Depends(get_current_active_user)],
    template_name: Optional[str] = None,
) -> RunInstance:
    if template_name is None:
        template_name = 'unname'

    # 检查合法
    # TODO

    # 更新节点uid
    workflow = update_node_id(workflow)

    # 创建运行实例
    run_uid = uuid7(as_type='str')
    run = RunInstance(uid=run_uid, project_id=None, owner_id=current_user.id, workflow=workflow, status=Status.WAITING)

    task_list = build_task_list(workflow)
    sample_name = get_sample_name(task_list[0])

    run.name = f'temporarily-{template_name}-{sample_name}'
    run = run_crud.insert(session, run)

    send_task(session, task_list, sample_name, f'temporarily/{template_name}-{sample_name}', run_uid, current_user)

    return run


@router.get('/{uid}')
async def get_workflow(
    uid: str, session: SessionDep, user: Annotated[User, Depends(get_current_active_user)]
) -> Workflow:
    return wf_crud.get(session, uid)


@router.post('')
async def save_workflow(
    workflow_prop: WorkflowCreateProp,
    session: SessionDep,
    current_user: Annotated[User, Depends(get_current_active_user)],
) -> str:
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


@router.patch('/{uid}')
async def update_workflow(
    uid: str,
    workflow: WorkflowDefinition,
    session: SessionDep,
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    # TODO: 合法性检查

    wf_crud.update(session, uid, workflow)
    return uid


@router.post('/reproduction')
async def reproduction_by_paper(
    session: SessionDep, current_user: Annotated[User, Depends(get_current_active_user)], paper_content: str = Form()
):
    async def generator():
        yield SSEMessage(event=EventType.LOADING, data='reading paper content...').to_sse()

    return StreamingResponse(generator(), media_type='text/event-stream')
