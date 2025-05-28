from typing import Annotated, Optional

from fastapi import APIRouter, Depends
from uuid_extensions import uuid7

import app.crud.run_instance as run_crud
from app.core.session import SessionDep
from app.models.react_flow import WorkflowDefinition
from app.models.user import User
from app.models.workflow import RunInstance, Status, RunUpdate
from app.utils.security import get_current_active_user
from app.utils.workflow import build_task_list, send_task, update_node_id

router = APIRouter()


@router.post('', description='新建工作流运行实例')
def run_workflow(
    workflow: WorkflowDefinition,
    session: SessionDep,
    current_user: Annotated[User, Depends(get_current_active_user)],
    project_id: Optional[int] = None,
) -> RunInstance:
    # 检查合法
    # TODO

    # 更新节点uid
    workflow = update_node_id(workflow)

    # 创建运行实例
    run_uid = uuid7(as_type='str')
    run = RunInstance(
        uid=run_uid,
        project_id=project_id,
        owner_id=current_user.id,
        workflow=workflow,
        status=Status.WAITING,
    )
    run = run_crud.insert(session, run)

    task_list = build_task_list(workflow)
    send_task(session, task_list, 'HC23', run_uid, current_user)

    return run


@router.patch('/{uid}', response_model=RunInstance)
def update_workflow(uid: str, run_update: RunUpdate, session: SessionDep):
    return run_crud.update(session, uid, run_update)
