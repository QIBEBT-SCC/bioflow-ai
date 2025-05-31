from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, Query, HTTPException
from fastapi.responses import FileResponse
from starlette import status
from uuid_extensions import uuid7

import app.crud.workflow as wf_crud
from app.core.session import SessionDep
from app.schemas.react_flow import WorkflowDefinition
from app.schemas.workflow import WorkflowCreateProp
from app.models.user import User
from app.core.config import get_settings
from app.models.workflow import Workflow
from app.utils.security import get_current_active_user

router = APIRouter()


@router.get('')
def get_workflows(
    session: SessionDep,
    current_user: Annotated[User, Depends(get_current_active_user)],
    offset: int = 0,
    limit: Annotated[int, Query(le=20)] = 20,
):
    return wf_crud.get_all(session, limit, offset, current_user.id)


@router.get('/count')
def get_workflows_count(
    session: SessionDep, current_user: Annotated[User, Depends(get_current_active_user)]
):
    return wf_crud.get_count(session, current_user.id)


@router.get('/{uid}')
def get_workflow(
    uid: str, session: SessionDep, user: Annotated[User, Depends(get_current_active_user)]
) -> Workflow:
    return wf_crud.get(session, uid)


@router.post('')
def save_workflow(
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
def update_workflow(
    uid: str,
    workflow: WorkflowDefinition,
    session: SessionDep,
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    # TODO: 合法性检查

    wf_crud.update(session, uid, workflow)
    return uid


@router.get('/{name}/export')
def export_workflow(name: str, user: Annotated[User, Depends(get_current_active_user)]):
    wf_store_path = Path(get_settings().persistent.STORAGE_PATH) / user.username / 'workflow'
    file_path = wf_store_path / f'{name}.json'
    return FileResponse(path=file_path, filename=f'{name}.json', media_type='application/json')
