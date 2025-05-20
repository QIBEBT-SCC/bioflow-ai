from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, Query
from fastapi.responses import FileResponse
from uuid_extensions import uuid7

import app.crud.workflow as wf_crud
from app.core.session import SessionDep
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
    workflow: Workflow,
    session: SessionDep,
    current_user: Annotated[User, Depends(get_current_active_user)],
) -> Workflow:
    uid = uuid7(as_type='str')
    workflow.uid = uid
    workflow.owner_id = current_user.id

    return wf_crud.insert(session, workflow)


@router.get('/{name}/export')
def export_workflow(name: str, user: Annotated[User, Depends(get_current_active_user)]):
    wf_store_path = Path(get_settings().persistent.STORAGE_PATH) / user.username / 'workflow'
    file_path = wf_store_path / f'{name}.json'
    return FileResponse(path=file_path, filename=f'{name}.json', media_type='application/json')


@router.post('/flows/{name}/run')
def run_workflow():
    pass
