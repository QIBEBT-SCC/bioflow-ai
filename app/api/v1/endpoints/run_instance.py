from fastapi import APIRouter

import app.crud.run_instance as run_crud
from app.core.session import SessionDep
from app.models.workflow import RunInstance
from app.schemas.run_instance import RunUpdate

router = APIRouter()


@router.patch('/{uid}', response_model=RunInstance)
def update_workflow(uid: str, run_update: RunUpdate, session: SessionDep):
    return run_crud.update(session, uid, run_update)
