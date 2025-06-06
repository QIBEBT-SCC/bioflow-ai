from typing import Annotated

from fastapi import APIRouter, Depends, Query

import app.crud.run_instance as run_crud
from app.core.session import SessionDep
from app.models.user import User
from app.models.workflow import RunInstance
from app.schemas.run_instance import RunUpdate, RunPublic, Stat
from app.utils.security import get_current_active_user

router = APIRouter()


@router.patch('/{uid}', response_model=RunInstance)
def update_workflow(uid: str, run_update: RunUpdate, session: SessionDep):
    return run_crud.update(session, uid, run_update)


@router.get('/count')
def get_run_counts(session: SessionDep, current_user: Annotated[User, Depends(get_current_active_user)]) -> int:
    return run_crud.get_run_counts(session, current_user.id)


@router.get('/stats', response_model=Stat)
def get_run_stats(session: SessionDep, current_user: Annotated[User, Depends(get_current_active_user)]):
    return run_crud.get_run_status(session, current_user.id)


@router.get('', response_model=list[RunPublic])
def get_all_runs(
    session: SessionDep,
    current_user: Annotated[User, Depends(get_current_active_user)],
    offset: int = 0,
    limit: Annotated[int, Query(le=20)] = 20,
) -> list[RunPublic]:
    return run_crud.get_all_runs(session, limit, offset, current_user.id)
