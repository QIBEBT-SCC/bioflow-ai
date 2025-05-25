from typing import Annotated

from fastapi import APIRouter, Depends, Query

import app.crud.task_instance as task_crud
from app.core.session import SessionDep
from app.models.user import User
from app.models.workflow import TaskInstance, TaskUpdate
from app.utils.security import get_current_active_user

router = APIRouter()


@router.get('')
def get_all_tasks(
    session: SessionDep,
    current_user: Annotated[User, Depends(get_current_active_user)],
    offset: int = 0,
    limit: Annotated[int, Query(le=20)] = 20,
) -> list[TaskInstance]:
    return task_crud.get_all(session, limit, offset, current_user.id)


@router.patch('/{uid}', response_model=TaskInstance)
def update_task(uid: str, task_update: TaskUpdate, session: SessionDep):
    return task_crud.update(session, uid, task_update)


@router.get('/{uid}/result')
def get_task_result(uid: str, session: SessionDep) -> dict[str, str]:
    task = task_crud.get(session, uid)
    return task.result
