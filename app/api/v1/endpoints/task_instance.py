from fastapi import APIRouter

import app.crud.task_instance as task_crud
from app.core.session import SessionDep
from app.models.workflow import TaskInstance, TaskUpdate

router = APIRouter()


@router.patch('/{uid}', response_model=TaskInstance)
def update_task(uid: str, task_update: TaskUpdate, session: SessionDep):
    return task_crud.update(session, uid, task_update)


@router.get('/{uid}/result')
def get_task_result(uid: str, session: SessionDep) -> dict[str, str]:
    task = task_crud.get(session, uid)
    return task.result
