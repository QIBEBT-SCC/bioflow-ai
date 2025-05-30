import json
from datetime import timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, Query

import app.crud.task_instance as task_crud
import app.crud.task_monitor as monitor_crud
from app.core.redis_client import monitor_redis_client
from app.core.session import SessionDep
from app.models.user import User
from app.models.workflow import TaskInstance, TaskUpdate, TaskMonitor, MonitorPublic
from app.utils.container import calculate_container_stats
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


@router.get('/count')
def get_task_count(
    session: SessionDep, current_user: Annotated[User, Depends(get_current_active_user)]
) -> int:
    return task_crud.get_count(session, current_user.id)


@router.get('/recent/{hour}')
def get_by_time(
    hour: int, session: SessionDep, current_user: Annotated[User, Depends(get_current_active_user)]
) -> list[TaskInstance]:
    return task_crud.get_all_by_time(session, current_user.id, timedelta(hours=hour))


@router.patch('/{uid}', response_model=TaskInstance)
def update_task(uid: str, task_update: TaskUpdate, session: SessionDep):
    return task_crud.update(session, uid, task_update)


@router.get('/{uid}/result')
def get_task_result(uid: str, session: SessionDep) -> dict[str, str]:
    task = task_crud.get(session, uid)
    return task.result


@router.post('/{uid}/notify')
def read_resource_monitor(uid: str, session: SessionDep):
    key = f'task-monitor-{uid}'
    pipe = monitor_redis_client.pipeline()
    pipe.lrange(key, 0, -1)
    pipe.delete(key)
    logs, _ = pipe.execute()

    stats_list = [json.loads(log) for log in logs]

    monitors = []
    for stat in stats_list:
        stat_data = calculate_container_stats(stat)
        if stat_data:
            stat_data['task_uid'] = uid
            monitors.append(TaskMonitor.model_validate(stat_data))

    monitor_crud.insert_list(session, monitors)


@router.get('/{uid}/monitor', response_model=list[MonitorPublic])
def get_monitor_log(uid: str, session: SessionDep) -> list[MonitorPublic]:
    return monitor_crud.get(session, uid)
