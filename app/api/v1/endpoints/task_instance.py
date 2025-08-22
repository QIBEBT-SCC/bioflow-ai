import json
from datetime import datetime, timedelta
from typing import Annotated
from pathlib import Path

from fastapi import APIRouter, Depends, Query, HTTPException

import app.crud.run_instance as run_crud
import app.crud.task_instance as task_crud
import app.crud.task_monitor as monitor_crud
from app.core.redis_client import monitor_redis_client
from app.core.session import SessionDep
from app.models.user import User
from app.models.workflow import TaskInstance, ToolOutput, TaskMonitor, Status
from app.schemas.run_instance import Stat, RunUpdate
from app.schemas.task_instance import SimpleTaskPublic, TaskPublic, TaskUpdate, MonitorPublic
from app.utils.security import get_current_active_user

router = APIRouter()


@router.get('', response_model=list[SimpleTaskPublic])
def get_all_tasks(
    session: SessionDep,
    current_user: Annotated[User, Depends(get_current_active_user)],
    offset: int = 0,
    limit: Annotated[int, Query(le=20)] = 20,
) -> list[SimpleTaskPublic]:
    return task_crud.get_all(session, limit, offset, current_user.id)


@router.get('/count')
def get_task_count(session: SessionDep, current_user: Annotated[User, Depends(get_current_active_user)]) -> int:
    return task_crud.get_count(session, current_user.id)


@router.get('/recent/{hour}', response_model=list[SimpleTaskPublic])
def get_by_time(
    hour: int, session: SessionDep, current_user: Annotated[User, Depends(get_current_active_user)]
) -> list[SimpleTaskPublic]:
    return task_crud.get_all_by_time(session, current_user.id, timedelta(hours=hour))


@router.patch('/{uid}', response_model=TaskInstance)
async def update_task(uid: str, task_update: TaskUpdate, session: SessionDep):
    updated_task = task_crud.update(session, uid, task_update)

    run_uid = updated_task.instance_uid
    run_cache = monitor_redis_client.get(f'run-stats-{run_uid}')
    if run_cache:
        run_cache_data = Stat.model_validate_json(run_cache.decode('utf-8'))
        match task_update.status:
            case Status.RUNNING:
                run_cache_data.add_running()
                if run_cache_data.running == 1:
                    run_crud.update(
                        session,
                        run_uid,
                        RunUpdate(
                            status=Status.RUNNING,
                            start_time=task_update.start_time if task_update.start_time else datetime.now(),
                        ),
                    )
            case Status.SUCCESS:
                run_cache_data.add_success()
                if run_cache_data.success == run_cache_data.total:
                    run_crud.update(
                        session,
                        run_uid,
                        RunUpdate(
                            status=Status.SUCCESS,
                            end_time=task_update.end_time if task_update.end_time else datetime.now(),
                        ),
                    )
            case Status.ERROR:
                run_cache_data.add_error()
                run_crud.update(
                    session,
                    run_uid,
                    RunUpdate(
                        status=Status.ERROR, end_time=task_update.end_time if task_update.end_time else datetime.now()
                    ),
                )
    else:
        run_cache_data = run_crud.get_run_task_stats(session, run_uid)

    monitor_redis_client.set(f'run-stats-{run_uid}', run_cache_data.model_dump_json(), ex=86400)

    return updated_task


@router.get('/{uid}', response_model=TaskPublic)
def get_task_instance(uid: str, session: SessionDep) -> TaskPublic:
    return task_crud.get(session, uid)


@router.get('/{uid}/result')
def get_task_result(uid: str, session: SessionDep) -> ToolOutput:
    task = task_crud.get(session, uid)
    return task.tool_output


@router.post('/{uid}/notify')
def read_resource_monitor(uid: str, session: SessionDep):
    key = f'task-monitor-{uid}'
    pipe = monitor_redis_client.pipeline()
    pipe.lrange(key, 0, -1)
    pipe.delete(key)
    logs, _ = pipe.execute()

    stats_list = [json.loads(log) for log in logs]

    monitors = []
    for stat_data in stats_list:
        stat_data['task_uid'] = uid
        monitors.append(TaskMonitor.model_validate(stat_data))

    monitor_crud.insert_list(session, monitors)


@router.get('/{uid}/monitor', response_model=list[MonitorPublic])
def get_monitor_log(uid: str, session: SessionDep) -> list[MonitorPublic]:
    return monitor_crud.get(session, uid)


@router.get('/{uid}/log')
def get_task_log(uid: str, session: SessionDep) -> dict:
    task = task_crud.get(session, uid)

    log_path = task.tool_output.log  # This will be replaced with actual log path
    if log_path and Path(log_path).exists():
        with open(log_path, 'r', encoding='utf-8') as f:
            lines = f.read()

            return {'content': lines}
    else:
        raise HTTPException(status_code=404, detail='Log file not found')
