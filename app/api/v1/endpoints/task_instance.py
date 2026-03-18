import json
from datetime import timedelta
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query

import app.crud.task_instance as task_crud
import app.crud.task_monitor as monitor_crud
from app.core.redis_client import monitor_redis_client
from app.core.session import SessionDep
from app.models.workflow import ToolOutput
from app.schemas.task_instance import MonitorPublic, SimpleTaskPublic, TaskPublic
from app.utils.security import UserDep

router = APIRouter()


@router.get('')
def get_all_tasks(
    session: SessionDep,
    current_user: UserDep,
    offset: int = 0,
    limit: Annotated[int, Query(le=20)] = 20,
) -> list[SimpleTaskPublic]:
    """Get a paginated list of task instances for the current user."""
    return task_crud.get_all(session, limit, offset, current_user.id)


@router.get('/count')
def get_task_count(session: SessionDep, current_user: UserDep) -> int:
    """Get the total count of task instances for the current user."""
    return task_crud.get_count(session, current_user.id)


@router.get('/recent/{hour}')
def get_by_time(hour: int, session: SessionDep, current_user: UserDep) -> list[SimpleTaskPublic]:
    """Get task instances created within the last N hours."""
    return task_crud.get_all_by_time(session, current_user.id, timedelta(hours=hour))


@router.get('/{uid}')
def get_task_instance(uid: str, session: SessionDep) -> TaskPublic:
    """Get a task instance by its UID."""
    return task_crud.get(session, uid)


@router.get('/{uid}/result')
def get_task_result(uid: str, session: SessionDep) -> ToolOutput:
    """Get the tool output result of a task instance."""
    task = task_crud.get(session, uid)
    return task.tool_output


@router.get('/{uid}/monitor')
async def get_monitor_log(
    uid: str,
    session: SessionDep,
    max_points: Annotated[int, Query(le=1000)] = 300,
) -> list[MonitorPublic]:
    """Get monitor log entries; reads from Redis if task is running, DB otherwise."""
    raw_list = await monitor_redis_client.lrange(f'task-monitor-{uid}', 0, -1)

    if raw_list:
        records: list = [MonitorPublic.model_validate(json.loads(r)) for r in raw_list]
    else:
        records = list(monitor_crud.get(session, uid))

    n = len(records)
    if n > max_points:
        step = n // max_points
        records = records[::step]

    return records


@router.get('/{uid}/log')
def get_task_log(uid: str, session: SessionDep) -> dict:
    """Get the text log file content for a task instance."""
    task = task_crud.get(session, uid)

    log_path = task.tool_output.log  # This will be replaced with actual log path
    if log_path and Path(log_path).exists():
        with Path(log_path).open('r', encoding='utf-8') as f:
            lines = f.read()

            return {'content': lines}
    else:
        raise HTTPException(status_code=404, detail='Log file not found')
