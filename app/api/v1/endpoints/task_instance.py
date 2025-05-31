import json
from datetime import timedelta
from typing import Annotated
from pathlib import Path

from fastapi import APIRouter, Depends, Query, HTTPException

import app.crud.task_instance as task_crud
import app.crud.task_monitor as monitor_crud
from app.core.redis_client import monitor_redis_client
from app.core.session import SessionDep
from app.models.user import User
from app.models.workflow import TaskInstance, ToolOutput, TaskMonitor
from app.schemas.task_instance import SimpleTaskPublic, TaskPublic, TaskUpdate, MonitorPublic
from app.utils.container import calculate_container_stats
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
def update_task(uid: str, task_update: TaskUpdate, session: SessionDep):
    return task_crud.update(session, uid, task_update)


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
    for stat in stats_list:
        stat_data = calculate_container_stats(stat)
        if stat_data:
            stat_data['task_uid'] = uid
            monitors.append(TaskMonitor.model_validate(stat_data))

    monitor_crud.insert_list(session, monitors)


@router.get('/{uid}/monitor', response_model=list[MonitorPublic])
def get_monitor_log(uid: str, session: SessionDep) -> list[MonitorPublic]:
    return monitor_crud.get(session, uid)


@router.get('/{uid}/log')
def get_task_log(
    uid: str,
    session: SessionDep,
    start_line: int = Query(0, ge=0, description='Start line number (0-based)'),
    limit: int = Query(100, ge=1, le=1000, description='Number of lines to return'),
) -> dict:
    """
    Get the log content for a specific task with pagination support.
    """
    # Verify task exists
    task = task_crud.get(session, uid)
    if not task:
        raise HTTPException(status_code=404, detail='Task not found')

    # Read log file
    log_path = Path('test.log')  # This will be replaced with actual log path
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            # Skip to start_line
            for _ in range(start_line):
                if not f.readline():
                    return {'content': [], 'total_lines': start_line, 'has_more': False}

            # Read requested number of lines
            lines = []
            for _ in range(limit):
                line = f.readline()
                if not line:
                    break
                lines.append(line.rstrip('\n'))

            # Check if there are more lines
            has_more = bool(f.readline())

            # Count total lines if needed
            if start_line == 0:
                with open(log_path, 'r', encoding='utf-8') as f:
                    total_lines = sum(1 for _ in f)
            else:
                total_lines = start_line + len(lines)

            return {
                'content': lines,
                'total_lines': total_lines,
                'has_more': has_more,
                'start_line': start_line,
                'end_line': start_line + len(lines) - 1,
            }

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail='Log file not found')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Error reading log file: {str(e)}')
