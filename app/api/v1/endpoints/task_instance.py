import asyncio
import json
from collections.abc import AsyncGenerator
from datetime import timedelta
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from starlette.requests import Request
from starlette.status import HTTP_404_NOT_FOUND

import app.crud.task_instance as task_crud
import app.crud.task_monitor as monitor_crud
from app.core.auth import UserDep
from app.core.redis_client import monitor_redis_client
from app.core.session import SessionDep
from app.schemas.task_instance import MonitorPublic, SimpleTaskPublic, TaskPublic
from app.schemas.user_config import UserRole
from app.schemas.workflow_config import Status, ToolOutput
from app.services import runtime_status

router = APIRouter()

IDLE_THRESHOLD = 3  # empty-read cycles before checking DB (~6s)


@router.get('')
def get_all_tasks(
    session: SessionDep,
    current_user: UserDep,
    offset: int = 0,
    limit: Annotated[int, Query(le=20)] = 20,
) -> list[SimpleTaskPublic]:
    """Get a paginated list of task instances for the current user."""
    # pyrefly: ignore [bad-argument-type, bad-return]
    owner_id = None if current_user.role == UserRole.ADMIN else current_user.id
    tasks = task_crud.get_all(session, limit, offset, owner_id)
    runtime_status.refresh_task_statuses(tasks)
    return [SimpleTaskPublic.model_validate(task) for task in tasks]


@router.get('/count')
def get_task_count(session: SessionDep, current_user: UserDep) -> int:
    """Get the total count of task instances for the current user."""
    # pyrefly: ignore [bad-argument-type]
    owner_id = None if current_user.role == UserRole.ADMIN else current_user.id
    return task_crud.get_count(session, owner_id)


@router.get('/recent/{hour}')
def get_by_time(hour: int, session: SessionDep, current_user: UserDep) -> list[SimpleTaskPublic]:
    """Get task instances created within the last N hours."""
    # pyrefly: ignore [bad-argument-type, bad-return]
    owner_id = None if current_user.role == UserRole.ADMIN else current_user.id
    tasks = task_crud.get_all_by_time(session, owner_id, timedelta(hours=hour))
    runtime_status.refresh_task_statuses(tasks)
    return [SimpleTaskPublic.model_validate(task) for task in tasks]


@router.get('/{uid}')
def get_task_instance(uid: str, session: SessionDep) -> TaskPublic:
    """Get a task instance by its UID."""
    task = task_crud.get(session, uid)
    if task is None:
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail='Task not found')
    return TaskPublic.model_validate(runtime_status.refresh_task_status(task))


@router.get('/{uid}/result')
def get_task_result(uid: str, session: SessionDep) -> ToolOutput:
    """Get the tool output result of a task instance."""
    task = task_crud.get(session, uid)
    if task is None:
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail='Task not found')
    if task.tool_output is None:
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail='Task has no output')
    return task.tool_output


@router.get('/{uid}/monitor')
async def get_monitor_log(
    uid: str,
    session: SessionDep,
    max_points: Annotated[int, Query(le=1000)] = 300,
) -> list[MonitorPublic]:
    """Get monitor log entries; reads from Redis if task is running, DB otherwise."""
    # pyrefly: ignore [not-async]
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
    if task is None:
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail='Task not found')
    if task.tool_output is None:
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail='Task has no output')

    # pyrefly: ignore [bad-argument-type]
    log_path = Path(task.tool_output.log) if hasattr(task.tool_output, 'log') else Path(task.base_dir) / '11_run.log'
    if Path(log_path).exists():
        with Path(log_path).open('r', encoding='utf-8') as f:
            lines = f.read()
            return {'content': lines}

    return {'content': ''}


async def _stream_task_log(uid: str, session: Session, request: Request) -> AsyncGenerator[str, None]:
    """Yield SSE data lines with incremental log content until task reaches terminal state.

    DB query strategy:
    - Query once at start to resolve log_path (re-poll if task hasn't started yet).
    - While the log file is actively producing content, skip DB queries entirely.
    - After log_path is resolved, use Celery backend state to stop on terminal status.
    """
    terminal_states = {Status.ERROR, Status.SUCCESS}

    file_offset = 0
    idle_cycles = 0
    log_path: Path | None = None
    db_status = Status.WAITING

    while True:
        if await request.is_disconnected():
            break

        # Phase 1 — resolve log_path; re-poll each cycle until task starts
        if log_path is None:
            session.expire_all()
            task = task_crud.get(session, uid)
            if task is None:
                break
            db_status = task.status
            if task.tool_output and hasattr(task.tool_output, 'log') and task.tool_output.log:
                log_path = Path(task.tool_output.log)
            elif task.base_dir:
                log_path = Path(task.base_dir) / '11_run.log'
            else:
                # Task hasn't started yet (no base_dir); check terminal and keep waiting
                if runtime_status.get_task_status(uid, task.status) in terminal_states:
                    break
                await asyncio.sleep(2)
                continue

        # Phase 2 — stream file content; query DB only on idle
        new_content = ''
        if log_path.exists():
            with log_path.open('r', encoding='utf-8') as f:
                f.seek(file_offset)
                new_content = f.read()
                file_offset = f.tell()

        if new_content:
            idle_cycles = 0
            yield f'data: {json.dumps({"content": new_content})}\n\n'
        else:
            idle_cycles += 1
            if idle_cycles >= IDLE_THRESHOLD:
                if runtime_status.get_task_status(uid, db_status) in terminal_states:
                    break
                idle_cycles = 0

        await asyncio.sleep(2)


@router.get('/{uid}/log/stream')
async def stream_task_log(
    uid: str,
    request: Request,
    session: SessionDep,
    _current_user: UserDep,
) -> StreamingResponse:
    """Stream task log content as SSE, sending only new lines on each tick."""
    if task_crud.get(session, uid) is None:
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail='Task not found')
    return StreamingResponse(
        _stream_task_log(uid, session, request),
        media_type='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
        },
    )
