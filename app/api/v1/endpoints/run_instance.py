import asyncio
import uuid
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response, StreamingResponse
from sqlmodel import Session
from starlette.requests import Request
from starlette.status import HTTP_403_FORBIDDEN, HTTP_404_NOT_FOUND

import app.crud.run_instance as run_crud
from app.core.redis_client import monitor_redis_client
from app.core.session import SessionDep
from app.models.workflow import RunInstance, Status
from app.schemas.common import Paginated, Statistics
from app.schemas.run_instance import (
    FileContentRequest,
    OutputFileNode,
    RunDetailPublic,
    RunPublic,
    RunUpdate,
)
from app.utils.security import UserDep
from app.utils.system import build_file_tree, serve_file
from app.utils.workflow import get_run_stats_from_redis

router = APIRouter()


@router.get('')
async def get_all_runs(
    session: SessionDep,
    current_user: UserDep,
    offset: int = 0,
    limit: Annotated[int, Query(le=20)] = 20,
) -> Paginated[RunPublic]:
    """获取所有运行实例(包含从 Redis 读取的实时统计信息)."""
    runs, total = run_crud.get_runs_page(session, limit, offset, current_user.id)

    result = [RunPublic.model_validate(run) for run in runs]

    if result:
        pipe = monitor_redis_client.pipeline()
        for run in runs:
            pipe.hgetall(f'run-stats-{run.uid}')
        stats_list = await pipe.execute()

        for run_data, raw_stats in zip(result, stats_list, strict=True):
            if raw_stats and isinstance(raw_stats, dict):
                run_data.task_statistics = Statistics(
                    total=int(raw_stats.get(b'total', 0)),
                    waiting=int(raw_stats.get(b'waiting', 0)),
                    running=int(raw_stats.get(b'running', 0)),
                    success=int(raw_stats.get(b'success', 0)),
                    error=int(raw_stats.get(b'error', 0)),
                )

    return Paginated(
        total=total,
        offset=offset,
        limit=limit,
        has_more=offset + limit < total,
        data=result,
    )


@router.get('/statistics')
def get_run_stats(session: SessionDep, current_user: UserDep) -> Statistics:
    """Get aggregated run status statistics for the current user."""
    # pyrefly: ignore [bad-argument-type]
    return run_crud.get_run_status(session, current_user.id)


@router.patch('/{uid}')
def update_workflow(uid: str, run_update: RunUpdate, session: SessionDep) -> RunInstance:
    """Update a run instance's status or metadata."""
    return run_crud.update(session, uid, run_update)


@router.get('/{uid}')
async def get_run(uid: str, session: SessionDep, _current_user: UserDep) -> RunDetailPublic:
    """获取单个运行实例, 返回 workflow 图结构并注入运行状态."""
    run = run_crud.get(session, uid)

    run_detail = RunDetailPublic.model_validate(run)

    # 尝试从 Redis 读取统计信息(覆盖 model_validator 计算的值)
    stats = await get_run_stats_from_redis(monitor_redis_client, uid)
    if stats:
        run_detail.task_statistics = stats

    return run_detail


async def _stream_run_updates(
    uid: str,
    session: Session,
    request: Request,
) -> AsyncGenerator[str, None]:
    """Yield SSE data lines for a run until terminal state or client disconnect."""
    terminal_states = {Status.ERROR, Status.SUCCESS}

    while True:
        if await request.is_disconnected():
            break

        session.expire_all()  # Force fresh DB read each iteration

        run = run_crud.get(session, uid)
        run_data = RunDetailPublic.model_validate(run)

        stats = await get_run_stats_from_redis(monitor_redis_client, uid)
        if stats:
            run_data.task_statistics = stats

        yield f'data: {run_data.model_dump_json()}\n\n'

        if run_data.status in terminal_states:
            break

        await asyncio.sleep(2)


@router.get('/{uid}/stream')
async def stream_run(
    uid: str,
    request: Request,
    session: SessionDep,
    _current_user: UserDep,
) -> StreamingResponse:
    """Stream run status updates via SSE until the run reaches a terminal state."""
    return StreamingResponse(
        _stream_run_updates(uid, session, request),
        media_type='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
        },
    )


@router.get('/{uid}/files')
async def get_run_output_files(
    uid: uuid.UUID,
    session: SessionDep,
    _current_user: UserDep,
) -> list[OutputFileNode]:
    """Get the file tree of the run instance output directory.

    Args:
        uid: Run instance UID

    Returns:
        list[OutputFileNode]: Nested file tree of the run output directory.
    """
    run = run_crud.get(session, str(uid))
    if not run:
        raise HTTPException(HTTP_404_NOT_FOUND, 'Run not found in this project')

    if not run.base_dir or not Path(run.base_dir).exists():
        return []

    return build_file_tree(Path(run.base_dir), Path(run.base_dir))


@router.post('/{uid}/files/content')
async def get_run_output_file_content(
    uid: uuid.UUID,
    body: FileContentRequest,
    session: SessionDep,
    _current_user: UserDep,
) -> Response:
    """Serve the content of a single file from the run instance output directory.

    Args:
        uid: Run instance UID
        body: Relative path from run base_dir (as returned by the /{uid}/files endpoint)

    Returns:
        Response: File content with appropriate Content-Type header.
            - Text files (txt, log, csv, tsv, json, html, fasta, etc.) are returned inline.
            - Binary files (images, PDF, etc.) are streamed with their native MIME type.
            - Unknown binary files are returned as application/octet-stream with
              Content-Disposition: attachment.
    """
    run = run_crud.get(session, str(uid))
    if not run:
        raise HTTPException(HTTP_404_NOT_FOUND, 'Run not found')

    if not run.base_dir:
        raise HTTPException(HTTP_404_NOT_FOUND, 'Run has no output directory')

    base = Path(run.base_dir).resolve()
    full_path = (base / body.path).resolve()

    if not full_path.is_relative_to(base):
        raise HTTPException(HTTP_403_FORBIDDEN, 'Access denied')

    if not full_path.exists() or not full_path.is_file():
        raise HTTPException(HTTP_404_NOT_FOUND, 'File not found')

    return serve_file(full_path)
