import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query
from starlette.status import HTTP_404_NOT_FOUND

import app.crud.run_instance as run_crud
from app.core.redis_client import monitor_redis_client
from app.core.session import SessionDep
from app.models.workflow import RunInstance
from app.schemas.common import Statistics
from app.schemas.run_instance import OutputFileNode, RunDetailPublic, RunPublic, RunUpdate
from app.utils.security import UserDep
from app.utils.system import build_file_tree
from app.utils.workflow import get_run_stats_from_redis

router = APIRouter()


@router.get('')
async def get_all_runs(
    session: SessionDep,
    current_user: UserDep,
    offset: int = 0,
    limit: Annotated[int, Query(le=20)] = 20,
) -> list[RunPublic]:
    """获取所有运行实例(包含从 Redis 读取的实时统计信息)."""
    runs = run_crud.get_all_runs(session, limit, offset, current_user.id)

    result = [RunPublic.model_validate(run) for run in runs]

    if result:
        pipe = monitor_redis_client.pipeline()
        for run in runs:
            pipe.hgetall(f'run-stats-{run.uid}')
        stats_list = await pipe.execute()

        for run_data, raw_stats in zip(result, stats_list, strict=True):
            if raw_stats:
                run_data.task_statistics = Statistics(
                    total=int(raw_stats.get(b'total', 0)),
                    waiting=int(raw_stats.get(b'waiting', 0)),
                    running=int(raw_stats.get(b'running', 0)),
                    success=int(raw_stats.get(b'success', 0)),
                    error=int(raw_stats.get(b'error', 0)),
                )

    return result


@router.get('/count')
def get_run_counts(session: SessionDep, current_user: UserDep) -> int:
    """Get the total number of run instances for the current user."""
    return run_crud.get_run_counts(session, current_user.id)


@router.get('/statistics')
def get_run_stats(session: SessionDep, current_user: UserDep) -> Statistics:
    """Get aggregated run status statistics for the current user."""
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
