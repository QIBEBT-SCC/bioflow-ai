from typing import Annotated

from fastapi import APIRouter, Depends, Query

import app.crud.run_instance as run_crud
from app.core.redis_client import monitor_redis_client
from app.core.session import SessionDep
from app.models.user import User
from app.models.workflow import RunInstance
from app.schemas.common import Statistics
from app.schemas.run_instance import RunPublic, RunUpdate
from app.utils.security import get_current_active_user
from app.utils.workflow import get_run_stats_from_redis

router = APIRouter()


@router.patch('/{uid}', response_model=RunInstance)
def update_workflow(uid: str, run_update: RunUpdate, session: SessionDep):
    return run_crud.update(session, uid, run_update)


@router.get('', response_model=list[RunPublic])
async def get_all_runs(
    session: SessionDep,
    current_user: Annotated[User, Depends(get_current_active_user)],
    offset: int = 0,
    limit: Annotated[int, Query(le=20)] = 20,
):
    """获取所有运行实例（包含从 Redis 读取的实时统计信息）"""
    runs = run_crud.get_all_runs(session, limit, offset, current_user.id)

    # 利用 Pydantic 自动转换 ORM 对象
    result = []
    for run in runs:
        # Pydantic 会自动处理 ORM -> Schema 的转换
        run_data = RunPublic.model_validate(run)

        # 尝试从 Redis 读取统计信息（覆盖 model_validator 计算的值）
        stats = await get_run_stats_from_redis(monitor_redis_client, str(run.uid))
        if stats:
            run_data.task_statistics = stats

        result.append(run_data)

    return result


@router.get('/count')
def get_run_counts(session: SessionDep, current_user: Annotated[User, Depends(get_current_active_user)]) -> int:
    return run_crud.get_run_counts(session, current_user.id)


@router.get('/statistics', response_model=Statistics)
def get_run_stats(session: SessionDep, current_user: Annotated[User, Depends(get_current_active_user)]):
    return run_crud.get_run_status(session, current_user.id)


@router.get('/{uid}', response_model=RunPublic)
async def get_run(uid: str, session: SessionDep, current_user: Annotated[User, Depends(get_current_active_user)]):
    """获取单个运行实例（包含从 Redis 读取的实时统计信息）"""
    run = run_crud.get(session, uid)

    # Pydantic 会自动处理 ORM -> Schema 的转换
    run_data = RunPublic.model_validate(run)

    # 尝试从 Redis 读取统计信息（覆盖 model_validator 计算的值）
    stats = await get_run_stats_from_redis(monitor_redis_client, uid)
    if stats:
        run_data.task_statistics = stats

    return run_data
